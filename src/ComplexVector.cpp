#include "ComplexVector.hpp"
#include "VectorOperations.hpp"
#include "StoragePaths.hpp"
#include <stdexcept>

namespace pycauset {

std::unique_ptr<ComplexVector> ComplexVector::conjugate(const std::string& saveas_real, const std::string& saveas_imag) const {
    // Real part is copied
    std::string r_path = real_->copy_storage(saveas_real);
    auto r_mapper = std::make_unique<MemoryMapper>(r_path, 0, false);
    auto new_real = std::make_unique<FloatVector>(n_, std::move(r_mapper));

    // Imag part is negated
    std::string i_path = imag_->copy_storage(saveas_imag);
    auto i_mapper = std::make_unique<MemoryMapper>(i_path, 0, false);
    auto new_imag = std::make_unique<FloatVector>(n_, std::move(i_mapper));
    
    // Negate imaginary part: new_imag = new_imag * -1.0
    // We can use scalar_multiply_vector, but that returns a new vector.
    // Or we can just iterate and negate since we own the memory now.
    // Or use set_scalar(-1.0) if the vector supports it? 
    // VectorBase has set_scalar, but that's a property of the object, not an operation on elements.
    // Wait, set_scalar sets the scalar multiplier for the whole vector.
    // So if we set scalar to -1.0, get() returns -val.
    // But the underlying data is unchanged.
    // If we want to persist the negation, we should probably do it properly.
    // But for now, setting the scalar property is efficient.
    new_imag->set_scalar(new_imag->get_scalar() * -1.0);

    return std::make_unique<ComplexVector>(std::move(new_real), std::move(new_imag));
}

std::unique_ptr<ComplexVector> add(const ComplexVector& a, const ComplexVector& b, const std::string& saveas_real, const std::string& saveas_imag) {
    if (a.size() != b.size()) throw std::invalid_argument("Vector sizes must match");

    // Real = a.real + b.real
    auto real_res_base = add_vectors(*a.real(), *b.real(), saveas_real);
    auto real_res = std::unique_ptr<FloatVector>(static_cast<FloatVector*>(real_res_base.release()));

    // Imag = a.imag + b.imag
    auto imag_res_base = add_vectors(*a.imag(), *b.imag(), saveas_imag);
    auto imag_res = std::unique_ptr<FloatVector>(static_cast<FloatVector*>(imag_res_base.release()));

    return std::make_unique<ComplexVector>(std::move(real_res), std::move(imag_res));
}

std::unique_ptr<ComplexVector> subtract(const ComplexVector& a, const ComplexVector& b, const std::string& saveas_real, const std::string& saveas_imag) {
    if (a.size() != b.size()) throw std::invalid_argument("Vector sizes must match");

    // Real = a.real - b.real
    auto real_res_base = subtract_vectors(*a.real(), *b.real(), saveas_real);
    auto real_res = std::unique_ptr<FloatVector>(static_cast<FloatVector*>(real_res_base.release()));

    // Imag = a.imag - b.imag
    auto imag_res_base = subtract_vectors(*a.imag(), *b.imag(), saveas_imag);
    auto imag_res = std::unique_ptr<FloatVector>(static_cast<FloatVector*>(imag_res_base.release()));

    return std::make_unique<ComplexVector>(std::move(real_res), std::move(imag_res));
}

std::complex<double> dot(const ComplexVector& a, const ComplexVector& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Vector sizes must match");
    
    // (ar + i*ai) . (br + i*bi) = sum( (ar*br - ai*bi) + i(ar*bi + ai*br) )
    // = sum(ar*br) - sum(ai*bi) + i( sum(ar*bi) + sum(ai*br) )
    
    double r1 = dot_product(*a.real(), *b.real());
    double r2 = dot_product(*a.imag(), *b.imag());
    
    double i1 = dot_product(*a.real(), *b.imag());
    double i2 = dot_product(*a.imag(), *b.real());
    
    return std::complex<double>(r1 - r2, i1 + i2);
}

std::unique_ptr<ComplexVector> cross(const ComplexVector& a, const ComplexVector& b, const std::string& saveas_real, const std::string& saveas_imag) {
    if (a.size() != 3 || b.size() != 3) throw std::invalid_argument("Cross product only defined for 3D vectors");

    // Cross product is bilinear.
    // (ar + i*ai) x (br + i*bi) = (ar x br) + i(ar x bi) + i(ai x br) - (ai x bi)
    // Real part: (ar x br) - (ai x bi)
    // Imag part: (ar x bi) + (ai x br)

    auto ar_x_br = cross_product(*a.real(), *b.real(), make_unique_storage_file("tmp_ar_br"));
    auto ai_x_bi = cross_product(*a.imag(), *b.imag(), make_unique_storage_file("tmp_ai_bi"));
    
    auto ar_x_bi = cross_product(*a.real(), *b.imag(), make_unique_storage_file("tmp_ar_bi"));
    auto ai_x_br = cross_product(*a.imag(), *b.real(), make_unique_storage_file("tmp_ai_br"));

    // Real = (ar x br) - (ai x bi)
    auto real_res_base = subtract_vectors(*ar_x_br, *ai_x_bi, saveas_real);
    auto real_res = std::unique_ptr<FloatVector>(static_cast<FloatVector*>(real_res_base.release()));

    // Imag = (ar x bi) + (ai x br)
    auto imag_res_base = add_vectors(*ar_x_bi, *ai_x_br, saveas_imag);
    auto imag_res = std::unique_ptr<FloatVector>(static_cast<FloatVector*>(imag_res_base.release()));

    return std::make_unique<ComplexVector>(std::move(real_res), std::move(imag_res));
}

std::unique_ptr<ComplexVector> multiply_scalar(const ComplexVector& v, std::complex<double> scalar, const std::string& saveas_real, const std::string& saveas_imag) {
    // (vr + i*vi) * (sr + i*si) = (vr*sr - vi*si) + i(vr*si + vi*sr)
    
    double sr = scalar.real();
    double si = scalar.imag();

    // Terms
    auto vr_sr = scalar_multiply_vector(*v.real(), sr, make_unique_storage_file("tmp_vr_sr"));
    auto vi_si = scalar_multiply_vector(*v.imag(), si, make_unique_storage_file("tmp_vi_si"));
    
    auto vr_si = scalar_multiply_vector(*v.real(), si, make_unique_storage_file("tmp_vr_si"));
    auto vi_sr = scalar_multiply_vector(*v.imag(), sr, make_unique_storage_file("tmp_vi_sr"));

    // Real = vr*sr - vi*si
    auto real_res_base = subtract_vectors(*vr_sr, *vi_si, saveas_real);
    auto real_res = std::unique_ptr<FloatVector>(static_cast<FloatVector*>(real_res_base.release()));

    // Imag = vr*si + vi*sr
    auto imag_res_base = add_vectors(*vr_si, *vi_sr, saveas_imag);
    auto imag_res = std::unique_ptr<FloatVector>(static_cast<FloatVector*>(imag_res_base.release()));

    return std::make_unique<ComplexVector>(std::move(real_res), std::move(imag_res));
}

std::unique_ptr<ComplexVector> add_scalar(const ComplexVector& v, std::complex<double> scalar, const std::string& saveas_real, const std::string& saveas_imag) {
    // (vr + i*vi) + (sr + i*si) = (vr+sr) + i(vi+si)
    
    auto real_res_base = scalar_add_vector(*v.real(), scalar.real(), saveas_real);
    auto real_res = std::unique_ptr<FloatVector>(static_cast<FloatVector*>(real_res_base.release()));

    auto imag_res_base = scalar_add_vector(*v.imag(), scalar.imag(), saveas_imag);
    auto imag_res = std::unique_ptr<FloatVector>(static_cast<FloatVector*>(imag_res_base.release()));

    return std::make_unique<ComplexVector>(std::move(real_res), std::move(imag_res));
}

}
