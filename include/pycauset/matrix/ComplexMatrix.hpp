#pragma once

#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include <complex>
#include <memory>
#include <string>

namespace pycauset {

using FloatMatrix = DenseMatrix<double>;

class ComplexMatrix {
public:
    // Create new complex matrix (creates two backing files)
    ComplexMatrix(uint64_t n, const std::string& backing_file_real = "", const std::string& backing_file_imag = "") 
        : n_(n) {
        real_ = std::make_unique<FloatMatrix>(n, backing_file_real);
        imag_ = std::make_unique<FloatMatrix>(n, backing_file_imag);
    }

    // Constructor for loading from offsets
    ComplexMatrix(uint64_t n, 
                  const std::string& backing_file_real, size_t offset_real,
                  const std::string& backing_file_imag, size_t offset_imag) 
        : n_(n) {
        // Calculate size in bytes: n * n * sizeof(double)
        size_t size_bytes = n * n * sizeof(double);
        
        auto mapper_real = std::make_unique<MemoryMapper>(backing_file_real, size_bytes, offset_real, false);
        real_ = std::make_unique<FloatMatrix>(n, std::move(mapper_real));
        
        auto mapper_imag = std::make_unique<MemoryMapper>(backing_file_imag, size_bytes, offset_imag, false);
        imag_ = std::make_unique<FloatMatrix>(n, std::move(mapper_imag));
    }

    // Construct from existing parts (takes ownership)
    ComplexMatrix(std::unique_ptr<FloatMatrix> real, std::unique_ptr<FloatMatrix> imag)
        : n_(real->size()), real_(std::move(real)), imag_(std::move(imag)) {
        if (n_ != imag_->size()) throw std::invalid_argument("Real and Imaginary parts must have same size");
    }

    uint64_t size() const { return n_; }

    std::complex<double> get(uint64_t i, uint64_t j) const {
        return std::complex<double>(real_->get_element_as_double(i, j), imag_->get_element_as_double(i, j));
    }

    void set(uint64_t i, uint64_t j, std::complex<double> val) {
        real_->set(i, j, val.real());
        imag_->set(i, j, val.imag());
    }

    FloatMatrix* real() { return real_.get(); }
    const FloatMatrix* real() const { return real_.get(); }
    
    FloatMatrix* imag() { return imag_.get(); }
    const FloatMatrix* imag() const { return imag_.get(); }

    // Conjugate: Returns new ComplexMatrix with same real part and negated imaginary part
    std::unique_ptr<ComplexMatrix> conjugate(const std::string& saveas_real = "", const std::string& saveas_imag = "") const {
        // Copy real part
        std::string r_path = real_->copy_storage(saveas_real);
        auto r_mapper = std::make_unique<MemoryMapper>(r_path, 0, false);
        auto new_real = std::make_unique<FloatMatrix>(n_, std::move(r_mapper));

        // Copy imag part and negate scalar
        std::string i_path = imag_->copy_storage(saveas_imag);
        auto i_mapper = std::make_unique<MemoryMapper>(i_path, 0, false);
        auto new_imag = std::make_unique<FloatMatrix>(n_, std::move(i_mapper));
        
        new_imag->set_scalar(new_imag->get_scalar() * -1.0);

        return std::make_unique<ComplexMatrix>(std::move(new_real), std::move(new_imag));
    }

    // Transpose
    std::unique_ptr<ComplexMatrix> transpose(const std::string& saveas_real = "", const std::string& saveas_imag = "") const {
        auto new_real_base = real_->transpose(saveas_real);
        auto new_imag_base = imag_->transpose(saveas_imag);
        
        // Cast back to FloatMatrix. We know transpose returns the same type (DenseMatrix<T>) wrapped in MatrixBase.
        // But wait, transpose returns MatrixBase.
        // For DenseMatrix<double>, it returns DenseMatrix<double>.
        // We can use static_cast because we know the type.
        
        auto new_real = std::unique_ptr<FloatMatrix>(static_cast<FloatMatrix*>(new_real_base.release()));
        auto new_imag = std::unique_ptr<FloatMatrix>(static_cast<FloatMatrix*>(new_imag_base.release()));
        
        return std::make_unique<ComplexMatrix>(std::move(new_real), std::move(new_imag));
    }

    // Hermitian (Conjugate Transpose)
    std::unique_ptr<ComplexMatrix> hermitian(const std::string& saveas_real = "", const std::string& saveas_imag = "") const {
        // (A + iB)^H = A^T - iB^T
        auto new_real_base = real_->transpose(saveas_real);
        auto new_imag_base = imag_->transpose(saveas_imag);
        
        auto new_real = std::unique_ptr<FloatMatrix>(static_cast<FloatMatrix*>(new_real_base.release()));
        auto new_imag = std::unique_ptr<FloatMatrix>(static_cast<FloatMatrix*>(new_imag_base.release()));
        
        // Negate imaginary part
        new_imag->set_scalar(new_imag->get_scalar() * -1.0);
        
        return std::make_unique<ComplexMatrix>(std::move(new_real), std::move(new_imag));
    }


    void close() {
        real_->close();
        imag_->close();
    }

private:
    uint64_t n_;
    std::unique_ptr<FloatMatrix> real_;
    std::unique_ptr<FloatMatrix> imag_;
};

// Operations
inline std::unique_ptr<ComplexMatrix> add(const ComplexMatrix& a, const ComplexMatrix& b, 
                                   const std::string& res_real = "", const std::string& res_imag = "") {
    // (a + bi) + (c + di) = (a+c) + (b+d)i
    auto r = pycauset::add(*a.real(), *b.real(), res_real);
    auto i = pycauset::add(*a.imag(), *b.imag(), res_imag);
    
    // We need to cast back to FloatMatrix because add returns MatrixBase
    // Since we know FloatMatrix + FloatMatrix -> FloatMatrix (or we cast)
    // Actually add returns MatrixBase. We need to ensure it's FloatMatrix.
    // The dispatch logic ensures Float+Float -> Float.
    
    // We need to cast unique_ptr<MatrixBase> to unique_ptr<FloatMatrix>
    // This requires releasing and casting.
    
    auto* r_ptr = dynamic_cast<FloatMatrix*>(r.release());
    auto* i_ptr = dynamic_cast<FloatMatrix*>(i.release());
    
    if (!r_ptr || !i_ptr) throw std::runtime_error("Complex addition failed to produce FloatMatrix components");
    
    return std::make_unique<ComplexMatrix>(std::unique_ptr<FloatMatrix>(r_ptr), std::unique_ptr<FloatMatrix>(i_ptr));
}

inline std::unique_ptr<ComplexMatrix> multiply(const ComplexMatrix& a, const ComplexMatrix& b,
                                        const std::string& res_real = "", const std::string& res_imag = "") {
    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    
    // Real part: ac - bd
    auto ac = a.real()->multiply(*b.real(), make_unique_storage_file("ac"));
    auto bd = a.imag()->multiply(*b.imag(), make_unique_storage_file("bd"));
    auto real_part = pycauset::subtract(*ac, *bd, res_real);
    
    // Imag part: ad + bc
    auto ad = a.real()->multiply(*b.imag(), make_unique_storage_file("ad"));
    auto bc = a.imag()->multiply(*b.real(), make_unique_storage_file("bc"));
    auto imag_part = pycauset::add(*ad, *bc, res_imag);

    auto* r_ptr = dynamic_cast<FloatMatrix*>(real_part.release());
    auto* i_ptr = dynamic_cast<FloatMatrix*>(imag_part.release());

    if (!r_ptr || !i_ptr) throw std::runtime_error("Complex multiplication failed");

    return std::make_unique<ComplexMatrix>(std::unique_ptr<FloatMatrix>(r_ptr), std::unique_ptr<FloatMatrix>(i_ptr));
}

inline std::unique_ptr<ComplexMatrix> multiply_scalar(const ComplexMatrix& m, std::complex<double> s,
                                               const std::string& res_real = "", const std::string& res_imag = "") {
    double a = s.real();
    double b = s.imag();
    
    // (R + iI) * (a + ib) = (Ra - Ib) + i(Rb + Ia)
    
    // Real part: Ra - Ib
    auto Ra = m.real()->multiply_scalar(a, make_unique_storage_file("Ra"));
    auto Ib = m.imag()->multiply_scalar(b, make_unique_storage_file("Ib"));
    auto real_part = pycauset::subtract(*Ra, *Ib, res_real);
    
    // Imag part: Rb + Ia
    auto Rb = m.real()->multiply_scalar(b, make_unique_storage_file("Rb"));
    auto Ia = m.imag()->multiply_scalar(a, make_unique_storage_file("Ia"));
    auto imag_part = pycauset::add(*Rb, *Ia, res_imag);
    
    auto* r_ptr = dynamic_cast<FloatMatrix*>(real_part.release());
    auto* i_ptr = dynamic_cast<FloatMatrix*>(imag_part.release());
    
    if (!r_ptr || !i_ptr) throw std::runtime_error("Complex scalar multiplication failed");
    
    return std::make_unique<ComplexMatrix>(std::unique_ptr<FloatMatrix>(r_ptr), std::unique_ptr<FloatMatrix>(i_ptr));
}

inline std::unique_ptr<ComplexMatrix> add_scalar(const ComplexMatrix& m, std::complex<double> s,
                                          const std::string& res_real = "", const std::string& res_imag = "") {
    double a = s.real();
    double b = s.imag();
    
    // (R + iI) + (a + ib) = (R + a) + i(I + b)
    
    auto real_part = m.real()->add_scalar(a, res_real);
    auto imag_part = m.imag()->add_scalar(b, res_imag);
    
    auto* r_ptr = dynamic_cast<FloatMatrix*>(real_part.release());
    auto* i_ptr = dynamic_cast<FloatMatrix*>(imag_part.release());
    
    if (!r_ptr || !i_ptr) throw std::runtime_error("Complex scalar addition failed");
    
    return std::make_unique<ComplexMatrix>(std::unique_ptr<FloatMatrix>(r_ptr), std::unique_ptr<FloatMatrix>(i_ptr));
}

}
