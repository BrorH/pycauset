#include "VectorOperations.hpp"
#include "DenseVector.hpp"
#include "UnitVector.hpp"
#include "ComputeContext.hpp"
#include <stdexcept>
#include <iostream>

namespace pycauset {

std::unique_ptr<VectorBase> add_vectors(const VectorBase& a, const VectorBase& b, const std::string& result_file) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions must match");
    }
    uint64_t n = a.size();

    // Determine result type
    bool a_is_int = (dynamic_cast<const DenseVector<int32_t>*>(&a) != nullptr) || (dynamic_cast<const DenseVector<bool>*>(&a) != nullptr);
    bool b_is_int = (dynamic_cast<const DenseVector<int32_t>*>(&b) != nullptr) || (dynamic_cast<const DenseVector<bool>*>(&b) != nullptr);

    std::unique_ptr<VectorBase> result;
    if (a_is_int && b_is_int) {
        result = std::make_unique<DenseVector<int32_t>>(n, result_file);
    } else {
        result = std::make_unique<DenseVector<double>>(n, result_file);
    }

    ComputeContext::instance().get_device()->add_vector(a, b, *result);
    return result;
}

std::unique_ptr<VectorBase> subtract_vectors(const VectorBase& a, const VectorBase& b, const std::string& result_file) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions must match");
    }
    uint64_t n = a.size();

    bool a_is_int = (dynamic_cast<const DenseVector<int32_t>*>(&a) != nullptr) || (dynamic_cast<const DenseVector<bool>*>(&a) != nullptr);
    bool b_is_int = (dynamic_cast<const DenseVector<int32_t>*>(&b) != nullptr) || (dynamic_cast<const DenseVector<bool>*>(&b) != nullptr);

    std::unique_ptr<VectorBase> result;
    if (a_is_int && b_is_int) {
        result = std::make_unique<DenseVector<int32_t>>(n, result_file);
    } else {
        result = std::make_unique<DenseVector<double>>(n, result_file);
    }

    ComputeContext::instance().get_device()->subtract_vector(a, b, *result);
    return result;
}

double dot_product(const VectorBase& a, const VectorBase& b) {
    return ComputeContext::instance().get_device()->dot(a, b);
}

std::unique_ptr<VectorBase> cross_product(const VectorBase& a, const VectorBase& b, const std::string& result_file) {
    if (a.size() != 3 || b.size() != 3) {
        throw std::invalid_argument("Cross product only defined for 3D vectors");
    }
    
    // For now, implement on CPU directly as it's small
    // TODO: Move to ComputeDevice if we need batched cross products
    
    auto* a_dbl = dynamic_cast<const DenseVector<double>*>(&a);
    auto* b_dbl = dynamic_cast<const DenseVector<double>*>(&b);
    
    // Fallback for non-double vectors (convert elements)
    double ax, ay, az, bx, by, bz;
    
    if (a_dbl) {
        const double* d = a_dbl->data();
        ax = d[0]; ay = d[1]; az = d[2];
    } else {
        ax = a.get_element_as_double(0);
        ay = a.get_element_as_double(1);
        az = a.get_element_as_double(2);
    }
    
    if (b_dbl) {
        const double* d = b_dbl->data();
        bx = d[0]; by = d[1]; bz = d[2];
    } else {
        bx = b.get_element_as_double(0);
        by = b.get_element_as_double(1);
        bz = b.get_element_as_double(2);
    }
    
    auto result = std::make_unique<DenseVector<double>>(3, result_file);
    double* r = result->data();
    
    r[0] = ay*bz - az*by;
    r[1] = az*bx - ax*bz;
    r[2] = ax*by - ay*bx;
    
    return result;
}

std::unique_ptr<VectorBase> scalar_multiply_vector(const VectorBase& v, double scalar, const std::string& result_file) {
    uint64_t n = v.size();
    auto result = std::make_unique<DenseVector<double>>(n, result_file);
    ComputeContext::instance().get_device()->scalar_multiply_vector(v, scalar, *result);
    return result;
}

std::unique_ptr<VectorBase> scalar_multiply_vector(const VectorBase& v, int64_t scalar, const std::string& result_file) {
    return scalar_multiply_vector(v, (double)scalar, result_file);
}

std::unique_ptr<VectorBase> scalar_add_vector(const VectorBase& v, double scalar, const std::string& result_file) {
    uint64_t n = v.size();
    auto result = std::make_unique<DenseVector<double>>(n, result_file);
    ComputeContext::instance().get_device()->scalar_add_vector(v, scalar, *result);
    return result;
}

std::unique_ptr<VectorBase> scalar_add_vector(const VectorBase& v, int64_t scalar, const std::string& result_file) {
    return scalar_add_vector(v, (double)scalar, result_file);
}

} // namespace pycauset
