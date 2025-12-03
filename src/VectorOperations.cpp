#include "VectorOperations.hpp"
#include "DenseVector.hpp"
#include "UnitVector.hpp"
#include <stdexcept>
#include <iostream>

namespace pycauset {

std::unique_ptr<VectorBase> add_vectors(const VectorBase& a, const VectorBase& b, const std::string& result_file) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions must match");
    }
    uint64_t n = a.size();

    // Check for UnitVector optimization
    const auto* a_uv = dynamic_cast<const UnitVector*>(&a);
    const auto* b_uv = dynamic_cast<const UnitVector*>(&b);

    if (a_uv && b_uv) {
        if (a_uv->get_active_index() == b_uv->get_active_index()) {
            return a_uv->add(*b_uv, result_file);
        }
        // If indices differ, fall through to dense addition
    }
    
    // Check if both are integer-like (IntegerVector or BitVector)
    bool a_is_int = (dynamic_cast<const DenseVector<int32_t>*>(&a) != nullptr) || (dynamic_cast<const DenseVector<bool>*>(&a) != nullptr);
    bool b_is_int = (dynamic_cast<const DenseVector<int32_t>*>(&b) != nullptr) || (dynamic_cast<const DenseVector<bool>*>(&b) != nullptr);

    if (a_is_int && b_is_int) {
        auto result = std::make_unique<DenseVector<int32_t>>(n, result_file);
        for (uint64_t i = 0; i < n; ++i) {
            result->set(i, (int32_t)a.get_element_as_double(i) + (int32_t)b.get_element_as_double(i));
        }
        return result;
    }

    auto result = std::make_unique<DenseVector<double>>(n, result_file);
    
    // Optimization: If one is UnitVector, copy the other and add scalar at index
    if (a_uv) {
        // Copy b to result
        for (uint64_t i = 0; i < n; ++i) result->set(i, b.get_element_as_double(i));
        // Add a
        uint64_t idx = a_uv->get_active_index();
        result->set(idx, result->get(idx) + a_uv->get_scalar());
        return result;
    }
    if (b_uv) {
        // Copy a to result
        for (uint64_t i = 0; i < n; ++i) result->set(i, a.get_element_as_double(i));
        // Add b
        uint64_t idx = b_uv->get_active_index();
        result->set(idx, result->get(idx) + b_uv->get_scalar());
        return result;
    }

    for (uint64_t i = 0; i < n; ++i) {
        result->set(i, a.get_element_as_double(i) + b.get_element_as_double(i));
    }
    return result;
}

std::unique_ptr<VectorBase> subtract_vectors(const VectorBase& a, const VectorBase& b, const std::string& result_file) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions must match");
    }
    uint64_t n = a.size();

    // Check if both are integer-like
    bool a_is_int = (dynamic_cast<const DenseVector<int32_t>*>(&a) != nullptr) || (dynamic_cast<const DenseVector<bool>*>(&a) != nullptr);
    bool b_is_int = (dynamic_cast<const DenseVector<int32_t>*>(&b) != nullptr) || (dynamic_cast<const DenseVector<bool>*>(&b) != nullptr);

    if (a_is_int && b_is_int) {
        auto result = std::make_unique<DenseVector<int32_t>>(n, result_file);
        for (uint64_t i = 0; i < n; ++i) {
            result->set(i, (int32_t)a.get_element_as_double(i) - (int32_t)b.get_element_as_double(i));
        }
        return result;
    }

    auto result = std::make_unique<DenseVector<double>>(n, result_file);
    for (uint64_t i = 0; i < n; ++i) {
        result->set(i, a.get_element_as_double(i) - b.get_element_as_double(i));
    }
    return result;
}

double dot_product(const VectorBase& a, const VectorBase& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions must match");
    }
    uint64_t n = a.size();
    
    double sum = 0.0;
    for (uint64_t i = 0; i < n; ++i) {
        sum += a.get_element_as_double(i) * b.get_element_as_double(i);
    }
    return sum;
}

std::unique_ptr<VectorBase> scalar_multiply_vector(const VectorBase& v, double scalar, const std::string& result_file) {
    uint64_t n = v.size();
    auto result = std::make_unique<DenseVector<double>>(n, result_file);
    for (uint64_t i = 0; i < n; ++i) {
        result->set(i, v.get_element_as_double(i) * scalar);
    }
    return result;
}

std::unique_ptr<VectorBase> scalar_multiply_vector(const VectorBase& v, int64_t scalar, const std::string& result_file) {
    uint64_t n = v.size();
    
    bool v_is_int = (dynamic_cast<const DenseVector<int32_t>*>(&v) != nullptr) || (dynamic_cast<const DenseVector<bool>*>(&v) != nullptr);

    if (v_is_int) {
        auto result = std::make_unique<DenseVector<int32_t>>(n, result_file);
        for (uint64_t i = 0; i < n; ++i) {
            result->set(i, (int32_t)v.get_element_as_double(i) * (int32_t)scalar);
        }
        return result;
    }

    auto result = std::make_unique<DenseVector<double>>(n, result_file);
    for (uint64_t i = 0; i < n; ++i) {
        result->set(i, v.get_element_as_double(i) * (double)scalar);
    }
    return result;
}

std::unique_ptr<VectorBase> scalar_add_vector(const VectorBase& v, double scalar, const std::string& result_file) {
    uint64_t n = v.size();
    auto result = std::make_unique<DenseVector<double>>(n, result_file);
    for (uint64_t i = 0; i < n; ++i) {
        result->set(i, v.get_element_as_double(i) + scalar);
    }
    return result;
}

std::unique_ptr<VectorBase> scalar_add_vector(const VectorBase& v, int64_t scalar, const std::string& result_file) {
    uint64_t n = v.size();
    
    bool v_is_int = (dynamic_cast<const DenseVector<int32_t>*>(&v) != nullptr) || (dynamic_cast<const DenseVector<bool>*>(&v) != nullptr);

    if (v_is_int) {
        auto result = std::make_unique<DenseVector<int32_t>>(n, result_file);
        for (uint64_t i = 0; i < n; ++i) {
            result->set(i, (int32_t)v.get_element_as_double(i) + (int32_t)scalar);
        }
        return result;
    }

    auto result = std::make_unique<DenseVector<double>>(n, result_file);
    for (uint64_t i = 0; i < n; ++i) {
        result->set(i, v.get_element_as_double(i) + (double)scalar);
    }
    return result;
}

std::unique_ptr<VectorBase> cross_product(const VectorBase& a, const VectorBase& b, const std::string& result_file) {
    if (a.size() != 3 || b.size() != 3) {
        throw std::invalid_argument("Cross product is only defined for 3D vectors");
    }
    
    double a0 = a.get_element_as_double(0);
    double a1 = a.get_element_as_double(1);
    double a2 = a.get_element_as_double(2);
    
    double b0 = b.get_element_as_double(0);
    double b1 = b.get_element_as_double(1);
    double b2 = b.get_element_as_double(2);
    
    auto result = std::make_unique<DenseVector<double>>(3, result_file);
    result->set(0, a1*b2 - a2*b1);
    result->set(1, a2*b0 - a0*b2);
    result->set(2, a0*b1 - a1*b0);
    
    return result;
}

} // namespace pycauset
