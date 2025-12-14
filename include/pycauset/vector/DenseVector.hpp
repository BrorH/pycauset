#pragma once

#include "pycauset/vector/VectorBase.hpp"
#include "pycauset/core/MatrixTraits.hpp"
#include "pycauset/core/StorageUtils.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>
#include <vector>
#include <type_traits>
#include <bit>

namespace pycauset {

template <typename T>
class DenseVector : public VectorBase {
public:
    DenseVector(uint64_t n, const std::string& backing_file = "")
        : VectorBase(n) {
        uint64_t size_in_bytes = n * sizeof(T);
        initialize_storage(size_in_bytes, backing_file, 
                         std::string("vector_") + MatrixTraits<T>::name, 
                         sizeof(T),
                         pycauset::MatrixType::DENSE_FLOAT, // Reusing MatrixType for now, or add VectorType
                         MatrixTraits<T>::data_type,
                         n, 1);
    }

    // Constructor for loading with explicit metadata
    DenseVector(uint64_t n, 
                const std::string& backing_file,
                size_t offset,
                uint64_t seed,
                std::complex<double> scalar,
                bool is_transposed)
        : VectorBase(n) {
        
        uint64_t size_in_bytes = n * sizeof(T);
        initialize_storage(size_in_bytes, backing_file, 
                         "", 
                         sizeof(T),
                         pycauset::MatrixType::DENSE_FLOAT, 
                         MatrixTraits<T>::data_type,
                         n, 1,
                         offset,
                         false);
        
        set_seed(seed);
        set_scalar(scalar);
        set_transposed(is_transposed);
    }

    DenseVector(uint64_t n, std::shared_ptr<MemoryMapper> mapper)
        : VectorBase(n, std::move(mapper), pycauset::MatrixType::DENSE_FLOAT, MatrixTraits<T>::data_type) {}

    void set(uint64_t i, T value) {
        ensure_unique();
        if (i >= n_) throw std::out_of_range("Index out of bounds");
        data()[i] = value;
    }

    T get(uint64_t i) const {
        if (i >= n_) throw std::out_of_range("Index out of bounds");
        return data()[i];
    }

    double get_element_as_double(uint64_t i) const override {
        if (scalar_ == 1.0) {
            return static_cast<double>(get(i));
        }
        return (static_cast<double>(get(i)) * scalar_).real();
    }

    T* data() { return static_cast<T*>(require_mapper()->get_data()); }
    const T* data() const { return static_cast<const T*>(require_mapper()->get_data()); }

    std::unique_ptr<VectorBase> multiply_scalar(double factor, const std::string& result_file = "") const override {
        std::string new_path = copy_storage(result_file);
        auto mapper = std::make_unique<MemoryMapper>(new_path, 0, false);
        auto new_vector = std::make_unique<DenseVector<T>>(n_, std::move(mapper));
        new_vector->set_scalar(scalar_ * factor);
        if (result_file.empty()) {
            new_vector->set_temporary(true);
        }
        return new_vector;
    }

    std::unique_ptr<VectorBase> add_scalar(double scalar, const std::string& result_file = "") const override {
        auto result = std::make_unique<DenseVector<T>>(n_, result_file);
        
        const T* src_data = data();
        T* dst_data = result->data();
        
        for (uint64_t i = 0; i < n_; ++i) {
            double val = static_cast<double>(src_data[i]) * scalar_.real() + scalar;
            dst_data[i] = static_cast<T>(val);
        }
        
        result->set_scalar(1.0);
        if (result_file.empty()) {
            result->set_temporary(true);
        }
        return result;
    }

    std::unique_ptr<VectorBase> transpose(const std::string& result_file = "") const override {
        std::string new_path = copy_storage(result_file);
        auto mapper = std::make_unique<MemoryMapper>(new_path, 0, false);
        auto new_vector = std::make_unique<DenseVector<T>>(n_, std::move(mapper));
        new_vector->set_scalar(scalar_);
        new_vector->set_seed(seed_);
        new_vector->set_transposed(!is_transposed());
        if (result_file.empty()) {
            new_vector->set_temporary(true);
        }
        return new_vector;
    }
};

template <>
class DenseVector<bool> : public VectorBase {
public:
    DenseVector(uint64_t n, const std::string& backing_file = "");
    
    DenseVector(uint64_t n, 
                const std::string& backing_file,
                size_t offset,
                uint64_t seed,
                std::complex<double> scalar,
                bool is_transposed);

    DenseVector(uint64_t n, std::shared_ptr<MemoryMapper> mapper);

    void set(uint64_t i, bool value);
    bool get(uint64_t i) const;
    double get_element_as_double(uint64_t i) const override;

    const uint64_t* data() const { return static_cast<const uint64_t*>(require_mapper()->get_data()); }
    uint64_t* data() { return static_cast<uint64_t*>(require_mapper()->get_data()); }

    std::unique_ptr<VectorBase> multiply_scalar(double factor, const std::string& result_file = "") const override;
    std::unique_ptr<VectorBase> add_scalar(double scalar, const std::string& result_file = "") const override;
    
    std::unique_ptr<VectorBase> transpose(const std::string& result_file = "") const override;
};

} // namespace pycauset
