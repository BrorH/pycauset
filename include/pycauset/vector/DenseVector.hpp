#pragma once

#include "pycauset/math/ViewValue.hpp"
#include "pycauset/vector/VectorBase.hpp"
#include "pycauset/core/MatrixTraits.hpp"
#include "pycauset/core/StorageUtils.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/core/ScalarUtils.hpp"
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>
#include <vector>
#include <type_traits>
#include <bit>

namespace pycauset {

template <typename T>
class DenseVector : public DenseVectorBase {
public:
    DenseVector(uint64_t n, const std::string& backing_file = "")
        : DenseVectorBase(n) {
        uint64_t size_in_bytes = n * sizeof(T);
        initialize_storage(size_in_bytes, backing_file, 
                         std::string("vector_") + MatrixTraits<T>::name, 
                         sizeof(T),
                         pycauset::Structure::VECTOR,
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
        : DenseVectorBase(n) {
        
        uint64_t size_in_bytes = n * sizeof(T);
        initialize_storage(size_in_bytes, backing_file, 
                         "", 
                         sizeof(T),
                         pycauset::Structure::VECTOR, 
                         MatrixTraits<T>::data_type,
                         n, 1,
                         offset,
                         false);
        
        set_seed(seed);
        set_scalar(scalar);
        set_transposed(is_transposed);
    }

    DenseVector(uint64_t n, std::shared_ptr<MemoryMapper> mapper)
        : DenseVectorBase(n, std::move(mapper), pycauset::Structure::VECTOR, MatrixTraits<T>::data_type) {}

    void set(uint64_t i, T value) {
        ensure_unique();
        if (i >= n_) throw std::out_of_range("Index out of bounds");
        data()[i] = value;
    }

    T get(uint64_t i) const {
        if (i >= n_) throw std::out_of_range("Index out of bounds");
        return data()[i];
    }

    double element_as_double(uint64_t i) const override {
        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            throw std::runtime_error("Complex vector does not support get_element_as_double; use get_element_as_complex");
        }
        return view_value(get(i), scalar_, is_conjugated()).real();  // the one view rule (ViewValue.hpp)
    }

    std::complex<double> element_as_complex(uint64_t i) const override {
        return view_value(get(i), scalar_, is_conjugated());  // the one view rule (ViewValue.hpp)
    }

    T* data() { return static_cast<T*>(require_mapper()->get_data()); }
    const T* data() const { return static_cast<const T*>(require_mapper()->get_data()); }




    std::unique_ptr<VectorBase> transpose(const std::string& result_file = "") const override {
        (void)result_file;
        auto out = std::make_unique<DenseVector<T>>(n_, mapper_);
        out->set_scalar(scalar_);
        out->set_seed(seed_);
        out->set_transposed(!is_transposed());
        out->set_conjugated(is_conjugated());
        return out;
    }
};

template <>
class DenseVector<bool> : public DenseVectorBase {
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
    double element_as_double(uint64_t i) const override;

    const uint64_t* data() const { return static_cast<const uint64_t*>(require_mapper()->get_data()); }
    uint64_t* data() { return static_cast<uint64_t*>(require_mapper()->get_data()); }

    
    std::unique_ptr<VectorBase> transpose(const std::string& result_file = "") const override;
};

} // namespace pycauset
