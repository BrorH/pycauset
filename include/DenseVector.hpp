#pragma once

#include "VectorBase.hpp"
#include "MatrixTraits.hpp"
#include "StoragePaths.hpp"
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>
#include <vector>
#include <type_traits>
#include <bit>

template <typename T>
class DenseVector : public VectorBase {
public:
    DenseVector(uint64_t n, const std::string& backing_file = "")
        : VectorBase(n) {
        uint64_t size_in_bytes = n * sizeof(T);
        initialize_storage(size_in_bytes, backing_file, 
                         std::string("vector_") + MatrixTraits<T>::name, 
                         sizeof(T),
                         pycauset::MatrixType::VECTOR,
                         MatrixTraits<T>::data_type,
                         n, 1);
    }

    DenseVector(uint64_t n, std::unique_ptr<MemoryMapper> mapper)
        : VectorBase(n, std::move(mapper)) {}

    void set(uint64_t i, T value) {
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
        return static_cast<double>(get(i)) * scalar_;
    }

    std::unique_ptr<VectorBase> transpose(const std::string& saveas = "") const override {
        std::string target = saveas;
        if (target.empty()) {
            target = pycauset::make_unique_storage_file("transpose");
        }
        std::string new_path = this->copy_storage(target);
        
        // Calculate data size (file size - header)
        uint64_t file_size = std::filesystem::file_size(new_path);
        uint64_t data_size = file_size - sizeof(pycauset::FileHeader);

        auto mapper = std::make_unique<MemoryMapper>(new_path, data_size, false);
        auto new_vec = std::make_unique<DenseVector<T>>(this->size(), std::move(mapper));
        
        // Flip the transposed bit
        new_vec->set_transposed(!this->is_transposed());
        
        return new_vec;
    }

    T* data() { return static_cast<T*>(require_mapper()->get_data()); }
    const T* data() const { return static_cast<const T*>(require_mapper()->get_data()); }
};

// Specialization for bool (Bit Vector)
template <>
class DenseVector<bool> : public VectorBase {
public:
    DenseVector(uint64_t n, const std::string& backing_file = "");
    DenseVector(uint64_t n, std::unique_ptr<MemoryMapper> mapper);

    void set(uint64_t i, bool value);
    bool get(uint64_t i) const;
    
    double get_element_as_double(uint64_t i) const override;

    std::unique_ptr<VectorBase> transpose(const std::string& saveas = "") const override;

    // Helper to get raw data for bulk operations
    uint64_t* data() { return static_cast<uint64_t*>(require_mapper()->get_data()); }
    const uint64_t* data() const { return static_cast<const uint64_t*>(require_mapper()->get_data()); }
};
