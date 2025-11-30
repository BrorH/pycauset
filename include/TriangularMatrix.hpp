#pragma once

#include "MatrixBase.hpp"
#include "MatrixTraits.hpp"
#include "StoragePaths.hpp"
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <type_traits>

// Non-template base class for shared logic (offsets)
class TriangularMatrixBase : public MatrixBase {
public:
    using MatrixBase::MatrixBase;

    uint64_t get_row_offset(uint64_t i) const { 
        if (i >= row_offsets_.size()) return 0;
        return row_offsets_[i]; 
    }

protected:
    TriangularMatrixBase(uint64_t n, std::unique_ptr<MemoryMapper> mapper) 
        : MatrixBase(n, std::move(mapper)) {}

    std::vector<uint64_t> row_offsets_;

    uint64_t calculate_triangular_offsets(uint64_t element_bits, uint64_t alignment_bits);
};

template <typename T>
class TriangularMatrix : public TriangularMatrixBase {
public:
    TriangularMatrix(uint64_t n, const std::string& backing_file = "")
        : TriangularMatrixBase(n, nullptr) { // Initialize with nullptr mapper first
        // Calculate offsets for T (sizeof(T)*8 bits per element), aligned to 64 bits
        uint64_t size_in_bytes = calculate_triangular_offsets(sizeof(T) * 8, 64);
        initialize_storage(size_in_bytes, backing_file, 
                         std::string("triangular_") + MatrixTraits<T>::name, 
                         8,
                         pycauset::MatrixType::TRIANGULAR_FLOAT, // Placeholder
                         MatrixTraits<T>::data_type,
                         n, n);
    }

    TriangularMatrix(uint64_t n, std::unique_ptr<MemoryMapper> mapper)
        : TriangularMatrixBase(n, std::move(mapper)) {
        calculate_triangular_offsets(sizeof(T) * 8, 64);
    }

    void set(uint64_t i, uint64_t j, T value) {
        if (i >= j) throw std::invalid_argument("Strictly upper triangular");
        if (j >= n_) throw std::out_of_range("Index out of bounds");

        uint64_t row_offset_bytes = get_row_offset(i);
        uint64_t col_index = j - (i + 1);
        
        char* base_ptr = static_cast<char*>(require_mapper()->get_data());
        T* row_ptr = reinterpret_cast<T*>(base_ptr + row_offset_bytes);
        
        row_ptr[col_index] = value;
    }

    T get(uint64_t i, uint64_t j) const {
        if (i >= j) return static_cast<T>(0);
        if (j >= n_) throw std::out_of_range("Index out of bounds");

        uint64_t row_offset_bytes = get_row_offset(i);
        uint64_t col_index = j - (i + 1);
        
        const char* base_ptr = static_cast<const char*>(require_mapper()->get_data());
        const T* row_ptr = reinterpret_cast<const T*>(base_ptr + row_offset_bytes);
        
        return row_ptr[col_index];
    }

    double get_element_as_double(uint64_t i, uint64_t j) const override {
        if (scalar_ == 1.0) {
            return static_cast<double>(get(i, j));
        }
        return static_cast<double>(get(i, j)) * scalar_;
    }

    T* data() { return static_cast<T*>(require_mapper()->get_data()); }
    const T* data() const { return static_cast<const T*>(require_mapper()->get_data()); }

    std::unique_ptr<MatrixBase> multiply_scalar(double factor, const std::string& result_file = "") const override {
        std::string new_path = copy_storage(result_file);
        auto mapper = std::make_unique<MemoryMapper>(new_path, 0, false);
        auto new_matrix = std::make_unique<TriangularMatrix<T>>(n_, std::move(mapper));
        new_matrix->set_scalar(scalar_ * factor);
        if (result_file.empty()) {
            new_matrix->set_temporary(true);
        }
        return new_matrix;
    }

    std::unique_ptr<MatrixBase> multiply_scalar(int64_t factor, const std::string& result_file = "") const override {
        return multiply_scalar(static_cast<double>(factor), result_file);
    }

    std::unique_ptr<TriangularMatrix<T>> bitwise_not(const std::string& result_file = "") const {
        auto result = std::make_unique<TriangularMatrix<T>>(n_, result_file);
        
        const T* src_data = data();
        T* dst_data = result->data();
        
        for (uint64_t i = 0; i < n_; ++i) {
            uint64_t row_len = (n_ - 1) - i;
            if (row_len == 0) continue;
            
            uint64_t row_offset_bytes = get_row_offset(i);
            const T* row_src = reinterpret_cast<const T*>(
                reinterpret_cast<const char*>(src_data) + row_offset_bytes);
            T* row_dst = reinterpret_cast<T*>(
                reinterpret_cast<char*>(dst_data) + row_offset_bytes);
                
            if constexpr (std::is_floating_point_v<T>) {
                 const uint64_t* src_bits = reinterpret_cast<const uint64_t*>(row_src);
                 uint64_t* dst_bits = reinterpret_cast<uint64_t*>(row_dst);
                 for (uint64_t k = 0; k < row_len; ++k) {
                     dst_bits[k] = ~src_bits[k];
                 }
            } else {
                for (uint64_t k = 0; k < row_len; ++k) {
                    row_dst[k] = ~row_src[k];
                }
            }
        }
        
        result->set_scalar(scalar_);
        return result;
    }

    std::unique_ptr<TriangularMatrix<T>> multiply(const TriangularMatrix<T>& other, const std::string& result_file = "") const {
        if (n_ != other.size()) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        auto result = std::make_unique<TriangularMatrix<T>>(n_, result_file);
        
        std::vector<T> accumulator(n_);

        const char* a_base = static_cast<const char*>(require_mapper()->get_data());
        const char* b_base = static_cast<const char*>(other.require_mapper()->get_data());
        char* c_base = static_cast<char*>(result->require_mapper()->get_data());

        for (uint64_t i = 0; i < n_; ++i) {
            std::fill(accumulator.begin(), accumulator.end(), static_cast<T>(0));

            uint64_t row_len_a = (n_ - 1) - i;
            if (row_len_a == 0) continue;

            uint64_t a_offset = get_row_offset(i);
            const T* a_row = reinterpret_cast<const T*>(a_base + a_offset);

            for (uint64_t k_idx = 0; k_idx < row_len_a; ++k_idx) {
                T val_a = a_row[k_idx];
                if (val_a == static_cast<T>(0)) continue;

                uint64_t k = i + 1 + k_idx;

                uint64_t row_len_b = (n_ - 1) - k;
                if (row_len_b > 0) {
                    uint64_t b_offset = other.get_row_offset(k);
                    const T* b_row = reinterpret_cast<const T*>(b_base + b_offset);
                    
                    for (uint64_t j_idx = 0; j_idx < row_len_b; ++j_idx) {
                        uint64_t j = k + 1 + j_idx;
                        accumulator[j] += val_a * b_row[j_idx];
                    }
                }
            }

            uint64_t c_offset = result->get_row_offset(i);
            T* c_row = reinterpret_cast<T*>(c_base + c_offset);
            
            for (uint64_t j_idx = 0; j_idx < row_len_a; ++j_idx) {
                uint64_t j = i + 1 + j_idx;
                c_row[j_idx] = accumulator[j];
            }
        }
        
        result->set_scalar(scalar_ * other.get_scalar());
        return result;
    }

    // Inverse (throws)
    std::unique_ptr<TriangularMatrix<double>> inverse(const std::string& result_file = "") const {
        throw std::runtime_error("Strictly upper triangular matrices are singular (not invertible).");
    }
};
