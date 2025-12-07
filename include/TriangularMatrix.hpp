#pragma once

#include "MatrixBase.hpp"
#include "MatrixTraits.hpp"
#include "StoragePaths.hpp"
#include "DenseMatrix.hpp"
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

    bool has_diagonal() const { return has_diagonal_; }

protected:
    std::vector<uint64_t> row_offsets_;
    bool has_diagonal_ = false;

    uint64_t calculate_triangular_offsets(uint64_t element_bits, uint64_t alignment_bits);
};

template <typename T>
class TriangularMatrix : public TriangularMatrixBase {
public:
    TriangularMatrix(uint64_t n, const std::string& backing_file = "", bool has_diagonal = false)
        : TriangularMatrixBase(n, pycauset::MatrixType::TRIANGULAR_FLOAT, MatrixTraits<T>::data_type) {
        has_diagonal_ = has_diagonal;
        // Calculate offsets for T (sizeof(T)*8 bits per element), aligned to 64 bits
        uint64_t size_in_bytes = calculate_triangular_offsets(sizeof(T) * 8, 64);
        initialize_storage(size_in_bytes, backing_file, 
                         std::string("triangular_") + MatrixTraits<T>::name, 
                         8,
                         pycauset::MatrixType::TRIANGULAR_FLOAT, 
                         MatrixTraits<T>::data_type,
                         n, n);
    }

    // Constructor for loading with explicit metadata
    TriangularMatrix(uint64_t n, 
                     const std::string& backing_file,
                     size_t offset,
                     uint64_t seed,
                     double scalar,
                     bool is_transposed,
                     bool has_diagonal = false)
        : TriangularMatrixBase(n, pycauset::MatrixType::TRIANGULAR_FLOAT, MatrixTraits<T>::data_type) {
        has_diagonal_ = has_diagonal;
        uint64_t size_in_bytes = calculate_triangular_offsets(sizeof(T) * 8, 64);
        initialize_storage(size_in_bytes, backing_file, 
                         "", 
                         8,
                         pycauset::MatrixType::TRIANGULAR_FLOAT, 
                         MatrixTraits<T>::data_type,
                         n, n,
                         offset,
                         false);
        
        set_seed(seed);
        set_scalar(scalar);
        set_transposed(is_transposed);
    }

    TriangularMatrix(uint64_t n, std::unique_ptr<MemoryMapper> mapper, bool has_diagonal = false)
        : TriangularMatrixBase(n, std::move(mapper), pycauset::MatrixType::TRIANGULAR_FLOAT, MatrixTraits<T>::data_type) {
        has_diagonal_ = has_diagonal;
        calculate_triangular_offsets(sizeof(T) * 8, 64);
    }

    void set(uint64_t i, uint64_t j, T value) {
        if (is_transposed()) {
            // Transposed: Lower Triangular.
            // User sets (i, j). If i > j (lower), we map to (j, i) (upper) in storage.
            if (has_diagonal_) {
                if (i < j) throw std::invalid_argument("Lower triangular (transposed)");
            } else {
                if (i <= j) throw std::invalid_argument("Strictly lower triangular (transposed)");
            }
            std::swap(i, j);
        } else {
            if (has_diagonal_) {
                if (i > j) throw std::invalid_argument("Upper triangular");
            } else {
                if (i >= j) throw std::invalid_argument("Strictly upper triangular");
            }
        }
        
        if (j >= n_) throw std::out_of_range("Index out of bounds");

        uint64_t row_offset_bytes = get_row_offset(i);
        // If strict: col_index = j - (i + 1)
        // If diagonal: col_index = j - i
        uint64_t col_index = has_diagonal_ ? (j - i) : (j - (i + 1));
        
        char* base_ptr = static_cast<char*>(require_mapper()->get_data());
        T* row_ptr = reinterpret_cast<T*>(base_ptr + row_offset_bytes);
        
        row_ptr[col_index] = value;
    }

    T get(uint64_t i, uint64_t j) const {
        if (is_transposed()) {
            // Transposed: Lower Triangular.
            // User gets (i, j). If i > j (lower), we map to (j, i) (upper).
            // If i <= j (upper/diag), it's 0 (unless diag and i==j).
            if (has_diagonal_) {
                if (i < j) return static_cast<T>(0);
            } else {
                if (i <= j) return static_cast<T>(0);
            }
            std::swap(i, j);
        } else {
            if (has_diagonal_) {
                if (i > j) return static_cast<T>(0);
            } else {
                if (i >= j) return static_cast<T>(0);
            }
        }
        
        if (j >= n_) throw std::out_of_range("Index out of bounds");

        uint64_t row_offset_bytes = get_row_offset(i);
        uint64_t col_index = has_diagonal_ ? (j - i) : (j - (i + 1));
        
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

    std::unique_ptr<MatrixBase> add_scalar(double scalar, const std::string& result_file = "") const override {
        auto result = std::make_unique<DenseMatrix<T>>(n_, result_file);
        T* dst_data = result->data();
        
        for (uint64_t i = 0; i < n_; ++i) {
            for (uint64_t j = 0; j < n_; ++j) {
                // get_element_as_double handles scalar_
                double val = get_element_as_double(i, j) + scalar;
                dst_data[i * n_ + j] = static_cast<T>(val);
            }
        }
        
        if (result_file.empty()) {
            result->set_temporary(true);
        }
        return result;
    }

    std::unique_ptr<MatrixBase> add_scalar(int64_t scalar, const std::string& result_file = "") const override {
        return add_scalar(static_cast<double>(scalar), result_file);
    }

    std::unique_ptr<MatrixBase> transpose(const std::string& result_file = "") const override {
        // Transposing a Triangular Matrix makes it Lower Triangular.
        // But our storage format is strictly Upper Triangular.
        // So we can't just flip a bit. We have to return a DenseMatrix (or a LowerTriangular wrapper if we had one).
        // Or, we can support "Implicit Transpose" where get(i, j) maps to get(j, i).
        // If we do implicit transpose, get(i, j) where i > j (lower triangle) becomes get(j, i) (upper triangle).
        // This works perfectly!
        
        std::string new_path = copy_storage(result_file);
        auto mapper = std::make_unique<MemoryMapper>(new_path, 0, false);
        auto new_matrix = std::make_unique<TriangularMatrix<T>>(n_, std::move(mapper));
        
        new_matrix->set_transposed(!this->is_transposed());
        
        if (result_file.empty()) {
            new_matrix->set_temporary(true);
        }
        return new_matrix;
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
        
        // Delegate to ComputeContext (AutoSolver)
        pycauset::ComputeContext::instance().get_device()->matmul(*this, other, *result);
        
        return result;
    }

    // Inverse (throws)
    std::unique_ptr<TriangularMatrix<double>> inverse(const std::string& result_file = "") const {
        throw std::runtime_error("Strictly upper triangular matrices are singular (not invertible).");
    }
};
