#pragma once

#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/core/MatrixTraits.hpp"
#include "pycauset/math/ViewValue.hpp"
#include "pycauset/core/StorageUtils.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <type_traits>
#include <cstring>

namespace pycauset {

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
        : TriangularMatrixBase(n, pycauset::Structure::TRIANGULAR, MatrixTraits<T>::data_type) {
        has_diagonal_ = has_diagonal;
        // Calculate offsets for T (sizeof(T)*8 bits per element), aligned to 64 bits
        uint64_t size_in_bytes = calculate_triangular_offsets(sizeof(T) * 8, 64);
        initialize_storage(size_in_bytes, backing_file, 
                         std::string("triangular_") + MatrixTraits<T>::name, 
                         8,
                         pycauset::Structure::TRIANGULAR, 
                         MatrixTraits<T>::data_type,
                         n, n);
    }

    // Constructor for loading with explicit metadata
    TriangularMatrix(uint64_t n, 
                     const std::string& backing_file,
                     size_t offset,
                     uint64_t seed,
                     std::complex<double> scalar,
                     bool is_transposed,
                     bool has_diagonal = false)
        : TriangularMatrixBase(n, pycauset::Structure::TRIANGULAR, MatrixTraits<T>::data_type) {
        has_diagonal_ = has_diagonal;
        uint64_t size_in_bytes = calculate_triangular_offsets(sizeof(T) * 8, 64);
        initialize_storage(size_in_bytes, backing_file, 
                         "", 
                         8,
                         pycauset::Structure::TRIANGULAR, 
                         MatrixTraits<T>::data_type,
                         n, n,
                         offset,
                         false);
        
        set_seed(seed);
        set_scalar(scalar);
        set_transposed(is_transposed);
    }

    // The generic MatrixBase::clone rebuilds through the factory, which does not know
    // has_diagonal (and the packed row offsets depend on it): conj() of a triangular matrix
    // with a diagonal used to read shifted values (F-064).
    std::unique_ptr<PersistentObject> clone() const override {
        auto out = std::make_unique<TriangularMatrix<T>>(base_rows(), mapper_, has_diagonal_);
        out->set_scalar(this->get_scalar());
        out->set_conjugated(this->is_conjugated());
        out->set_transposed(this->is_transposed());
        out->set_seed(this->get_seed());
        out->set_view(logical_rows_, logical_cols_, row_offset_, col_offset_);
        return out;
    }

    TriangularMatrix(uint64_t n, std::shared_ptr<MemoryMapper> mapper, bool has_diagonal = false)
        : TriangularMatrixBase(n, std::move(mapper), pycauset::Structure::TRIANGULAR, MatrixTraits<T>::data_type) {
        has_diagonal_ = has_diagonal;
        calculate_triangular_offsets(sizeof(T) * 8, 64);
    }

    void set(uint64_t i, uint64_t j, T value) {
        ensure_unique();
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
        
        if (j >= cols()) throw std::out_of_range("Index out of bounds");

        uint64_t row_offset_bytes = get_row_offset(i);
        // If strict: col_index = j - (i + 1)
        // If diagonal: col_index = j - i
        uint64_t col_index = has_diagonal_ ? (j - i) : (j - (i + 1));
        
        char* base_ptr = static_cast<char*>(require_mapper()->get_data());
        T* row_ptr = reinterpret_cast<T*>(base_ptr + row_offset_bytes);
        
        row_ptr[col_index] = value;
    }

    void set_element_double(uint64_t i, uint64_t j, double value) override {
        set(i, j, static_cast<T>(value));
    }

    T get(uint64_t i, uint64_t j) const {
        if (is_transposed()) {
            // Transposed: Lower Triangular.
            // User gets (i, j). If i > j (lower), we map to (j, i) (upper).
            // If i <= j (upper/diag), its 0 (unless diag and i==j).
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
        
        if (j >= cols()) throw std::out_of_range("Index out of bounds");

        uint64_t row_offset_bytes = get_row_offset(i);
        uint64_t col_index = has_diagonal_ ? (j - i) : (j - (i + 1));
        
        const char* base_ptr = static_cast<const char*>(require_mapper()->get_data());
        const T* row_ptr = reinterpret_cast<const T*>(base_ptr + row_offset_bytes);
        
        return row_ptr[col_index];
    }

    // The view rule lives in math/ViewValue.hpp (one definition, F-086).
    double element_as_double(uint64_t i, uint64_t j) const override {
        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            throw std::runtime_error("Complex matrix does not support get_element_as_double; use get_element_as_complex");
        } else {
            return element_as_complex(i, j).real();
        }
    }

    std::complex<double> element_as_complex(uint64_t i, uint64_t j) const override {
        return view_value(get(i, j), scalar_, is_conjugated());
    }

    T* data() { return static_cast<T*>(require_mapper()->get_data()); }
    const T* data() const { return static_cast<const T*>(require_mapper()->get_data()); }

    std::unique_ptr<MatrixBase> multiply_scalar(double factor, const std::string& = "") const override {
        return scaled_clone(factor);
    }

    std::unique_ptr<MatrixBase> multiply_scalar(int64_t factor, const std::string& = "") const override {
        return scaled_clone(static_cast<double>(factor));
    }

    std::unique_ptr<MatrixBase> multiply_scalar(std::complex<double> factor, const std::string& = "") const override {
        return scaled_clone(factor);
    }



    std::unique_ptr<MatrixBase> transpose(const std::string& result_file = "") const override {
        // Transposing a Triangular Matrix makes it Lower Triangular.
        // But our storage format is strictly Upper Triangular.
        // So we can not just flip a bit. We have to return a DenseMatrix (or a LowerTriangular wrapper if we had one).
        // Or, we can support "Implicit Transpose" where get(i, j) maps to get(j, i).
        // If we do implicit transpose, get(i, j) where i > j (lower triangle) becomes get(j, i) (upper triangle).
        // This works perfectly!
        
        (void)result_file;
        auto out = std::make_unique<TriangularMatrix<T>>(base_rows(), mapper_, has_diagonal_);
        out->set_transposed(!this->is_transposed());
        out->set_scalar(scalar_);
        out->set_seed(seed_);
        out->set_conjugated(is_conjugated());
        return out;
    }

    std::unique_ptr<TriangularMatrix<T>> bitwise_not(const std::string& result_file = "") const {
        const uint64_t n = rows();
        auto result = std::make_unique<TriangularMatrix<T>>(n, result_file, has_diagonal_);
        
        const T* src_data = data();
        T* dst_data = result->data();
        
        for (uint64_t i = 0; i < n; ++i) {
            uint64_t row_len = has_diagonal_ ? ((n > i) ? (n - i) : 0) : ((n > i) ? (n - 1 - i) : 0);
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

    // Inverse (throws)
    std::unique_ptr<TriangularMatrix<double>> inverse(const std::string& result_file = "") const {
        throw std::runtime_error("Strictly upper triangular matrices are singular (not invertible).");
    }
};

} // namespace pycauset
