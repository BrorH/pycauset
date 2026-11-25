#pragma once

#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/core/MatrixTraits.hpp"
#include "pycauset/core/StorageUtils.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>
#include <vector>
#include <type_traits>

namespace pycauset {

template <typename T>
class DiagonalMatrix : public DiagonalMatrixBase {
public:
    DiagonalMatrix(uint64_t n, const std::string& backing_file = "")
        : DiagonalMatrixBase(n, pycauset::Structure::DIAGONAL, MatrixTraits<T>::data_type) {
        uint64_t size_in_bytes = n * sizeof(T);
        initialize_storage(size_in_bytes, backing_file, 
                         std::string("diagonal_") + MatrixTraits<T>::name, 
                         sizeof(T),
                         pycauset::Structure::DIAGONAL, 
                         MatrixTraits<T>::data_type,
                         n, n);
    }

    // Constructor for loading with explicit metadata
    DiagonalMatrix(uint64_t n, 
                const std::string& backing_file,
                size_t offset,
                uint64_t seed,
                std::complex<double> scalar,
                bool is_transposed)
        : DiagonalMatrixBase(n, pycauset::Structure::DIAGONAL, MatrixTraits<T>::data_type) {
        
        uint64_t size_in_bytes = n * sizeof(T);
        initialize_storage(size_in_bytes, backing_file, 
                         "", 
                         sizeof(T),
                         pycauset::Structure::DIAGONAL, 
                         MatrixTraits<T>::data_type,
                         n, n,
                         offset,
                         false);
        
        set_seed(seed);
        set_scalar(scalar);
        set_transposed(is_transposed);
    }

    DiagonalMatrix(uint64_t n, std::shared_ptr<MemoryMapper> mapper)
        : DiagonalMatrixBase(n, std::move(mapper), pycauset::Structure::DIAGONAL, MatrixTraits<T>::data_type) {}

    virtual void set(uint64_t i, uint64_t j, T value) {
        ensure_unique();
        if (i >= rows() || j >= cols()) throw std::out_of_range("Index out of bounds");
        if (i != j) {
            if (value != T(0)) throw std::invalid_argument("Cannot set off-diagonal element to non-zero");
            return;
        }
        data()[i] = value;
    }

    virtual void set_diagonal(uint64_t i, T value) {
        ensure_unique();
        if (i >= rows()) throw std::out_of_range("Index out of bounds");
        data()[i] = value;
    }

    virtual T get_diagonal(uint64_t i) const {
        if (i >= rows()) throw std::out_of_range("Index out of bounds");
        return data()[i];
    }

    virtual T get(uint64_t i, uint64_t j) const {
        if (i >= rows() || j >= cols()) throw std::out_of_range("Index out of bounds");
        if (i != j) return T(0);
        return data()[i];
    }

    double element_as_double(uint64_t i, uint64_t j) const override {
        if (scalar_ == 1.0) {
            return static_cast<double>(get(i, j));
        }
        return (static_cast<double>(get(i, j)) * scalar_).real();
    }

    void set_element_double(uint64_t i, uint64_t j, double value) override {
        set(i, j, static_cast<T>(value));
    }

    T* data() { return static_cast<T*>(require_mapper()->get_data()); }
    const T* data() const { return static_cast<const T*>(require_mapper()->get_data()); }

    std::unique_ptr<MatrixBase> multiply_scalar(double factor, const std::string& result_file = "") const override {
        std::string new_path = copy_storage(result_file);
        auto mapper = std::make_unique<MemoryMapper>(new_path, 0, false);
        auto new_matrix = std::make_unique<DiagonalMatrix<T>>(base_rows(), std::move(mapper));
        new_matrix->set_scalar(scalar_ * factor);
        if (result_file.empty()) {
            new_matrix->set_temporary(true);
        }
        return new_matrix;
    }

    std::unique_ptr<MatrixBase> multiply_scalar(int64_t factor, const std::string& result_file = "") const override {
        return multiply_scalar(static_cast<double>(factor), result_file);
    }



    std::unique_ptr<MatrixBase> transpose(const std::string& result_file = "") const override {
        // A diagonal matrix is its own transpose: share storage (copy-on-write), O(1) (F-044).
        if (!result_file.empty()) return multiply_scalar(1.0, result_file);  // explicit file: copy
        auto c = this->clone();
        return std::unique_ptr<MatrixBase>(static_cast<MatrixBase*>(c.release()));
    }

protected:
    // Protected constructor for subclasses (like IdentityMatrix) that might want 0 storage
    DiagonalMatrix(uint64_t n, 
                   uint64_t size_in_bytes, 
                   const std::string& backing_file,
                   const std::string& fallback_name,
                   pycauset::Structure mtype,
                   pycauset::DataType dtype)
        : DiagonalMatrixBase(n, mtype, dtype) {
        
        initialize_storage(size_in_bytes, backing_file, 
                         fallback_name, 
                         size_in_bytes > 0 ? sizeof(T) : 0,
                         mtype, 
                         dtype,
                         n, n);
    }
    
    // Protected constructor for loading subclasses
    DiagonalMatrix(uint64_t n, 
                   uint64_t size_in_bytes,
                   const std::string& backing_file,
                   size_t offset,
                   uint64_t seed,
                   std::complex<double> scalar,
                   bool is_transposed,
                   pycauset::Structure mtype,
                   pycauset::DataType dtype)
        : DiagonalMatrixBase(n, mtype, dtype) {
            
        initialize_storage(size_in_bytes, backing_file, 
                         "", 
                         size_in_bytes > 0 ? sizeof(T) : 0,
                         mtype, 
                         dtype,
                         n, n,
                         offset,
                         false);
        
        set_seed(seed);
        set_scalar(scalar);
        set_transposed(is_transposed);
    }

    DiagonalMatrix(uint64_t n, std::shared_ptr<MemoryMapper> mapper, pycauset::Structure mtype, pycauset::DataType dtype)
        : DiagonalMatrixBase(n, std::move(mapper), mtype, dtype) {}
};

} // namespace pycauset
