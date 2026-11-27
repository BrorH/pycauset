#pragma once

#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/core/MatrixTraits.hpp"
#include "pycauset/math/ViewValue.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace pycauset {

template <typename T>
class IdentityMatrix : public IdentityMatrixBase {
public:
    IdentityMatrix(uint64_t n, const std::string& backing_file = "")
        : IdentityMatrix(n, n, backing_file) {}

    IdentityMatrix(uint64_t rows, uint64_t cols, const std::string& backing_file)
        : IdentityMatrixBase(rows, cols, pycauset::Structure::IDENTITY, MatrixTraits<T>::data_type) {
        // Identity matrices have no data payload; they only need metadata.
        initialize_storage(
            /*size_in_bytes=*/0,
            backing_file,
            /*fallback_prefix=*/"identity",
            /*min_size_bytes=*/0,
            pycauset::Structure::IDENTITY,
            MatrixTraits<T>::data_type,
            rows,
            cols);
    }

    // Constructor for loading (square)
    IdentityMatrix(uint64_t n,
                   const std::string& backing_file,
                   size_t offset,
                   uint64_t seed,
                   std::complex<double> scalar,
                   bool is_transposed)
        : IdentityMatrix(n, n, backing_file, offset, seed, scalar, is_transposed) {}

    // Constructor for loading (rectangular)
    IdentityMatrix(uint64_t rows,
                   uint64_t cols,
                   const std::string& backing_file,
                   size_t offset,
                   uint64_t seed,
                   std::complex<double> scalar,
                   bool is_transposed)
        : IdentityMatrixBase(rows, cols, pycauset::Structure::IDENTITY, MatrixTraits<T>::data_type) {
        initialize_storage(
            /*size_in_bytes=*/0,
            backing_file,
            /*fallback_prefix=*/"",
            /*min_size_bytes=*/0,
            pycauset::Structure::IDENTITY,
            MatrixTraits<T>::data_type,
            rows,
            cols,
            offset,
            /*create_new=*/false);

        set_seed(seed);
        set_scalar(scalar);
        set_transposed(is_transposed);
    }

    IdentityMatrix(uint64_t n, std::shared_ptr<MemoryMapper> mapper)
        : IdentityMatrix(n, n, std::move(mapper)) {}

    IdentityMatrix(uint64_t rows, uint64_t cols, std::shared_ptr<MemoryMapper> mapper)
        : IdentityMatrixBase(rows,
                     cols,
                     std::move(mapper),
                     pycauset::Structure::IDENTITY,
                     MatrixTraits<T>::data_type) {}

    void set(uint64_t /*i*/, uint64_t /*j*/, T /*value*/) {
        throw std::runtime_error("Cannot modify IdentityMatrix elements");
    }

    T get(uint64_t i, uint64_t j) const {
        if (i >= rows() || j >= cols()) throw std::out_of_range("Index out of bounds");
        if (i == j) return static_cast<T>(1);
        return static_cast<T>(0);
    }

    // The view rule lives in math/ViewValue.hpp (one definition, F-086).
    double element_as_double(uint64_t i, uint64_t j) const override {
        if constexpr (is_complex_storage_v<T>) {
            throw std::runtime_error("Complex matrix does not support get_element_as_double; use get_element_as_complex");
        } else {
            return element_as_complex(i, j).real();
        }
    }

    std::complex<double> element_as_complex(uint64_t i, uint64_t j) const override {
        return view_value(get(i, j), scalar_, is_conjugated());
    }

    void set_element_double(uint64_t i, uint64_t j, double value) override {
        // Identity is immutable in structure.
        bool is_diag = (i == j);
        double expected = is_diag ? scalar_.real() : 0.0;
        if (std::abs(value - expected) > 1e-9) {
            throw std::runtime_error("Cannot modify IdentityMatrix structure");
        }
    }

    // Specialized operations returning IdentityMatrix
    std::unique_ptr<IdentityMatrix<T>> add(const IdentityMatrix<T>& other, const std::string& result_file = "") const {
        if (rows() != other.rows() || cols() != other.cols()) throw std::invalid_argument("Dimension mismatch");
        auto result = std::make_unique<IdentityMatrix<T>>(base_rows(), base_cols(), result_file);
        result->set_scalar(scalar_ + other.get_scalar());
        result->set_transposed(is_transposed());
        return result;
    }

    std::unique_ptr<IdentityMatrix<T>> subtract(const IdentityMatrix<T>& other, const std::string& result_file = "") const {
        if (rows() != other.rows() || cols() != other.cols()) throw std::invalid_argument("Dimension mismatch");
        auto result = std::make_unique<IdentityMatrix<T>>(base_rows(), base_cols(), result_file);
        result->set_scalar(scalar_ - other.get_scalar());
        result->set_transposed(is_transposed());
        return result;
    }

    std::unique_ptr<IdentityMatrix<T>> elementwise_multiply(const IdentityMatrix<T>& other, const std::string& result_file = "") const {
        if (rows() != other.rows() || cols() != other.cols()) throw std::invalid_argument("Dimension mismatch");
        auto result = std::make_unique<IdentityMatrix<T>>(base_rows(), base_cols(), result_file);
        result->set_scalar(scalar_ * other.get_scalar());
        result->set_transposed(is_transposed());
        return result;
    }

    std::unique_ptr<MatrixBase> transpose(const std::string& result_file = "") const override {
        (void)result_file;
        return transposed_clone();
    }

    std::unique_ptr<MatrixBase> multiply_scalar(double factor, const std::string& = "") const override {
        return scaled_clone(factor);
    }

    std::unique_ptr<MatrixBase> multiply_scalar(int64_t factor, const std::string& = "") const override {
        return scaled_clone(static_cast<double>(factor));
    }

    std::unique_ptr<MatrixBase> multiply_scalar(std::complex<double> factor, const std::string& = "") const override {
        return scaled_clone(factor);
    }


};

} // namespace pycauset
