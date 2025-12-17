#pragma once

#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/core/MatrixTraits.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace pycauset {

template <typename T>
class IdentityMatrix : public MatrixBase {
public:
    IdentityMatrix(uint64_t n, const std::string& backing_file = "")
        : IdentityMatrix(n, n, backing_file) {}

    IdentityMatrix(uint64_t rows, uint64_t cols, const std::string& backing_file)
        : MatrixBase(rows, cols, pycauset::MatrixType::IDENTITY, MatrixTraits<T>::data_type) {
        // Identity matrices have no data payload; they only need metadata.
        initialize_storage(
            /*size_in_bytes=*/0,
            backing_file,
            /*fallback_prefix=*/"identity",
            /*min_size_bytes=*/0,
            pycauset::MatrixType::IDENTITY,
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
        : MatrixBase(rows, cols, pycauset::MatrixType::IDENTITY, MatrixTraits<T>::data_type) {
        initialize_storage(
            /*size_in_bytes=*/0,
            backing_file,
            /*fallback_prefix=*/"",
            /*min_size_bytes=*/0,
            pycauset::MatrixType::IDENTITY,
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
        : MatrixBase(rows,
                     cols,
                     std::move(mapper),
                     pycauset::MatrixType::IDENTITY,
                     MatrixTraits<T>::data_type) {}

    void set(uint64_t /*i*/, uint64_t /*j*/, T /*value*/) {
        throw std::runtime_error("Cannot modify IdentityMatrix elements");
    }

    T get(uint64_t i, uint64_t j) const {
        if (i >= rows() || j >= cols()) throw std::out_of_range("Index out of bounds");
        if (i == j) return static_cast<T>(1);
        return static_cast<T>(0);
    }

    double get_element_as_double(uint64_t i, uint64_t j) const override {
        if (i >= rows() || j >= cols()) throw std::out_of_range("Index out of bounds");
        if (i == j) {
            return scalar_.real();
        }
        return 0.0;
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

    std::unique_ptr<IdentityMatrix<T>> multiply(const IdentityMatrix<T>& other, const std::string& result_file = "") const {
        if (rows() != other.rows() || cols() != other.cols()) throw std::invalid_argument("Dimension mismatch");
        auto result = std::make_unique<IdentityMatrix<T>>(base_rows(), base_cols(), result_file);
        result->set_scalar(scalar_ * other.get_scalar());
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
        auto result = std::make_unique<IdentityMatrix<T>>(base_rows(), base_cols(), result_file);
        result->set_scalar(scalar_);
        result->set_transposed(!is_transposed());
        return result;
    }

    std::unique_ptr<MatrixBase> multiply_scalar(double factor, const std::string& result_file = "") const override {
        auto result = std::make_unique<IdentityMatrix<T>>(base_rows(), base_cols(), result_file);
        result->set_scalar(scalar_ * factor);
        result->set_transposed(is_transposed());
        if (result_file.empty()) {
            result->set_temporary(true);
        }
        return result;
    }

    std::unique_ptr<MatrixBase> multiply_scalar(int64_t factor, const std::string& result_file = "") const override {
        return multiply_scalar(static_cast<double>(factor), result_file);
    }

    std::unique_ptr<MatrixBase> add_scalar(double scalar, const std::string& result_file = "") const override {
        const uint64_t r = rows();
        const uint64_t c = cols();
        auto result = std::make_unique<DenseMatrix<double>>(r, c, result_file);
        double* dst_data = result->data();

        for (uint64_t i = 0; i < r; ++i) {
            for (uint64_t j = 0; j < c; ++j) {
                if (i == j) {
                    dst_data[i * c + j] = (scalar_ + scalar).real();
                } else {
                    dst_data[i * c + j] = scalar;
                }
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
};

} // namespace pycauset
