#pragma once

#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/core/MatrixTraits.hpp"
#include "pycauset/core/StorageUtils.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/compute/ComputeContext.hpp"
#include "pycauset/core/ScalarUtils.hpp"
#include "pycauset/vector/DenseVector.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pycauset {

template <typename T>
class DenseMatrix : public MatrixBase {
public:
    using MatrixBase::operator=;

    DenseMatrix(uint64_t n, const std::string& backing_file = "")
        : DenseMatrix(n, n, backing_file) {}

    DenseMatrix(uint64_t rows, uint64_t cols, const std::string& backing_file = "")
        : MatrixBase(rows, cols, pycauset::MatrixType::DENSE_FLOAT, MatrixTraits<T>::data_type) {
        const uint64_t size_in_bytes = rows * cols * sizeof(T);
        initialize_storage(size_in_bytes,
                           backing_file,
                           std::string("dense_") + MatrixTraits<T>::name,
                           sizeof(T),
                           pycauset::MatrixType::DENSE_FLOAT,
                           MatrixTraits<T>::data_type,
                           rows,
                           cols);
    }

    DenseMatrix(uint64_t n,
                const std::string& backing_file,
                size_t offset,
                uint64_t seed,
                std::complex<double> scalar,
                bool is_transposed)
        : DenseMatrix(n, n, backing_file, offset, seed, scalar, is_transposed) {}

    DenseMatrix(uint64_t rows,
                uint64_t cols,
                const std::string& backing_file,
                size_t offset,
                uint64_t seed,
                std::complex<double> scalar,
                bool is_transposed)
        : MatrixBase(rows, cols, pycauset::MatrixType::DENSE_FLOAT, MatrixTraits<T>::data_type) {
        const uint64_t size_in_bytes = rows * cols * sizeof(T);
        initialize_storage(size_in_bytes,
                           backing_file,
                           "",
                           sizeof(T),
                           pycauset::MatrixType::DENSE_FLOAT,
                           MatrixTraits<T>::data_type,
                           rows,
                           cols,
                           offset,
                           false);

        set_seed(seed);
        set_scalar(scalar);
        set_transposed(is_transposed);
    }

    DenseMatrix(uint64_t n, std::shared_ptr<MemoryMapper> mapper)
        : DenseMatrix(n, n, std::move(mapper)) {}

    DenseMatrix(uint64_t rows, uint64_t cols, std::shared_ptr<MemoryMapper> mapper)
        : MatrixBase(rows, cols, std::move(mapper), pycauset::MatrixType::DENSE_FLOAT, MatrixTraits<T>::data_type) {}

    DenseMatrix(uint64_t view_rows,
                uint64_t view_cols,
                std::shared_ptr<MemoryMapper> mapper,
                uint64_t base_rows,
                uint64_t base_cols,
                uint64_t row_offset,
                uint64_t col_offset,
                uint64_t seed,
                std::complex<double> scalar,
                bool is_transposed,
                bool is_conjugated,
                bool is_temporary)
        : MatrixBase(base_rows, base_cols, std::move(mapper), pycauset::MatrixType::DENSE_FLOAT, MatrixTraits<T>::data_type, seed, scalar, is_transposed, is_temporary) {
        set_view(view_rows, view_cols, row_offset, col_offset);
        set_conjugated(is_conjugated);
    }

    void set(uint64_t i, uint64_t j, T value) {
        ensure_unique();
        if (i >= rows() || j >= cols()) throw std::out_of_range("Index out of bounds");
        const uint64_t storage_cols = base_cols();
        const uint64_t storage_row = row_offset() + (is_transposed() ? j : i);
        const uint64_t storage_col = col_offset() + (is_transposed() ? i : j);
        const uint64_t idx = storage_row * storage_cols + storage_col;
        data()[idx] = value;
    }

    void set_element_as_double(uint64_t i, uint64_t j, double value) override {
        set(i, j, static_cast<T>(value));
    }

    T get(uint64_t i, uint64_t j) const {
        if (i >= rows() || j >= cols()) throw std::out_of_range("Index out of bounds");
        const uint64_t storage_cols = base_cols();
        const uint64_t storage_row = row_offset() + (is_transposed() ? j : i);
        const uint64_t storage_col = col_offset() + (is_transposed() ? i : j);
        const uint64_t idx = storage_row * storage_cols + storage_col;
        return data()[idx];
    }

    void fill(T value) {
        ensure_unique();
        const bool full_span = row_offset() == 0 && col_offset() == 0 && !is_transposed() && rows() == base_rows() && cols() == base_cols();
        if (full_span) {
            T* ptr = data();
            const uint64_t total = base_rows() * base_cols();
            pycauset::ParallelFor(0, total, [&](size_t i) { ptr[i] = value; });
            return;
        }

        for (uint64_t i = 0; i < rows(); ++i) {
            for (uint64_t j = 0; j < cols(); ++j) {
                set(i, j, value);
            }
        }
    }

    void set_identity() {
        if (base_rows() != base_cols()) {
            throw std::invalid_argument("Identity requires a square matrix");
        }
        fill(pycauset::scalar::from_double<T>(0.0));
        const uint64_t n = base_rows();
        for (uint64_t i = 0; i < n; ++i) {
            set(i, i, pycauset::scalar::from_double<T>(1.0));
        }
        set_scalar(1.0);
    }

    double get_element_as_double(uint64_t i, uint64_t j) const override {
        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            throw std::runtime_error("Complex matrix does not support get_element_as_double; use get_element_as_complex");
        }
        const std::complex<double> v = pycauset::scalar::to_complex_double(get(i, j));
        if (scalar_ == 1.0) {
            return v.real();
        }
        return (v * scalar_).real();
    }

    std::complex<double> get_element_as_complex(uint64_t i, uint64_t j) const override {
        const std::complex<double> v = pycauset::scalar::to_complex_double(get(i, j));
        std::complex<double> z = (scalar_ == 1.0) ? v : (v * scalar_);
        if (is_conjugated()) {
            z = std::conj(z);
        }
        return z;
    }

    T* data() { return static_cast<T*>(require_mapper()->get_data()); }
    const T* data() const { return static_cast<const T*>(require_mapper()->get_data()); }

    bool pin_range(size_t start_idx, size_t count) const {
        T* ptr = const_cast<T*>(data()) + start_idx;
        size_t bytes = count * sizeof(T);
        return require_mapper()->pin_region(ptr, bytes);
    }

    void unpin_range(size_t start_idx, size_t count) const {
        T* ptr = const_cast<T*>(data()) + start_idx;
        size_t bytes = count * sizeof(T);
        require_mapper()->unpin_region(ptr, bytes);
    }

    std::unique_ptr<DenseMatrix<T>> submatrix(uint64_t r, uint64_t c, uint64_t h, uint64_t w) const {
        if (r + h > rows() || c + w > cols()) {
            throw std::out_of_range("Submatrix out of bounds");
        }
        // Calculate logic for standard view (no transpose logic here as slicing happens in logical space)
        // However, checks for is_transposed usually affect how we CALCULATE the offsets.
        // Base MatrixBase logic stores PHYSICAL row_offset and col_offset.
        
        uint64_t new_base_row_offset = row_offset();
        uint64_t new_base_col_offset = col_offset();

        if (is_transposed()) {
            // Logical (r,c) maps to Physical (c, r)
            // So we add 'c' to row_offset, and 'r' to col_offset ?
            // No. Logical row 'r' comes from physical col 'r'.
            // Logical col 'c' comes from physical row 'c'.
            // Wait, Transpose means A_logical_ij = A_physical_ji.
            // So logical A(r, c) refers to physical A(c, r).
            // So row offset (physical) increases by c.
            // Col offset (physical) increases by r.
            new_base_row_offset += c;
            new_base_col_offset += r;
        } else {
            new_base_row_offset += r;
            new_base_col_offset += c;
        }

        return std::make_unique<DenseMatrix<T>>(
             h, w,
             mapper_,
             base_rows(), base_cols(),
             new_base_row_offset,
             new_base_col_offset,
             seed_,
             scalar_,
             is_transposed(),
             is_conjugated(),
             is_temporary()
        );
    }

    std::unique_ptr<MatrixBase> multiply_scalar(double factor, const std::string& result_file = "") const override {
        (void)result_file;
        auto out = std::make_unique<DenseMatrix<T>>(rows(),
                                                    cols(),
                                                    mapper_,
                                                    base_rows(),
                                                    base_cols(),
                                                    row_offset(),
                                                    col_offset(),
                                                    seed_,
                                                    scalar_ * factor,
                                                    is_transposed(),
                                                    is_conjugated(),
                                                    is_temporary());
        return out;
    }

    std::unique_ptr<MatrixBase> multiply_scalar(int64_t factor, const std::string& result_file = "") const override {
        return multiply_scalar(static_cast<double>(factor), result_file);
    }

    std::unique_ptr<MatrixBase> multiply_scalar(std::complex<double> factor, const std::string& result_file = "") const override {
        (void)result_file;
        auto out = std::make_unique<DenseMatrix<T>>(rows(),
                                                    cols(),
                                                    mapper_,
                                                    base_rows(),
                                                    base_cols(),
                                                    row_offset(),
                                                    col_offset(),
                                                    seed_,
                                                    scalar_ * factor,
                                                    is_transposed(),
                                                    is_conjugated(),
                                                    is_temporary());
        return out;
    }

    std::unique_ptr<MatrixBase> add_scalar(double scalar, const std::string& result_file = "") const override {
        auto result = std::make_unique<DenseMatrix<T>>(rows(), cols(), result_file);

        const bool full_span = row_offset() == 0 && col_offset() == 0 && !is_transposed() && rows() == base_rows() && cols() == base_cols();

        if (full_span) {
            const T* src_data = data();
            T* dst_data = result->data();
            const uint64_t total_elements = base_rows() * base_cols();

            if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
                const std::complex<double> add = {scalar, 0.0};
                pycauset::ParallelFor(0, total_elements, [&](size_t i) {
                    const std::complex<double> v = pycauset::scalar::to_complex_double(src_data[i]);
                    dst_data[i] = pycauset::scalar::from_complex_double<T>(v * scalar_ + add);
                });
            } else if constexpr (std::is_same_v<T, float>) {
                const float s = static_cast<float>(scalar);
                const float s_self = static_cast<float>(scalar_.real());
                pycauset::ParallelFor(0, total_elements, [&](size_t i) { dst_data[i] = src_data[i] * s_self + s; });
            } else {
                const double s_self = scalar_.real();
                pycauset::ParallelFor(0, total_elements, [&](size_t i) {
                    double val = static_cast<double>(src_data[i]) * s_self + scalar;
                    dst_data[i] = pycauset::scalar::from_double<T>(val);
                });
            }
        } else {
            for (uint64_t i = 0; i < rows(); ++i) {
                for (uint64_t j = 0; j < cols(); ++j) {
                    const auto v = get(i, j);
                    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
                        const std::complex<double> z = pycauset::scalar::to_complex_double(v) * scalar_ + std::complex<double>{scalar, 0.0};
                        result->set(i, j, pycauset::scalar::from_complex_double<T>(z));
                    } else {
                        const double z = static_cast<double>(v) * scalar_.real() + scalar;
                        result->set(i, j, pycauset::scalar::from_double<T>(z));
                    }
                }
            }
        }

        result->set_scalar(1.0);
        result->set_seed(seed_);
        result->set_transposed(is_transposed());
        if (result_file.empty()) {
            result->set_temporary(true);
        }
        return result;
    }

    std::unique_ptr<MatrixBase> add_scalar(int64_t scalar, const std::string& result_file = "") const override {
        return add_scalar(static_cast<double>(scalar), result_file);
    }

    std::unique_ptr<MatrixBase> transpose(const std::string& result_file = "") const override {
        (void)result_file;
        auto out = std::make_unique<DenseMatrix<T>>(logical_rows(),
                                                    logical_cols(),
                                                    mapper_,
                                                    base_rows(),
                                                    base_cols(),
                                                    col_offset(),
                                                    row_offset(),
                                                    seed_,
                                                    scalar_,
                                                    !is_transposed(),
                                                    is_conjugated(),
                                                    is_temporary());
        return out;
    }

    std::unique_ptr<DenseMatrix<T>> bitwise_not(const std::string& result_file = "") const {
        if constexpr (!std::is_integral_v<T> || std::is_same_v<T, bool>) {
            throw std::runtime_error("bitwise_not only supported for integer DenseMatrix types");
        } else {
            auto result = std::make_unique<DenseMatrix<T>>(rows(), cols(), result_file);

            const bool full_span = row_offset() == 0 && col_offset() == 0 && !is_transposed() && rows() == base_rows() && cols() == base_cols();
            if (full_span) {
                const T* src = data();
                T* dst = result->data();
                const uint64_t total = base_rows() * base_cols();
                pycauset::ParallelFor(0, total, [&](size_t i) { dst[i] = static_cast<T>(~src[i]); });
            } else {
                for (uint64_t i = 0; i < rows(); ++i) {
                    for (uint64_t j = 0; j < cols(); ++j) {
                        result->set(i, j, static_cast<T>(~get(i, j)));
                    }
                }
            }

            result->set_scalar(scalar_);
            result->set_seed(seed_);
            result->set_transposed(is_transposed());
            if (result_file.empty()) {
                result->set_temporary(true);
            }
            return result;
        }
    }

    std::unique_ptr<DenseMatrix<T>> multiply(const DenseMatrix<T>& other, const std::string& result_file = "") const {
        if (has_view_offset() || other.has_view_offset()) {
            throw std::runtime_error("matmul on views with offsets is not supported; materialize with copy() first");
        }
        if (cols() != other.rows()) {
            throw std::invalid_argument("Dimension mismatch");
        }

        auto result = std::make_unique<DenseMatrix<T>>(rows(), other.cols(), result_file);
        pycauset::ComputeContext::instance().get_device()->matmul(*this, other, *result);
        return result;
    }

    void inverse_to(DenseMatrix<T>& out) const {
        if (has_view_offset() || out.has_view_offset()) {
            throw std::runtime_error("inverse on views with offsets is not supported; materialize with copy() first");
        }
        if constexpr (!std::is_same_v<T, double> && !std::is_same_v<T, float>) {
            throw std::runtime_error("Inverse only supported for FloatMatrix (double) or Float32Matrix (float)");
        } else {
            if (rows() != cols()) {
                throw std::runtime_error("Inverse requires a square matrix");
            }
            const uint64_t n = rows();
            if (out.rows() != n || out.cols() != n) {
                throw std::runtime_error("Output matrix dimension mismatch");
            }
            pycauset::ComputeContext::instance().get_device()->inverse(*this, out);
        }
    }

    std::unique_ptr<DenseMatrix<T>> inverse(const std::string& result_file = "") const {
        if (has_view_offset()) {
            throw std::runtime_error("inverse on views with offsets is not supported; materialize with copy() first");
        }
        if constexpr (!std::is_same_v<T, double> && !std::is_same_v<T, float>) {
            throw std::runtime_error("Inverse only supported for FloatMatrix (double) or Float32Matrix (float)");
        } else {
            if (rows() != cols()) {
                throw std::runtime_error("Inverse requires a square matrix");
            }
            auto result = std::make_unique<DenseMatrix<T>>(rows(), cols(), result_file);
            pycauset::ComputeContext::instance().get_device()->inverse(*this, *result);
            return result;
        }
    }

    std::pair<std::unique_ptr<DenseMatrix<double>>, std::unique_ptr<DenseMatrix<double>>> qr(const std::string& q_file = "",
                                                                                              const std::string& r_file = "") const {
        if (has_view_offset()) {
            throw std::runtime_error("QR on views with offsets is not supported; materialize with copy() first");
        }
        if constexpr (!std::is_same_v<T, double>) {
            throw std::runtime_error("QR only supported for FloatMatrix");
        } else {
            if (rows() != cols()) {
                throw std::runtime_error("QR requires a square matrix");
            }
            const uint64_t n = rows();
            auto Q = std::make_unique<DenseMatrix<double>>(n, q_file);
            auto R = std::make_unique<DenseMatrix<double>>(n, r_file);

            pycauset::ComputeContext::instance().get_device()->qr(*this, *Q, *R);
            return {std::move(Q), std::move(R)};
        }
    }

    // Pair of (Eigenvalues, Eigenvectors)
    // Eigenavlues: Vector of reals (if sym) or complex (if general). For eigh, it is real.
    // Eigenvectors: Matrix.
    std::pair<std::unique_ptr<VectorBase>, std::unique_ptr<DenseMatrix<T>>> eigh(const std::string& vals_file = "",
                                                                                 const std::string& vecs_file = "") const {
        
        if constexpr (!std::is_same_v<T, double> && !std::is_same_v<T, float>) {
             throw std::runtime_error("eigh only supported for FloatMatrix/Float32Matrix");
        } else {
             // For eigh, eigenvalues are real, so we use DenseVector<T>.
             auto vals = std::make_unique<DenseVector<T>>(rows(), vals_file);
             auto vecs = std::make_unique<DenseMatrix<T>>(rows(), cols(), vecs_file);
             
             pycauset::ComputeContext::instance().get_device()->eigh(*this, *vals, *vecs);
             return {std::move(vals), std::move(vecs)};
        }
    }

    std::unique_ptr<VectorBase> eigvalsh(const std::string& vals_file = "") const {
        if constexpr (!std::is_same_v<T, double> && !std::is_same_v<T, float>) {
             throw std::runtime_error("eigvalsh only supported for FloatMatrix/Float32Matrix");
        } else {
             auto vals = std::make_unique<DenseVector<T>>(rows(), vals_file);
             pycauset::ComputeContext::instance().get_device()->eigvalsh(*this, *vals);
             return vals;
        }
    }

    // General Non-Symmetric Eigenvalue Decomposition
    // Returns (Eigenvalues, Eigenvectors)
    // Eigenvalues are always complex. Eigenvectors are always complex.
    std::pair<std::unique_ptr<VectorBase>, std::unique_ptr<MatrixBase>> eig(const std::string& vals_file = "",
                                                                            const std::string& vecs_file = "") const {
        if constexpr (std::is_same_v<T, double>) {
             auto vals = std::make_unique<DenseVector<std::complex<double>>>(rows(), vals_file);
             auto vecs = std::make_unique<DenseMatrix<std::complex<double>>>(rows(), cols(), vecs_file);
             pycauset::ComputeContext::instance().get_device()->eig(*this, *vals, *vecs);
             return {std::move(vals), std::move(vecs)};
        } 
        else if constexpr (std::is_same_v<T, float>) {
             auto vals = std::make_unique<DenseVector<std::complex<float>>>(rows(), vals_file);
             auto vecs = std::make_unique<DenseMatrix<std::complex<float>>>(rows(), cols(), vecs_file);
             pycauset::ComputeContext::instance().get_device()->eig(*this, *vals, *vecs);
             return {std::move(vals), std::move(vecs)};
        } else {
             throw std::runtime_error("eig only supported for Real Float/Double matrices currently");
        }
    }

    // General Non-Symmetric Eigenvalues only
    std::unique_ptr<VectorBase> eigvals(const std::string& vals_file = "") const {
        if constexpr (std::is_same_v<T, double>) {
             auto vals = std::make_unique<DenseVector<std::complex<double>>>(rows(), vals_file);
             pycauset::ComputeContext::instance().get_device()->eigvals(*this, *vals);
             return vals;
        } 
        else if constexpr (std::is_same_v<T, float>) {
             auto vals = std::make_unique<DenseVector<std::complex<float>>>(rows(), vals_file);
             pycauset::ComputeContext::instance().get_device()->eigvals(*this, *vals);
             return vals;
        } else {
             throw std::runtime_error("eigvals only supported for Real Float/Double matrices currently");
        }
    }

    std::pair<std::unique_ptr<DenseMatrix<double>>, std::unique_ptr<DenseMatrix<double>>> lu(const std::string& l_file = "",
                                                                                              const std::string& u_file = "") const {
        if (has_view_offset()) {
            throw std::runtime_error("LU on views with offsets is not supported; materialize with copy() first");
        }
        if constexpr (!std::is_same_v<T, double>) {
            throw std::runtime_error("LU only supported for FloatMatrix");
        } else {
            if (rows() != cols()) {
                throw std::runtime_error("LU requires a square matrix");
            }
            const uint64_t n = rows();
            auto L = std::make_unique<DenseMatrix<double>>(n, l_file);
            auto U = std::make_unique<DenseMatrix<double>>(n, u_file);

            const double* src = reinterpret_cast<const double*>(data());
            double* u_data = U->data();
            std::copy(src, src + n * n, u_data);

            double* l_data = L->data();
            std::fill(l_data, l_data + n * n, 0.0);
            for (size_t i = 0; i < n; ++i) l_data[i * n + i] = 1.0;

            for (size_t k = 0; k < n; ++k) {
                size_t pivot = k;
                double max_val = std::abs(u_data[k * n + k]);
                for (size_t i = k + 1; i < n; ++i) {
                    double val = std::abs(u_data[i * n + k]);
                    if (val > max_val) {
                        max_val = val;
                        pivot = i;
                    }
                }

                if (max_val < 1e-12) throw std::runtime_error("Matrix is singular");

                if (pivot != k) {
                    for (size_t j = k; j < n; ++j) std::swap(u_data[k * n + j], u_data[pivot * n + j]);
                    for (size_t j = 0; j < k; ++j) std::swap(l_data[k * n + j], l_data[pivot * n + j]);
                }

                double diag = u_data[k * n + k];
                pycauset::ParallelFor(k + 1, n, [&](size_t i) {
                    l_data[i * n + k] = u_data[i * n + k] / diag;
                    u_data[i * n + k] = 0.0;
                });

                pycauset::ParallelFor(k + 1, n, [&](size_t i) {
                    double l_ik = l_data[i * n + k];
                    for (size_t j = k + 1; j < n; ++j) {
                        u_data[i * n + j] -= l_ik * u_data[k * n + j];
                    }
                });
            }

            return {std::move(L), std::move(U)};
        }
    }
};

} // namespace pycauset
