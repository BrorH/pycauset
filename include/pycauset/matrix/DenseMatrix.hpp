#pragma once

#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/core/MatrixTraits.hpp"
#include "pycauset/core/StorageUtils.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/compute/ComputeContext.hpp"
#include "pycauset/core/ScalarUtils.hpp"

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
    DenseMatrix(uint64_t n, const std::string& backing_file = "")
        : MatrixBase(n, pycauset::MatrixType::DENSE_FLOAT, MatrixTraits<T>::data_type) {
        const uint64_t size_in_bytes = n * n * sizeof(T);
        initialize_storage(size_in_bytes,
                           backing_file,
                           std::string("dense_") + MatrixTraits<T>::name,
                           sizeof(T),
                           pycauset::MatrixType::DENSE_FLOAT,
                           MatrixTraits<T>::data_type,
                           n,
                           n);
    }

    DenseMatrix(uint64_t n,
                const std::string& backing_file,
                size_t offset,
                uint64_t seed,
                std::complex<double> scalar,
                bool is_transposed)
        : MatrixBase(n, pycauset::MatrixType::DENSE_FLOAT, MatrixTraits<T>::data_type) {
        const uint64_t size_in_bytes = n * n * sizeof(T);
        initialize_storage(size_in_bytes,
                           backing_file,
                           "",
                           sizeof(T),
                           pycauset::MatrixType::DENSE_FLOAT,
                           MatrixTraits<T>::data_type,
                           n,
                           n,
                           offset,
                           false);

        set_seed(seed);
        set_scalar(scalar);
        set_transposed(is_transposed);
    }

    DenseMatrix(uint64_t n, std::shared_ptr<MemoryMapper> mapper)
        : MatrixBase(n, std::move(mapper), pycauset::MatrixType::DENSE_FLOAT, MatrixTraits<T>::data_type) {}

    void set(uint64_t i, uint64_t j, T value) {
        ensure_unique();
        if (i >= n_ || j >= n_) throw std::out_of_range("Index out of bounds");
        const uint64_t idx = is_transposed() ? (j * n_ + i) : (i * n_ + j);
        data()[idx] = value;
    }

    T get(uint64_t i, uint64_t j) const {
        if (i >= n_ || j >= n_) throw std::out_of_range("Index out of bounds");
        const uint64_t idx = is_transposed() ? (j * n_ + i) : (i * n_ + j);
        return data()[idx];
    }

    void fill(T value) {
        ensure_unique();
        T* ptr = data();
        const uint64_t total = n_ * n_;
        pycauset::ParallelFor(0, total, [&](size_t i) { ptr[i] = value; });
        cached_trace_ = std::nullopt;
        cached_determinant_ = std::nullopt;
    }

    void set_identity() {
        fill(pycauset::scalar::from_double<T>(0.0));
        for (uint64_t i = 0; i < n_; ++i) {
            set(i, i, pycauset::scalar::from_double<T>(1.0));
        }
        set_scalar(1.0);
        cached_trace_ = std::nullopt;
        cached_determinant_ = std::nullopt;
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

    std::unique_ptr<MatrixBase> multiply_scalar(double factor, const std::string& result_file = "") const override {
        std::string new_path = copy_storage(result_file);
        auto mapper = std::make_unique<MemoryMapper>(new_path, 0, false);
        auto out = std::make_unique<DenseMatrix<T>>(n_, std::move(mapper));
        out->set_scalar(scalar_ * factor);
        out->set_seed(seed_);
        out->set_transposed(is_transposed());
        out->set_conjugated(is_conjugated());
        if (result_file.empty()) {
            out->set_temporary(true);
        }
        return out;
    }

    std::unique_ptr<MatrixBase> multiply_scalar(int64_t factor, const std::string& result_file = "") const override {
        return multiply_scalar(static_cast<double>(factor), result_file);
    }

    std::unique_ptr<MatrixBase> multiply_scalar(std::complex<double> factor, const std::string& result_file = "") const override {
        std::string new_path = copy_storage(result_file);
        auto mapper = std::make_unique<MemoryMapper>(new_path, 0, false);
        auto out = std::make_unique<DenseMatrix<T>>(n_, std::move(mapper));
        out->set_scalar(scalar_ * factor);
        out->set_seed(seed_);
        out->set_transposed(is_transposed());
        out->set_conjugated(is_conjugated());
        if (result_file.empty()) {
            out->set_temporary(true);
        }
        return out;
    }

    std::unique_ptr<MatrixBase> add_scalar(double scalar, const std::string& result_file = "") const override {
        auto result = std::make_unique<DenseMatrix<T>>(n_, result_file);

        const T* src_data = data();
        T* dst_data = result->data();
        const uint64_t total_elements = n_ * n_;

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
        std::string new_path = copy_storage(result_file);
        auto mapper = std::make_unique<MemoryMapper>(new_path, 0, false);
        auto out = std::make_unique<DenseMatrix<T>>(n_, std::move(mapper));
        out->set_transposed(!is_transposed());
        out->set_scalar(scalar_);
        out->set_seed(seed_);
        out->set_conjugated(is_conjugated());
        if (result_file.empty()) {
            out->set_temporary(true);
        }
        return out;
    }

    std::unique_ptr<DenseMatrix<T>> bitwise_not(const std::string& result_file = "") const {
        if constexpr (!std::is_integral_v<T> || std::is_same_v<T, bool>) {
            throw std::runtime_error("bitwise_not only supported for integer DenseMatrix types");
        } else {
            auto result = std::make_unique<DenseMatrix<T>>(n_, result_file);
            const T* src = data();
            T* dst = result->data();
            const uint64_t total = n_ * n_;
            pycauset::ParallelFor(0, total, [&](size_t i) { dst[i] = static_cast<T>(~src[i]); });
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
        if (n_ != other.size()) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        auto result = std::make_unique<DenseMatrix<T>>(n_, result_file);
        pycauset::ComputeContext::instance().get_device()->matmul(*this, other, *result);
        return result;
    }

    void inverse_to(DenseMatrix<T>& out) const {
        if constexpr (!std::is_same_v<T, double> && !std::is_same_v<T, float>) {
            throw std::runtime_error("Inverse only supported for FloatMatrix (double) or Float32Matrix (float)");
        } else {
            if (out.rows() != n_ || out.cols() != n_) {
                throw std::runtime_error("Output matrix dimension mismatch");
            }
            pycauset::ComputeContext::instance().get_device()->inverse(*this, out);
        }
    }

    std::unique_ptr<DenseMatrix<T>> inverse(const std::string& result_file = "") const {
        if constexpr (!std::is_same_v<T, double> && !std::is_same_v<T, float>) {
            throw std::runtime_error("Inverse only supported for FloatMatrix (double) or Float32Matrix (float)");
        } else {
            auto result = std::make_unique<DenseMatrix<T>>(n_, result_file);
            pycauset::ComputeContext::instance().get_device()->inverse(*this, *result);
            return result;
        }
    }

    std::pair<std::unique_ptr<DenseMatrix<double>>, std::unique_ptr<DenseMatrix<double>>> qr(const std::string& q_file = "",
                                                                                              const std::string& r_file = "") const {
        if constexpr (!std::is_same_v<T, double>) {
            throw std::runtime_error("QR only supported for FloatMatrix");
        } else {
            auto Q = std::make_unique<DenseMatrix<double>>(n_, q_file);
            auto R = std::make_unique<DenseMatrix<double>>(n_, r_file);

            const double* src = reinterpret_cast<const double*>(data());
            double* q_data = Q->data();
            std::copy(src, src + n_ * n_, q_data);

            double* r_data = R->data();
            std::fill(r_data, r_data + n_ * n_, 0.0);

            for (size_t k = 0; k < n_; ++k) {
                double norm_sq = 0.0;
                for (size_t i = 0; i < n_; ++i) {
                    double val = q_data[i * n_ + k];
                    norm_sq += val * val;
                }
                double norm = std::sqrt(norm_sq);
                r_data[k * n_ + k] = norm;

                if (norm > 1e-12) {
                    double inv_norm = 1.0 / norm;
                    pycauset::ParallelFor(0, n_, [&](size_t i) { q_data[i * n_ + k] *= inv_norm; });
                } else {
                    pycauset::ParallelFor(0, n_, [&](size_t i) { q_data[i * n_ + k] = 0.0; });
                }

                pycauset::ParallelFor(k + 1, n_, [&](size_t j) {
                    double dot = 0.0;
                    for (size_t i = 0; i < n_; ++i) {
                        dot += q_data[i * n_ + k] * q_data[i * n_ + j];
                    }
                    r_data[k * n_ + j] = dot;

                    for (size_t i = 0; i < n_; ++i) {
                        q_data[i * n_ + j] -= dot * q_data[i * n_ + k];
                    }
                });
            }

            return {std::move(Q), std::move(R)};
        }
    }

    std::pair<std::unique_ptr<DenseMatrix<double>>, std::unique_ptr<DenseMatrix<double>>> lu(const std::string& l_file = "",
                                                                                              const std::string& u_file = "") const {
        if constexpr (!std::is_same_v<T, double>) {
            throw std::runtime_error("LU only supported for FloatMatrix");
        } else {
            auto L = std::make_unique<DenseMatrix<double>>(n_, l_file);
            auto U = std::make_unique<DenseMatrix<double>>(n_, u_file);

            const double* src = reinterpret_cast<const double*>(data());
            double* u_data = U->data();
            std::copy(src, src + n_ * n_, u_data);

            double* l_data = L->data();
            std::fill(l_data, l_data + n_ * n_, 0.0);
            for (size_t i = 0; i < n_; ++i) l_data[i * n_ + i] = 1.0;

            for (size_t k = 0; k < n_; ++k) {
                size_t pivot = k;
                double max_val = std::abs(u_data[k * n_ + k]);
                for (size_t i = k + 1; i < n_; ++i) {
                    double val = std::abs(u_data[i * n_ + k]);
                    if (val > max_val) {
                        max_val = val;
                        pivot = i;
                    }
                }

                if (max_val < 1e-12) throw std::runtime_error("Matrix is singular");

                if (pivot != k) {
                    for (size_t j = k; j < n_; ++j) std::swap(u_data[k * n_ + j], u_data[pivot * n_ + j]);
                    for (size_t j = 0; j < k; ++j) std::swap(l_data[k * n_ + j], l_data[pivot * n_ + j]);
                }

                double diag = u_data[k * n_ + k];
                pycauset::ParallelFor(k + 1, n_, [&](size_t i) {
                    l_data[i * n_ + k] = u_data[i * n_ + k] / diag;
                    u_data[i * n_ + k] = 0.0;
                });

                pycauset::ParallelFor(k + 1, n_, [&](size_t i) {
                    double l_ik = l_data[i * n_ + k];
                    for (size_t j = k + 1; j < n_; ++j) {
                        u_data[i * n_ + j] -= l_ik * u_data[k * n_ + j];
                    }
                });
            }

            return {std::move(L), std::move(U)};
        }
    }
};

} // namespace pycauset
