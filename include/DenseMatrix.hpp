#pragma once

#include "MatrixBase.hpp"
#include "MatrixTraits.hpp"
#include "StoragePaths.hpp"
#include "ParallelUtils.hpp"
#include "ComputeContext.hpp"
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>
#include <vector>
#include <type_traits>

template <typename T>
class DenseMatrix : public MatrixBase {
public:
    DenseMatrix(uint64_t n, const std::string& backing_file = "")
        : MatrixBase(n, pycauset::MatrixType::DENSE_FLOAT, MatrixTraits<T>::data_type) {
        uint64_t size_in_bytes = n * n * sizeof(T);
        initialize_storage(size_in_bytes, backing_file, 
                         std::string("dense_") + MatrixTraits<T>::name, 
                         sizeof(T),
                         pycauset::MatrixType::DENSE_FLOAT, 
                         MatrixTraits<T>::data_type,
                         n, n);
    }

    // Constructor for loading with explicit metadata
    DenseMatrix(uint64_t n, 
                const std::string& backing_file,
                size_t offset,
                uint64_t seed,
                double scalar,
                bool is_transposed)
        : MatrixBase(n, pycauset::MatrixType::DENSE_FLOAT, MatrixTraits<T>::data_type) {
        
        uint64_t size_in_bytes = n * n * sizeof(T);
        initialize_storage(size_in_bytes, backing_file, 
                         "", // No fallback needed for loading
                         sizeof(T),
                         pycauset::MatrixType::DENSE_FLOAT, 
                         MatrixTraits<T>::data_type,
                         n, n,
                         offset,
                         false); // Do not create new file, open existing
        
        set_seed(seed);
        set_scalar(scalar);
        set_transposed(is_transposed);
    }

    DenseMatrix(uint64_t n, std::unique_ptr<MemoryMapper> mapper)
        : MatrixBase(n, std::move(mapper), pycauset::MatrixType::DENSE_FLOAT, MatrixTraits<T>::data_type) {}

    void set(uint64_t i, uint64_t j, T value) {
        if (i >= n_ || j >= n_) throw std::out_of_range("Index out of bounds");
        if (is_transposed()) {
            data()[j * n_ + i] = value;
        } else {
            data()[i * n_ + j] = value;
        }
    }

    T get(uint64_t i, uint64_t j) const {
        if (i >= n_ || j >= n_) throw std::out_of_range("Index out of bounds");
        if (is_transposed()) {
            return data()[j * n_ + i];
        }
        return data()[i * n_ + j];
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
        auto new_matrix = std::make_unique<DenseMatrix<T>>(n_, std::move(mapper));
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
        
        const T* src_data = data();
        T* dst_data = result->data();
        uint64_t total_elements = n_ * n_;
        
        // Use ParallelFor for large matrices
        // And avoid double promotion if T is float
        if constexpr (std::is_same_v<T, float>) {
            float s = static_cast<float>(scalar);
            float s_self = static_cast<float>(scalar_);
            pycauset::ParallelFor(0, total_elements, [&](size_t i) {
                dst_data[i] = src_data[i] * s_self + s;
            });
        } else {
            pycauset::ParallelFor(0, total_elements, [&](size_t i) {
                double val = static_cast<double>(src_data[i]) * scalar_ + scalar;
                dst_data[i] = static_cast<T>(val);
            });
        }
        
        result->set_scalar(1.0);
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
        auto new_matrix = std::make_unique<DenseMatrix<T>>(n_, std::move(mapper));
        
        // Flip the transposed bit
        new_matrix->set_transposed(!this->is_transposed());
        
        if (result_file.empty()) {
            new_matrix->set_temporary(true);
        }
        return new_matrix;
    }

    std::unique_ptr<DenseMatrix<T>> bitwise_not(const std::string& result_file = "") const {
        auto result = std::make_unique<DenseMatrix<T>>(n_, result_file);
        
        const T* src_data = data();
        T* dst_data = result->data();
        
        uint64_t total_elements = n_ * n_;
        
        // Treat as raw bits for bitwise not
        if constexpr (std::is_floating_point_v<T>) {
            const uint64_t* src_bits = reinterpret_cast<const uint64_t*>(src_data);
            uint64_t* dst_bits = reinterpret_cast<uint64_t*>(dst_data);
            // Assuming 64-bit double. For float it would be 32-bit.
            // This implementation assumes T=double.
            for (uint64_t i = 0; i < total_elements; ++i) {
                dst_bits[i] = ~src_bits[i];
            }
        } else {
            for (uint64_t i = 0; i < total_elements; ++i) {
                dst_data[i] = ~src_data[i];
            }
        }
        
        result->set_scalar(scalar_);
        return result;
    }

    std::unique_ptr<DenseMatrix<T>> multiply(const DenseMatrix<T>& other, const std::string& result_file = "") const {
        if (n_ != other.size()) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        auto result = std::make_unique<DenseMatrix<T>>(n_, result_file);
        
        // Delegate to ComputeContext (AutoSolver)
        pycauset::ComputeContext::instance().get_device()->matmul(*this, other, *result);
        
        return result;
    }

    // Inverse implementation
    std::unique_ptr<DenseMatrix<T>> inverse(const std::string& result_file = "") const {
        if constexpr (!std::is_same_v<T, double> && !std::is_same_v<T, float>) {
            throw std::runtime_error("Inverse only supported for FloatMatrix (double) or Float32Matrix (float)");
        } else {
            auto result = std::make_unique<DenseMatrix<T>>(n_, result_file);
            
            // Delegate to ComputeContext (AutoSolver)
            pycauset::ComputeContext::instance().get_device()->inverse(*this, *result);
            
            return result;
        }
    }

    // QR Decomposition (Modified Gram-Schmidt)
    // Returns {Q, R} where A = QR
    std::pair<std::unique_ptr<DenseMatrix<double>>, std::unique_ptr<DenseMatrix<double>>> qr(const std::string& q_file = "", const std::string& r_file = "") const {
        if constexpr (!std::is_same_v<T, double>) {
            throw std::runtime_error("QR only supported for FloatMatrix");
        } else {
            auto Q = std::make_unique<DenseMatrix<double>>(n_, q_file);
            auto R = std::make_unique<DenseMatrix<double>>(n_, r_file);
            
            // Copy A to Q (we will orthogonalize Q in-place)
            const double* src = reinterpret_cast<const double*>(data());
            double* q_data = Q->data();
            std::copy(src, src + n_ * n_, q_data);
            
            double* r_data = R->data();
            std::fill(r_data, r_data + n_ * n_, 0.0);

            // Parallel Modified Gram-Schmidt
            // Note: MGS is inherently sequential in k, but parallel in the inner loops.
            
            for (size_t k = 0; k < n_; ++k) {
                // 1. Compute norm of column k (Parallel Reduction)
                double norm_sq = 0.0;
                // We can't easily reduce a scalar in ParallelFor without atomic or thread-local accumulation.
                // For N=5000, a simple loop is fast enough for one column (5000 elements).
                // But let's try to use ParallelFor with chunks if needed.
                // Actually, for column-major access (which this is NOT, it's row-major), accessing column k is strided.
                // q_data[i * n_ + k]
                
                // Strided access is bad for cache.
                // But we have to live with it or transpose.
                
                for (size_t i = 0; i < n_; ++i) {
                    double val = q_data[i * n_ + k];
                    norm_sq += val * val;
                }
                double norm = std::sqrt(norm_sq);
                
                r_data[k * n_ + k] = norm;
                
                if (norm > 1e-12) {
                    double inv_norm = 1.0 / norm;
                    // Scale column k (Parallel)
                    pycauset::ParallelFor(0, n_, [&](size_t i) {
                        q_data[i * n_ + k] *= inv_norm;
                    });
                } else {
                    // Singular
                    pycauset::ParallelFor(0, n_, [&](size_t i) {
                        q_data[i * n_ + k] = 0.0;
                    });
                }

                // 2. Orthogonalize remaining columns against column k
                // For j = k+1 to n
                // R[k, j] = Q[:, k] . Q[:, j]
                // Q[:, j] -= R[k, j] * Q[:, k]
                
                // This loop over j can be parallelized!
                // Each column j is independent.
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

    // LU Decomposition (Doolittle Algorithm with Partial Pivoting)
    // Returns {L, U, P} where PA = LU
    // For simplicity, we return L and U. P is applied implicitly or we can return it.
    // Currently returns {L, U}
    std::pair<std::unique_ptr<DenseMatrix<double>>, std::unique_ptr<DenseMatrix<double>>> lu(const std::string& l_file = "", const std::string& u_file = "") const {
        if constexpr (!std::is_same_v<T, double>) {
            throw std::runtime_error("LU only supported for FloatMatrix");
        } else {
            auto L = std::make_unique<DenseMatrix<double>>(n_, l_file);
            auto U = std::make_unique<DenseMatrix<double>>(n_, u_file);
            
            // Copy A to U (we will work on U in-place)
            const double* src = reinterpret_cast<const double*>(data());
            double* u_data = U->data();
            std::copy(src, src + n_ * n_, u_data);
            
            double* l_data = L->data();
            std::fill(l_data, l_data + n_ * n_, 0.0);
            for(size_t i=0; i<n_; ++i) l_data[i*n_ + i] = 1.0; // Unit diagonal for L

            // Parallel LU (Right-looking)
            for (size_t k = 0; k < n_; ++k) {
                // 1. Pivot (Sequential)
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
                    // Swap rows in U
                    for (size_t j = k; j < n_; ++j) std::swap(u_data[k * n_ + j], u_data[pivot * n_ + j]);
                    // Swap rows in L (only columns 0..k-1)
                    for (size_t j = 0; j < k; ++j) std::swap(l_data[k * n_ + j], l_data[pivot * n_ + j]);
                    // Note: We are not returning P, so this is PA = LU. 
                    // The user must know that rows were swapped. 
                    // For a proper API, we should return P.
                }

                // 2. Compute L column (Parallel)
                double diag = u_data[k * n_ + k];
                pycauset::ParallelFor(k + 1, n_, [&](size_t i) {
                    l_data[i * n_ + k] = u_data[i * n_ + k] / diag;
                    u_data[i * n_ + k] = 0.0; // Strictly upper triangular
                });

                // 3. Update Trailing Submatrix (Parallel)
                // U[i, j] = U[i, j] - L[i, k] * U[k, j]
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
