#include "CpuSolver.hpp"
#include "DenseBitMatrix.hpp"
#include "ParallelUtils.hpp"
#include "DenseMatrix.hpp"
#include "TriangularMatrix.hpp"
#include "DiagonalMatrix.hpp"
#include "IdentityMatrix.hpp"
#include "Eigen.hpp"
#include "Float16.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "CpuSolver.hpp"
#include "DenseBitMatrix.hpp"
#include "ParallelUtils.hpp"
#include "DenseMatrix.hpp"
#include "TriangularMatrix.hpp"
#include "DiagonalMatrix.hpp"
#include "IdentityMatrix.hpp"
#include "Eigen.hpp"
#include "Float16.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace pycauset {

namespace {
    template <typename T>
    void matmul_impl(const DenseMatrix<T>* a_dense, const DenseMatrix<T>* b_dense, DenseMatrix<T>* c_dense) {
        uint64_t n = a_dense->size();
        if (b_dense->size() != n || c_dense->size() != n) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        const T* a_data = a_dense->data();
        const T* b_data = b_dense->data();
        T* c_data = c_dense->data();

        // Initialize result to 0
        std::fill(c_data, c_data + n * n, static_cast<T>(0));

        bool t_a = a_dense->is_transposed();
        bool t_b = b_dense->is_transposed();
        
        size_t block_size = 64;

        ParallelBlockMap(n, n, block_size, [&](size_t i_start, size_t i_end, size_t j_start, size_t j_end) {
            for (size_t k_start = 0; k_start < n; k_start += block_size) {
                size_t k_end = std::min(k_start + block_size, (size_t)n);
                
                if (!t_a && !t_b) {
                    // A * B (Standard)
                    // IKJ algorithm
                    for (size_t i = i_start; i < i_end; ++i) {
                        for (size_t k = k_start; k < k_end; ++k) {
                            T val_a = a_data[i * n + k];
                            if (val_a == static_cast<T>(0)) continue;
                            
                            const T* b_ptr = b_data + k * n;
                            T* c_ptr = c_data + i * n;
                            
                            for (size_t j = j_start; j < j_end; ++j) {
                                c_ptr[j] += val_a * b_ptr[j];
                            }
                        }
                    }
                } else if (!t_a && t_b) {
                    // A * B^T
                    // IJK algorithm (Dot Product)
                    for (size_t i = i_start; i < i_end; ++i) {
                        const T* a_ptr = a_data + i * n;
                        T* c_ptr = c_data + i * n;
                        
                        for (size_t j = j_start; j < j_end; ++j) {
                            T sum = static_cast<T>(0);
                            const T* b_ptr = b_data + j * n;
                            
                            for (size_t k = k_start; k < k_end; ++k) {
                                sum += a_ptr[k] * b_ptr[k];
                            }
                            c_ptr[j] += sum;
                        }
                    }
                } else if (t_a && !t_b) {
                    // A^T * B
                    // IKJ algorithm
                    for (size_t i = i_start; i < i_end; ++i) {
                        T* c_ptr = c_data + i * n;
                        for (size_t k = k_start; k < k_end; ++k) {
                            T val_a = a_data[k * n + i];
                            if (val_a == static_cast<T>(0)) continue;
                            
                            const T* b_ptr = b_data + k * n;
                            for (size_t j = j_start; j < j_end; ++j) {
                                c_ptr[j] += val_a * b_ptr[j];
                            }
                        }
                    }
                } else {
                    // A^T * B^T
                    // IJK algorithm
                    for (size_t i = i_start; i < i_end; ++i) {
                        T* c_ptr = c_data + i * n;
                        for (size_t j = j_start; j < j_end; ++j) {
                            T sum = static_cast<T>(0);
                            for (size_t k = k_start; k < k_end; ++k) {
                                sum += a_data[k * n + i] * b_data[j * n + k];
                            }
                            c_ptr[j] += sum;
                        }
                    }
                }
            }
        });
        
        c_dense->set_scalar(a_dense->get_scalar() * b_dense->get_scalar());
    }
}

void CpuSolver::matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    // Dispatch to specialized implementations
    
    // 1. Double Precision (Float64)
    auto* a_f64 = dynamic_cast<const DenseMatrix<double>*>(&a);
    auto* b_f64 = dynamic_cast<const DenseMatrix<double>*>(&b);
    auto* c_f64 = dynamic_cast<DenseMatrix<double>*>(&result);

    if (a_f64 && b_f64 && c_f64) {
        matmul_impl(a_f64, b_f64, c_f64);
        return;
    }

    // 2. Single Precision (Float32)
    auto* a_f32 = dynamic_cast<const DenseMatrix<float>*>(&a);
    auto* b_f32 = dynamic_cast<const DenseMatrix<float>*>(&b);
    auto* c_f32 = dynamic_cast<DenseMatrix<float>*>(&result);

    if (a_f32 && b_f32 && c_f32) {
        matmul_impl(a_f32, b_f32, c_f32);
        return;
    }

    // 3. Integer (Int32)
    auto* a_i32 = dynamic_cast<const DenseMatrix<int32_t>*>(&a);
    auto* b_i32 = dynamic_cast<const DenseMatrix<int32_t>*>(&b);
    auto* c_i32 = dynamic_cast<DenseMatrix<int32_t>*>(&result);

    if (a_i32 && b_i32 && c_i32) {
        matmul_impl(a_i32, b_i32, c_i32);
        return;
    }

    // 4. BitMatrix Support
    auto* a_bit = dynamic_cast<const DenseMatrix<bool>*>(&a);
    auto* b_bit = dynamic_cast<const DenseMatrix<bool>*>(&b);
    auto* c_int = dynamic_cast<DenseMatrix<int32_t>*>(&result);

    if (a_bit && b_bit && c_int) {
        uint64_t n = a.size();
        ParallelFor(0, n, [&](size_t i) {
            for (size_t j = 0; j < n; ++j) {
                int32_t sum = 0;
                for (size_t k = 0; k < n; ++k) {
                    if (a_bit->get(i, k) && b_bit->get(k, j)) {
                        sum++;
                    }
                }
                c_int->set(i, j, sum);
            }
        });
        return;
    }

    // Fallback for generic types (Parallelized)
    uint64_t n = a.size();
    auto* res_dense = dynamic_cast<DenseMatrix<double>*>(&result);
    if (!res_dense) {
        // If result is not dense double, we might need other handlers or throw
        // For now, let's assume result is dense double as per original CpuDevice logic
        throw std::runtime_error("CpuSolver::matmul currently only supports DenseMatrix<double> result for generic inputs");
    }
    
    ParallelFor(0, n, [&](size_t i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += a.get_element_as_double(i, k) * b.get_element_as_double(k, j);
            }
            res_dense->set(i, j, sum);
        }
    });
}

void CpuSolver::matmul_dense(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    // Legacy wrapper for double precision
    auto* a_dense = dynamic_cast<const DenseMatrix<double>*>(&a);
    auto* b_dense = dynamic_cast<const DenseMatrix<double>*>(&b);
    auto* c_dense = dynamic_cast<DenseMatrix<double>*>(&result);

    if (a_dense && b_dense && c_dense) {
        matmul_impl(a_dense, b_dense, c_dense);
    } else {
        throw std::invalid_argument("matmul_dense called with non-double matrices");
    }
}

void CpuSolver::inverse(const MatrixBase& in, MatrixBase& out) {
    // Double
    if (auto* in_dense = dynamic_cast<const DenseMatrix<double>*>(&in)) {
        if (auto* out_dense = dynamic_cast<DenseMatrix<double>*>(&out)) {
            auto res = in_dense->inverse();
            auto* res_dense = dynamic_cast<DenseMatrix<double>*>(res.get());
            
            uint64_t n = in.size();
            if (out.size() != n) throw std::invalid_argument("Output matrix size mismatch");
            
            const double* src = res_dense->data();
            double* dst = out_dense->data();
            std::copy(src, src + n * n, dst);
            
            out_dense->set_scalar(res_dense->get_scalar());
            return;
        }
    }
    // Float
    if (auto* in_dense = dynamic_cast<const DenseMatrix<float>*>(&in)) {
        if (auto* out_dense = dynamic_cast<DenseMatrix<float>*>(&out)) {
            auto res = in_dense->inverse();
            auto* res_dense = dynamic_cast<DenseMatrix<float>*>(res.get());
            
            uint64_t n = in.size();
            if (out.size() != n) throw std::invalid_argument("Output matrix size mismatch");
            
            const float* src = res_dense->data();
            float* dst = out_dense->data();
            std::copy(src, src + n * n, dst);
            
            out_dense->set_scalar(res_dense->get_scalar());
            return;
        }
    }
    
    throw std::runtime_error("CpuSolver::inverse only supports DenseMatrix<double> or DenseMatrix<float>");
}

void CpuSolver::eigvals(const MatrixBase& matrix, ComplexVector& result) {
    pycauset::eigvals_cpu(matrix, result);
}

void CpuSolver::batch_gemv(const MatrixBase& A, const double* x_data, double* y_data, size_t b) {
    uint64_t n = A.size();
    
    // Check for optimized types
    auto* f32 = dynamic_cast<const DenseMatrix<float>*>(&A);
    auto* f16 = dynamic_cast<const DenseMatrix<pycauset::Float16>*>(&A);
    auto* f64 = dynamic_cast<const DenseMatrix<double>*>(&A);

    ParallelFor(0, n, [&](size_t i) {
        // Accumulator for row i of Y (size b)
        std::vector<double> acc(b, 0.0);
        
        // Scan row i of A
        for (size_t j = 0; j < n; ++j) {
            double a_val;
            if (f32) a_val = (double)f32->get(i, j);
            else if (f16) a_val = (double)f16->get(i, j);
            else if (f64) a_val = f64->get(i, j);
            else a_val = A.get_element_as_double(i, j);

            // SAXPY: acc += a_val * X[j, :]
            size_t x_row_offset = j * b;
            for (size_t k = 0; k < b; ++k) {
                acc[k] += a_val * x_data[x_row_offset + k];
            }
        }

        // Store result
        size_t y_row_offset = i * b;
        for (size_t k = 0; k < b; ++k) {
            y_data[y_row_offset + k] = acc[k];
        }
    });
}

namespace {
    template <typename T, typename Op>
    void binary_op_impl(const MatrixBase& a, const MatrixBase& b, DenseMatrix<T>* res, Op op) {
        uint64_t n = a.size();
        auto* a_dense = dynamic_cast<const DenseMatrix<T>*>(&a);
        auto* b_dense = dynamic_cast<const DenseMatrix<T>*>(&b);
        
        if (a_dense && b_dense && 
            a_dense->get_scalar() == 1.0 && !a_dense->is_transposed() &&
            b_dense->get_scalar() == 1.0 && !b_dense->is_transposed()) {
            
            const T* a_data = a_dense->data();
            const T* b_data = b_dense->data();
            T* res_data = res->data();
            
            ParallelFor(0, n * n, [&](size_t i) {
                res_data[i] = op(a_data[i], b_data[i]);
            });
            return;
        }
        
        ParallelFor(0, n, [&](size_t i) {
            for (size_t j = 0; j < n; ++j) {
                T val_a = a_dense ? a_dense->get(i, j) : static_cast<T>(a.get_element_as_double(i, j));
                T val_b = b_dense ? b_dense->get(i, j) : static_cast<T>(b.get_element_as_double(i, j));
                res->set(i, j, op(val_a, val_b));
            }
        });
    }

    template <typename T>
    void scalar_op_impl(const MatrixBase& a, double scalar, DenseMatrix<T>* res) {
        uint64_t n = a.size();
        auto* a_dense = dynamic_cast<const DenseMatrix<T>*>(&a);
        
        if (a_dense && a_dense->get_scalar() == 1.0 && !a_dense->is_transposed()) {
            const T* a_data = a_dense->data();
            T* res_data = res->data();
            T s = static_cast<T>(scalar);
            ParallelFor(0, n * n, [&](size_t i) {
                res_data[i] = a_data[i] * s;
            });
            return;
        }
        
        ParallelFor(0, n, [&](size_t i) {
            for (size_t j = 0; j < n; ++j) {
                T val_a = a_dense ? a_dense->get(i, j) : static_cast<T>(a.get_element_as_double(i, j));
                res->set(i, j, val_a * static_cast<T>(scalar));
            }
        });
    }
}

void CpuSolver::add(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    if (auto* r = dynamic_cast<DenseMatrix<double>*>(&result)) {
        binary_op_impl(a, b, r, std::plus<>());
    } else if (auto* r = dynamic_cast<DenseMatrix<float>*>(&result)) {
        binary_op_impl(a, b, r, std::plus<>());
    } else {
        throw std::runtime_error("CpuSolver::add result type not supported");
    }
}

void CpuSolver::subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    if (auto* r = dynamic_cast<DenseMatrix<double>*>(&result)) {
        binary_op_impl(a, b, r, std::minus<>());
    } else if (auto* r = dynamic_cast<DenseMatrix<float>*>(&result)) {
        binary_op_impl(a, b, r, std::minus<>());
    } else {
        throw std::runtime_error("CpuSolver::subtract result type not supported");
    }
}

void CpuSolver::multiply_scalar(const MatrixBase& a, double scalar, MatrixBase& result) {
    if (auto* r = dynamic_cast<DenseMatrix<double>*>(&result)) {
        scalar_op_impl(a, scalar, r);
    } else if (auto* r = dynamic_cast<DenseMatrix<float>*>(&result)) {
        scalar_op_impl(a, scalar, r);
    } else {
        throw std::runtime_error("CpuSolver::multiply_scalar result type not supported");
    }
}

} // namespace pycauset

