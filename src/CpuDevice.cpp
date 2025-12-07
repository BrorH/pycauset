#include "CpuDevice.hpp"
#include "MatrixOperations.hpp"
#include "ParallelUtils.hpp"
#include "TriangularMatrix.hpp"
#include "DenseMatrix.hpp"
#include "DiagonalMatrix.hpp"
#include "IdentityMatrix.hpp"
#include "Eigen.hpp"
#include "Float16.hpp"
#include <stdexcept>

namespace pycauset {

// Helper template for executing binary operations (Moved from MatrixOperations.cpp)
template <typename T, typename Op>
void execute_binary_op_cpu(const MatrixBase& a, const MatrixBase& b, MatrixBase& result, Op op) {
    uint64_t n = a.size();
    
    // Try to cast to TriangularMatrix<T>
    auto* tri_res = dynamic_cast<TriangularMatrix<T>*>(&result);
    if (tri_res) {
        ParallelFor(0, n, [&](size_t i) {
            for (uint64_t j = i + 1; j < n; ++j) {
                T val = op(static_cast<T>(a.get_element_as_double(i, j)), 
                           static_cast<T>(b.get_element_as_double(i, j)));
                if (val != static_cast<T>(0)) {
                    tri_res->set(i, j, val);
                }
            }
        });
        return;
    }

    // Try to cast to DiagonalMatrix<T>
    auto* diag_res = dynamic_cast<DiagonalMatrix<T>*>(&result);
    if (diag_res) {
        if (result.get_matrix_type() == MatrixType::IDENTITY) {
             if (n > 0) {
                 T val = op(static_cast<T>(a.get_element_as_double(0, 0)), 
                            static_cast<T>(b.get_element_as_double(0, 0)));
                 result.set_scalar(static_cast<double>(val));
             }
             return;
        }

        ParallelFor(0, n, [&](size_t i) {
            T val = op(static_cast<T>(a.get_element_as_double(i, i)), 
                       static_cast<T>(b.get_element_as_double(i, i)));
            diag_res->set(i, i, val);
        });
        return;
    }

    // Try to cast to DenseMatrix<T>
    auto* dense_res = dynamic_cast<DenseMatrix<T>*>(&result);
    if (dense_res) {
        ParallelFor(0, n, [&](size_t i) {
            for (uint64_t j = 0; j < n; ++j) {
                T val = op(static_cast<T>(a.get_element_as_double(i, j)), 
                           static_cast<T>(b.get_element_as_double(i, j)));
                dense_res->set(i, j, val);
            }
        });
        return;
    }

    throw std::runtime_error("Unknown result matrix type in execute_binary_op_cpu");
}

void CpuDevice::matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    auto* a_dense = dynamic_cast<const DenseMatrix<double>*>(&a);
    auto* b_dense = dynamic_cast<const DenseMatrix<double>*>(&b);
    auto* c_dense = dynamic_cast<DenseMatrix<double>*>(&result);

    if (a_dense && b_dense && c_dense) {
        uint64_t n = a.size();
        if (b.size() != n || result.size() != n) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        const double* a_data = a_dense->data();
        const double* b_data = b_dense->data();
        double* c_data = c_dense->data();

        // Initialize result to 0
        std::fill(c_data, c_data + n * n, 0.0);

        bool t_a = a_dense->is_transposed();
        bool t_b = b_dense->is_transposed();
        
        size_t block_size = 64;

        pycauset::ParallelBlockMap(n, n, block_size, [&](size_t i_start, size_t i_end, size_t j_start, size_t j_end) {
            for (size_t k_start = 0; k_start < n; k_start += block_size) {
                size_t k_end = std::min(k_start + block_size, (size_t)n);
                
                if (!t_a && !t_b) {
                    // A * B (Standard)
                    // IKJ algorithm: A sequential, B sequential, C sequential
                    for (size_t i = i_start; i < i_end; ++i) {
                        for (size_t k = k_start; k < k_end; ++k) {
                            double val_a = a_data[i * n + k];
                            if (val_a == 0.0) continue;
                            
                            const double* b_ptr = b_data + k * n;
                            double* c_ptr = c_data + i * n;
                            
                            for (size_t j = j_start; j < j_end; ++j) {
                                c_ptr[j] += val_a * b_ptr[j];
                            }
                        }
                    }
                } else if (!t_a && t_b) {
                    // A * B^T
                    // IJK algorithm (Dot Product): A sequential, B sequential
                    for (size_t i = i_start; i < i_end; ++i) {
                        const double* a_ptr = a_data + i * n;
                        double* c_ptr = c_data + i * n;
                        
                        for (size_t j = j_start; j < j_end; ++j) {
                            double sum = 0;
                            const double* b_ptr = b_data + j * n;
                            
                            for (size_t k = k_start; k < k_end; ++k) {
                                sum += a_ptr[k] * b_ptr[k];
                            }
                            c_ptr[j] += sum;
                        }
                    }
                } else if (t_a && !t_b) {
                    // A^T * B
                    // IKJ algorithm: A stride-N, B sequential
                    for (size_t i = i_start; i < i_end; ++i) {
                        double* c_ptr = c_data + i * n;
                        for (size_t k = k_start; k < k_end; ++k) {
                            double val_a = a_data[k * n + i];
                            if (val_a == 0.0) continue;
                            
                            const double* b_ptr = b_data + k * n;
                            for (size_t j = j_start; j < j_end; ++j) {
                                c_ptr[j] += val_a * b_ptr[j];
                            }
                        }
                    }
                } else {
                    // A^T * B^T
                    // IJK algorithm: A stride-N, B stride-N
                    for (size_t i = i_start; i < i_end; ++i) {
                        double* c_ptr = c_data + i * n;
                        for (size_t j = j_start; j < j_end; ++j) {
                            double sum = 0;
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
        return;
    }

    // Fallback for non-dense matrices or mixed types
    uint64_t n = a.size();
    auto* res_dense = dynamic_cast<DenseMatrix<double>*>(&result);
    if (!res_dense) {
        throw std::runtime_error("CpuDevice::matmul currently only supports DenseMatrix<double> result");
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

void CpuDevice::inverse(const MatrixBase& in, MatrixBase& out) {
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
    
    throw std::runtime_error("CpuDevice::inverse only supports DenseMatrix<double> or DenseMatrix<float>");
}

void CpuDevice::eigvals(const MatrixBase& matrix, ComplexVector& result) {
    pycauset::eigvals_cpu(matrix, result);
}

void CpuDevice::batch_gemv(const MatrixBase& A, const double* x_data, double* y_data, size_t b) {
    uint64_t n = A.size();
    
    // Check for optimized types
    auto* f32 = dynamic_cast<const DenseMatrix<float>*>(&A);
    auto* f16 = dynamic_cast<const DenseMatrix<pycauset::Float16>*>(&A);
    auto* f64 = dynamic_cast<const DenseMatrix<double>*>(&A);

    pycauset::ParallelFor(0, n, [&](size_t i) {
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

} // namespace pycauset
