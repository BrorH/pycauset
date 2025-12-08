#include "pycauset/compute/cpu/CpuSolver.hpp"
#include "pycauset/vector/ComplexVector.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/vector/DenseVector.hpp"
#include "pycauset/vector/UnitVector.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/matrix/IdentityMatrix.hpp"
#include "pycauset/math/Eigen.hpp"
#include "pycauset/core/Float16.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/core/MemoryHints.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <bit>
#include <filesystem>

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
        
        // --- Lookahead Protocol: Send Memory Hints ---
        using namespace pycauset::core;
        size_t total_bytes = n * n * sizeof(T);
        size_t stride_bytes = n * sizeof(T);
        size_t block_bytes = sizeof(T);

        // Hint A
        if (t_a) {
            // A is transposed, so we access it column-wise (Strided)
            a_dense->hint(MemoryHint::strided(0, total_bytes, stride_bytes, block_bytes));
        } else {
            // A is normal, accessed row-wise (Sequential)
            a_dense->hint(MemoryHint::sequential(0, total_bytes));
        }

        // Hint B
        // B is always accessed sequentially in the inner loops of our optimized kernels
        // (See logic below: we switch loop order to favor B's row-major layout)
        b_dense->hint(MemoryHint::sequential(0, total_bytes));

        // Hint C (Write)
        c_dense->hint(MemoryHint::sequential(0, total_bytes));
        // ---------------------------------------------

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

    template <typename T>
    void inverse_impl(const DenseMatrix<T>* in_dense, DenseMatrix<T>* out_dense) {
        uint64_t n = in_dense->size();
        if (out_dense->size() != n) throw std::invalid_argument("Output matrix size mismatch");
        if (in_dense->get_scalar() == 0.0) throw std::runtime_error("Matrix scalar is 0, cannot invert");

        // Create temporary work matrix
        std::string work_path = pycauset::make_unique_storage_file("inverse_work");
        // We need to copy input to work, but input might be transposed or scaled.
        // The simplest way is to copy element by element or use copy_storage if not transposed.
        // But copy_storage creates a file.
        // Let's just create a new matrix and copy data.
        
        // Actually, we can use MemoryMapper directly to avoid full DenseMatrix overhead if we want,
        // but using DenseMatrix is cleaner.
        
        // Note: We need to handle the scalar. The inverse of (s*A) is (1/s)*A^-1.
        // We will invert the raw data and then set the scalar of result.
        
        // Create work matrix (copy of input raw data)
        // If input is transposed, we need to handle that.
        // Gauss-Jordan works on rows. If transposed, we are working on columns.
        // Inverse of A^T is (A^-1)^T.
        // So if input is transposed, we can invert the non-transposed data and then transpose the result?
        // Or just copy data into work matrix in correct order.
        
        uint64_t size_bytes = n * n * sizeof(T);
        auto work_mapper = std::make_unique<MemoryMapper>(work_path, size_bytes, 0, true); // Create new file
        DenseMatrix<T> work(n, std::move(work_mapper));
        work.set_temporary(true);
        
        T* w = work.data();
        T* r = out_dense->data();
        
        // Initialize Work = Input, Result = Identity
        const T* in_data = in_dense->data();
        bool is_transposed = in_dense->is_transposed();
        
        ParallelFor(0, n, [&](size_t i) {
            for (size_t j = 0; j < n; ++j) {
                // Copy input to work
                if (is_transposed) {
                    w[i * n + j] = in_data[j * n + i];
                } else {
                    w[i * n + j] = in_data[i * n + j];
                }
                
                // Initialize result to Identity
                if (i == j) {
                    r[i * n + j] = (T)1.0;
                } else {
                    r[i * n + j] = (T)0.0;
                }
            }
        });

        // Block Gauss-Jordan
        size_t block_size = 64;
        T epsilon = std::is_same_v<T, float> ? 1e-5f : 1e-12;
        T zero_threshold = std::is_same_v<T, float> ? 1e-7f : 1e-15;
        
        for (size_t k_start = 0; k_start < n; k_start += block_size) {
            size_t k_end = std::min(k_start + block_size, (size_t)n);
            
            for (size_t i = k_start; i < k_end; ++i) {
                // Pivot
                size_t pivot = i;
                T max_val = std::abs(w[i * n + i]);
                for (size_t k = i + 1; k < n; ++k) {
                    T val = std::abs(w[k * n + i]);
                    if (val > max_val) {
                        max_val = val;
                        pivot = k;
                    }
                }

                if (max_val < epsilon) {
                    work.close();
                    std::filesystem::remove(work_path);
                    throw std::runtime_error("Matrix is singular or nearly singular");
                }

                if (pivot != i) {
                    for (size_t j = 0; j < n; ++j) {
                        std::swap(w[i * n + j], w[pivot * n + j]);
                        std::swap(r[i * n + j], r[pivot * n + j]);
                    }
                }

                T div = w[i * n + i];
                T inv_div = (T)1.0 / div;
                
                if (n > 1000) {
                    ParallelFor(0, n, [&](size_t j) {
                        w[i * n + j] *= inv_div;
                        r[i * n + j] *= inv_div;
                    });
                } else {
                    for (size_t j = 0; j < n; ++j) {
                        w[i * n + j] *= inv_div;
                        r[i * n + j] *= inv_div;
                    }
                }
                w[i * n + i] = (T)1.0;

                for (size_t k = k_start; k < k_end; ++k) {
                    if (k != i) {
                        T factor = w[k * n + i];
                        if (std::abs(factor) > zero_threshold) {
                            w[k * n + i] = (T)0.0;
                            for (size_t j = i + 1; j < n; ++j) w[k * n + j] -= factor * w[i * n + j];
                            for (size_t j = 0; j < n; ++j) r[k * n + j] -= factor * r[i * n + j];
                        }
                    }
                }
            }

            auto update_rows = [&](size_t r_start, size_t r_end) {
                ParallelFor(r_start, r_end, [&](size_t i) {
                    for (size_t k = k_start; k < k_end; ++k) {
                        double factor = w[i * n + k];
                        if (std::abs(factor) > 1e-15) {
                            w[i * n + k] = 0.0;
                            for (size_t j = k_end; j < n; ++j) w[i * n + j] -= factor * w[k * n + j];
                            for (size_t j = 0; j < n; ++j) r[i * n + j] -= factor * r[k * n + j];
                        }
                    }
                });
            };

            if (k_start > 0) update_rows(0, k_start);
            if (k_end < n) update_rows(k_end, n);
        }
        
        if (in_dense->get_scalar() != 1.0) {
            out_dense->set_scalar(1.0 / in_dense->get_scalar());
        }
        
        // If input was transposed, the result we calculated is (A^T)^-1 = (A^-1)^T.
        // So the result data is transposed relative to the true inverse.
        // We should mark the output as transposed?
        // But we copied data handling transposition at the start (w[i,j] = in[j,i]).
        // So 'w' represents A^T. We inverted A^T. Result 'r' is (A^T)^-1 = (A^-1)^T.
        // So 'r' contains the transpose of the inverse.
        // If we want the inverse, we should transpose 'r'.
        // OR, we can just say out_dense->set_transposed(true).
        if (is_transposed) {
            out_dense->set_transposed(true);
        }
        
        work.close();
        std::filesystem::remove(work_path);
    }
} // namespace

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

    // 4. Half Precision (Float16)
    auto* a_f16 = dynamic_cast<const DenseMatrix<pycauset::Float16>*>(&a);
    auto* b_f16 = dynamic_cast<const DenseMatrix<pycauset::Float16>*>(&b);
    auto* c_f16 = dynamic_cast<DenseMatrix<pycauset::Float16>*>(&result);

    if (a_f16 && b_f16 && c_f16) {
        matmul_impl(a_f16, b_f16, c_f16);
        return;
    }

    // 5. BitMatrix Support
    auto* a_bit = dynamic_cast<const DenseMatrix<bool>*>(&a);
    auto* b_bit = dynamic_cast<const DenseMatrix<bool>*>(&b);
    auto* c_int = dynamic_cast<DenseMatrix<int32_t>*>(&result);

    if (a_bit && b_bit && c_int) {
        // ... (existing bit matrix code) ...
        uint64_t n = a.size();
        
        // Optimized implementation using popcount and transpose
        // We need access to the raw data, which is protected/private in DenseMatrix<bool>
        // But DenseMatrix<bool> exposes data() as public method returning uint64_t*
        
        const uint64_t* a_data = a_bit->data();
        uint64_t stride_bytes = a_bit->stride_bytes();
        uint64_t words_per_row = stride_bytes / 8;
        
        // Transpose B for cache efficiency and row-row operations
        // Use DenseBitMatrix to allow spilling to disk if needed
        auto b_transposed_mat = std::make_unique<DenseBitMatrix>(n, "");
        b_transposed_mat->set_temporary(true);
        uint64_t* b_transposed_data = b_transposed_mat->data();
        
        // Single-threaded transpose for now (or use atomic OR)
        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t j = 0; j < n; ++j) {
                if (b_bit->get(i, j)) {
                    // Set (j, i) in transposed
                    uint64_t word_idx = i / 64;
                    uint64_t bit_idx = i % 64;
                    b_transposed_data[j * words_per_row + word_idx] |= (1ULL << bit_idx);
                }
            }
        }

        // Parallel Matrix Multiplication
        ParallelFor(0, n, [&](size_t i) {
            const uint64_t* a_row = a_data + i * words_per_row;
            for (size_t j = 0; j < n; ++j) {
                const uint64_t* b_col = b_transposed_data + j * words_per_row;
                
                int32_t dot_product = 0;
                for (size_t k = 0; k < words_per_row; ++k) {
                    dot_product += std::popcount(a_row[k] & b_col[k]);
                }
                c_int->set(i, j, dot_product);
            }
        });
        return;
    }

    // 5. Triangular Matrix Support (Double)
    auto* a_tri = dynamic_cast<const TriangularMatrix<double>*>(&a);
    auto* b_tri = dynamic_cast<const TriangularMatrix<double>*>(&b);
    auto* c_tri = dynamic_cast<TriangularMatrix<double>*>(&result);

    if (a_tri && b_tri && c_tri) {
        uint64_t n = a.size();
        
        const char* a_base = reinterpret_cast<const char*>(a_tri->data());
        const char* b_base = reinterpret_cast<const char*>(b_tri->data());
        char* c_base = reinterpret_cast<char*>(c_tri->data());
        
        bool a_diag = a_tri->has_diagonal();
        bool b_diag = b_tri->has_diagonal();
        bool c_diag = c_tri->has_diagonal();

        ParallelFor(0, n, [&](size_t i) {
            std::vector<double> accumulator(n, 0.0);

            // Iterate over row i of A
            uint64_t k_start = a_diag ? i : i + 1;
            if (k_start >= n) return;
            
            uint64_t row_len_a = n - k_start;
            uint64_t a_offset = a_tri->get_row_offset(i);
            const double* a_row = reinterpret_cast<const double*>(a_base + a_offset);

            for (uint64_t k_idx = 0; k_idx < row_len_a; ++k_idx) {
                double val_a = a_row[k_idx];
                if (val_a == 0.0) continue;

                uint64_t k = k_start + k_idx;

                // Multiply by row k of B
                uint64_t j_start = b_diag ? k : k + 1;
                if (j_start >= n) continue;
                
                uint64_t row_len_b = n - j_start;
                uint64_t b_offset = b_tri->get_row_offset(k);
                const double* b_row = reinterpret_cast<const double*>(b_base + b_offset);
                
                for (uint64_t j_idx = 0; j_idx < row_len_b; ++j_idx) {
                    uint64_t j = j_start + j_idx;
                    accumulator[j] += val_a * b_row[j_idx];
                }
            }

            // Write accumulator to row i of C
            uint64_t c_start = c_diag ? i : i + 1;
            if (c_start >= n) return;
            
            uint64_t row_len_c = n - c_start;
            uint64_t c_offset = c_tri->get_row_offset(i);
            double* c_row = reinterpret_cast<double*>(c_base + c_offset);
            
            for (uint64_t j_idx = 0; j_idx < row_len_c; ++j_idx) {
                uint64_t j = c_start + j_idx;
                c_row[j_idx] = accumulator[j];
            }
        });
        
        c_tri->set_scalar(a_tri->get_scalar() * b_tri->get_scalar());
        return;
    }

    // 6. Diagonal Matrix Support
    // Case 6a: Diagonal * Diagonal -> Diagonal
    auto* a_diag = dynamic_cast<const DiagonalMatrix<double>*>(&a);
    auto* b_diag = dynamic_cast<const DiagonalMatrix<double>*>(&b);
    auto* c_diag = dynamic_cast<DiagonalMatrix<double>*>(&result);

    if (a_diag && b_diag && c_diag) {
        uint64_t n = a.size();
        const double* a_data = a_diag->data();
        const double* b_data = b_diag->data();
        double* c_data = c_diag->data();
        
        ParallelFor(0, n, [&](size_t i) {
            c_data[i] = a_data[i] * b_data[i];
        });
        
        c_diag->set_scalar(a_diag->get_scalar() * b_diag->get_scalar());
        return;
    }

    // Case 6b: Diagonal * Dense -> Dense
    auto* c_dense_dbl = dynamic_cast<DenseMatrix<double>*>(&result);
    // Reuse existing b_f64 if available, or cast again if needed (but b_f64 covers it)
    // But we need to be sure b is DenseMatrix<double>.
    // b_f64 is defined at top scope.
    
    if (a_diag && b_f64 && c_dense_dbl) {
        uint64_t n = a.size();
        const double* a_data = a_diag->data();
        const double* b_data = b_f64->data();
        double* c_data = c_dense_dbl->data();
        
        // Scale rows of B by A's diagonal
        ParallelFor(0, n, [&](size_t i) {
            double scale = a_data[i] * a_diag->get_scalar();
            const double* b_row = b_data + i * n;
            double* c_row = c_data + i * n;
            for (size_t j = 0; j < n; ++j) {
                c_row[j] = scale * b_row[j];
            }
        });
        c_dense_dbl->set_scalar(b_f64->get_scalar());
        return;
    }

    // Case 6c: Dense * Diagonal -> Dense
    if (a_f64 && b_diag && c_dense_dbl) {
        uint64_t n = a.size();
        const double* a_data = a_f64->data();
        const double* b_data = b_diag->data();
        double* c_data = c_dense_dbl->data();
        
        // Scale columns of A by B's diagonal
        ParallelFor(0, n, [&](size_t i) {
            const double* a_row = a_data + i * n;
            double* c_row = c_data + i * n;
            for (size_t j = 0; j < n; ++j) {
                c_row[j] = a_row[j] * b_data[j] * b_diag->get_scalar();
            }
        });
        c_dense_dbl->set_scalar(a_f64->get_scalar());
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

namespace {
    void invert_triangular(const TriangularMatrix<double>* a_tri, TriangularMatrix<double>* b_tri) {
        if (!a_tri->has_diagonal()) {
            throw std::invalid_argument("Cannot invert strictly triangular matrix (singular)");
        }
        
        uint64_t n = a_tri->size();
        
        ParallelFor(0, n, [&](size_t j) {
            double ajj = a_tri->get(j, j);
            if (std::abs(ajj) < 1e-15) {
                b_tri->set(j, j, 0.0);
            } else {
                b_tri->set(j, j, 1.0 / ajj);
            }
            
            for (int64_t i = (int64_t)j - 1; i >= 0; --i) {
                double sum = 0.0;
                for (size_t k = i + 1; k <= j; ++k) {
                    sum += a_tri->get(i, k) * b_tri->get(k, j);
                }
                
                double aii = a_tri->get(i, i);
                if (std::abs(aii) < 1e-15) {
                     b_tri->set(i, j, 0.0);
                } else {
                     b_tri->set(i, j, -sum / aii);
                }
            }
        });
        
        if (a_tri->get_scalar() != 1.0) {
            b_tri->set_scalar(1.0 / a_tri->get_scalar());
        }
    }
}

void CpuSolver::inverse(const MatrixBase& in, MatrixBase& out) {
    // Triangular
    if (auto* in_tri = dynamic_cast<const TriangularMatrix<double>*>(&in)) {
        if (auto* out_tri = dynamic_cast<TriangularMatrix<double>*>(&out)) {
            invert_triangular(in_tri, out_tri);
            return;
        }
    }

    // Double
    if (auto* in_dense = dynamic_cast<const DenseMatrix<double>*>(&in)) {
        if (auto* out_dense = dynamic_cast<DenseMatrix<double>*>(&out)) {
            inverse_impl(in_dense, out_dense);
            return;
        }
    }
    // Float
    if (auto* in_dense = dynamic_cast<const DenseMatrix<float>*>(&in)) {
        if (auto* out_dense = dynamic_cast<DenseMatrix<float>*>(&out)) {
            inverse_impl(in_dense, out_dense);
            return;
        }
    }
    
    // Diagonal
    if (auto* in_diag = dynamic_cast<const DiagonalMatrix<double>*>(&in)) {
        if (auto* out_diag = dynamic_cast<DiagonalMatrix<double>*>(&out)) {
            uint64_t n = in.size();
            const double* in_data = in_diag->data();
            double* out_data = out_diag->data();
            
            ParallelFor(0, n, [&](size_t i) {
                double val = in_data[i];
                if (std::abs(val) < 1e-12) {
                    // Singular
                    out_data[i] = 0.0; // Or throw?
                } else {
                    out_data[i] = 1.0 / val;
                }
            });
            
            out_diag->set_scalar(1.0 / in_diag->get_scalar());
            return;
        }
    }

    throw std::runtime_error("CpuSolver::inverse only supports DenseMatrix<double>, DenseMatrix<float>, or DiagonalMatrix<double>");
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
    void binary_op_impl(const MatrixBase& a, const MatrixBase& b, MatrixBase& result, Op op) {
        uint64_t n = a.size();
        
        // 1. DenseMatrix Result
        if (auto* res_dense = dynamic_cast<DenseMatrix<T>*>(&result)) {
            auto* a_dense = dynamic_cast<const DenseMatrix<T>*>(&a);
            auto* b_dense = dynamic_cast<const DenseMatrix<T>*>(&b);
            
            // Fast path: All dense, no transpose, scalar=1
            if (a_dense && b_dense && 
                a_dense->get_scalar() == 1.0 && !a_dense->is_transposed() &&
                b_dense->get_scalar() == 1.0 && !b_dense->is_transposed()) {
                
                const T* a_data = a_dense->data();
                const T* b_data = b_dense->data();
                T* res_data = res_dense->data();
                
                ParallelFor(0, n * n, [&](size_t i) {
                    res_data[i] = op(a_data[i], b_data[i]);
                });
                return;
            }
            
            // Generic Dense path
            ParallelFor(0, n, [&](size_t i) {
                for (size_t j = 0; j < n; ++j) {
                    T val_a = static_cast<T>(a.get_element_as_double(i, j));
                    T val_b = static_cast<T>(b.get_element_as_double(i, j));
                    res_dense->set(i, j, op(val_a, val_b));
                }
            });
            return;
        }

        // 2. TriangularMatrix Result
        if (auto* res_tri = dynamic_cast<TriangularMatrix<T>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                for (uint64_t j = i + 1; j < n; ++j) {
                    T val = op(static_cast<T>(a.get_element_as_double(i, j)), 
                               static_cast<T>(b.get_element_as_double(i, j)));
                    if (val != static_cast<T>(0)) {
                        res_tri->set(i, j, val);
                    }
                }
            });
            return;
        }

        // 3. DiagonalMatrix Result
        if (auto* res_diag = dynamic_cast<DiagonalMatrix<T>*>(&result)) {
            // Special case for IdentityMatrix (which is a DiagonalMatrix but immutable/special)
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
                res_diag->set(i, i, val);
            });
            return;
        }

        throw std::runtime_error("Unsupported result type in binary_op_impl");
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
    DataType dtype = result.get_data_type();
    if (dtype == DataType::FLOAT64) {
        binary_op_impl<double>(a, b, result, std::plus<>());
    } else if (dtype == DataType::FLOAT32) {
        binary_op_impl<float>(a, b, result, std::plus<>());
    } else if (dtype == DataType::INT32) {
        binary_op_impl<int32_t>(a, b, result, std::plus<>());
    } else {
        throw std::runtime_error("CpuSolver::add result data type not supported");
    }
}

void CpuSolver::subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    DataType dtype = result.get_data_type();
    if (dtype == DataType::FLOAT64) {
        binary_op_impl<double>(a, b, result, std::minus<>());
    } else if (dtype == DataType::FLOAT32) {
        binary_op_impl<float>(a, b, result, std::minus<>());
    } else if (dtype == DataType::INT32) {
        binary_op_impl<int32_t>(a, b, result, std::minus<>());
    } else {
        throw std::runtime_error("CpuSolver::subtract result data type not supported");
    }
}

void CpuSolver::elementwise_multiply(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    DataType dtype = result.get_data_type();
    if (dtype == DataType::FLOAT64) {
        binary_op_impl<double>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::FLOAT32) {
        binary_op_impl<float>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::INT32) {
        binary_op_impl<int32_t>(a, b, result, std::multiplies<>());
    } else {
        throw std::runtime_error("CpuSolver::elementwise_multiply result data type not supported");
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

double CpuSolver::dot(const VectorBase& a, const VectorBase& b) {
    uint64_t n = a.size();
    if (b.size() != n) throw std::invalid_argument("Vector dimensions mismatch");

    // 1. Dense Double * Dense Double
    auto* a_dbl = dynamic_cast<const DenseVector<double>*>(&a);
    auto* b_dbl = dynamic_cast<const DenseVector<double>*>(&b);

    if (a_dbl && b_dbl) {
        const double* a_data = a_dbl->data();
        const double* b_data = b_dbl->data();
        
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (int64_t i = 0; i < (int64_t)n; ++i) {
            sum += a_data[i] * b_data[i];
        }
        return sum;
    }
    
    // 2. BitVector * BitVector (Popcount)
    auto* a_bit = dynamic_cast<const DenseVector<bool>*>(&a);
    auto* b_bit = dynamic_cast<const DenseVector<bool>*>(&b);
    
    if (a_bit && b_bit) {
        const uint64_t* a_data = a_bit->data();
        const uint64_t* b_data = b_bit->data();
        uint64_t num_words = (n + 63) / 64;
        
        int64_t count = 0;
        #pragma omp parallel for reduction(+:count)
        for (int64_t i = 0; i < (int64_t)num_words; ++i) {
            count += std::popcount(a_data[i] & b_data[i]);
        }
        return static_cast<double>(count);
    }

    // Fallback
    double sum = 0.0;
    for (uint64_t i = 0; i < n; ++i) {
        sum += a.get_element_as_double(i) * b.get_element_as_double(i);
    }
    return sum;
}

void CpuSolver::matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, VectorBase& result) {
    if (m.size() != v.size()) throw std::invalid_argument("Dimension mismatch");
    uint64_t n = m.size();

    // Identity Optimization
    if (m.get_matrix_type() == MatrixType::IDENTITY) {
        double scalar = m.get_scalar();
        if (auto* res_int = dynamic_cast<DenseVector<int32_t>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_int->set(i, (int32_t)(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_dbl = dynamic_cast<DenseVector<double>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_dbl->set(i, v.get_element_as_double(i) * scalar);
            });
        }
        return;
    }

    // BitMatrix * BitVector Optimization
    auto* m_bit = dynamic_cast<const DenseMatrix<bool>*>(&m);
    auto* v_bit = dynamic_cast<const DenseVector<bool>*>(&v);
    auto* res_int = dynamic_cast<DenseVector<int32_t>*>(&result);
    
    if (m_bit && v_bit && res_int) {
        const uint64_t* m_data = m_bit->data();
        const uint64_t* v_data = v_bit->data();
        int32_t* res_data = res_int->data();
        
        uint64_t stride_bytes = m_bit->stride_bytes();
        uint64_t words_per_row = stride_bytes / 8;
        
        ParallelFor(0, n, [&](size_t i) {
            const uint64_t* row_ptr = m_data + i * words_per_row;
            int32_t sum = 0;
            for (size_t k = 0; k < words_per_row; ++k) {
                sum += std::popcount(row_ptr[k] & v_data[k]);
            }
            res_data[i] = sum;
        });
        return;
    }

    // Generic Fallback
    if (auto* res_int = dynamic_cast<DenseVector<int32_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            int32_t sum = 0;
            for (uint64_t j = 0; j < n; ++j) {
                sum += (int32_t)(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_int->set(i, sum);
        });
    } else if (auto* res_dbl = dynamic_cast<DenseVector<double>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            double sum = 0.0;
            for (uint64_t j = 0; j < n; ++j) {
                sum += m.get_element_as_double(i, j) * v.get_element_as_double(j);
            }
            res_dbl->set(i, sum);
        });
    } else {
        throw std::runtime_error("Unsupported result type for matrix_vector_multiply");
    }
}

void CpuSolver::vector_matrix_multiply(const VectorBase& v, const MatrixBase& m, VectorBase& result) {
    if (m.size() != v.size()) throw std::invalid_argument("Dimension mismatch");
    uint64_t n = m.size();

    // Identity Optimization
    if (m.get_matrix_type() == MatrixType::IDENTITY) {
        double scalar = m.get_scalar();
        if (auto* res_int = dynamic_cast<DenseVector<int32_t>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_int->set(i, (int32_t)(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_dbl = dynamic_cast<DenseVector<double>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_dbl->set(i, v.get_element_as_double(i) * scalar);
            });
        }
        result.set_transposed(true);
        return;
    }

    // Generic Fallback
    if (auto* res_int = dynamic_cast<DenseVector<int32_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t j) {
            int32_t sum = 0;
            for (uint64_t i = 0; i < n; ++i) {
                sum += (int32_t)(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_int->set(j, sum);
        });
    } else if (auto* res_dbl = dynamic_cast<DenseVector<double>*>(&result)) {
        ParallelFor(0, n, [&](size_t j) {
            double sum = 0.0;
            for (uint64_t i = 0; i < n; ++i) {
                sum += v.get_element_as_double(i) * m.get_element_as_double(i, j);
            }
            res_dbl->set(j, sum);
        });
    } else {
        throw std::runtime_error("Unsupported result type for vector_matrix_multiply");
    }
}

void CpuSolver::outer_product(const VectorBase& a, const VectorBase& b, MatrixBase& result) {
    if (a.size() != b.size()) throw std::invalid_argument("Vectors must be same size");
    uint64_t n = a.size();

    if (auto* m = dynamic_cast<DenseMatrix<bool>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            double val_a = a.get_element_as_double(i);
            if (val_a != 0.0) { 
                for (uint64_t j = 0; j < n; ++j) {
                    double val_b = b.get_element_as_double(j);
                    if (val_b != 0.0) {
                        m->set(i, j, true);
                    }
                }
            }
        });
    } else if (auto* m = dynamic_cast<DenseMatrix<int32_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            int32_t val_a = (int32_t)a.get_element_as_double(i);
            for (uint64_t j = 0; j < n; ++j) {
                m->set(i, j, val_a * (int32_t)b.get_element_as_double(j));
            }
        });
    } else if (auto* m = dynamic_cast<DenseMatrix<double>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            double val_a = a.get_element_as_double(i);
            for (uint64_t j = 0; j < n; ++j) {
                m->set(i, j, val_a * b.get_element_as_double(j));
            }
        });
    } else {
        throw std::runtime_error("Unsupported result type for outer_product");
    }
}

void CpuSolver::add_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) {
    uint64_t n = a.size();
    
    auto* a_dbl = dynamic_cast<const DenseVector<double>*>(&a);
    auto* b_dbl = dynamic_cast<const DenseVector<double>*>(&b);
    auto* res_dbl = dynamic_cast<DenseVector<double>*>(&result);
    
    if (a_dbl && b_dbl && res_dbl) {
        const double* a_data = a_dbl->data();
        const double* b_data = b_dbl->data();
        double* res_data = res_dbl->data();
        
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = a_data[i] + b_data[i];
        });
        return;
    }
    
    // Fallback
    ParallelFor(0, n, [&](size_t i) {
        if (res_dbl) {
            res_dbl->set(i, a.get_element_as_double(i) + b.get_element_as_double(i));
        } else {
             // Try int32
             auto* res_int = dynamic_cast<DenseVector<int32_t>*>(&result);
             if (res_int) {
                 res_int->set(i, (int32_t)(a.get_element_as_double(i) + b.get_element_as_double(i)));
             }
        }
    });
}

void CpuSolver::subtract_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) {
    uint64_t n = a.size();
    
    auto* a_dbl = dynamic_cast<const DenseVector<double>*>(&a);
    auto* b_dbl = dynamic_cast<const DenseVector<double>*>(&b);
    auto* res_dbl = dynamic_cast<DenseVector<double>*>(&result);
    
    if (a_dbl && b_dbl && res_dbl) {
        const double* a_data = a_dbl->data();
        const double* b_data = b_dbl->data();
        double* res_data = res_dbl->data();
        
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = a_data[i] - b_data[i];
        });
        return;
    }
    
    // Fallback
    ParallelFor(0, n, [&](size_t i) {
        if (res_dbl) {
            res_dbl->set(i, a.get_element_as_double(i) - b.get_element_as_double(i));
        }
    });
}

void CpuSolver::scalar_multiply_vector(const VectorBase& a, double scalar, VectorBase& result) {
    uint64_t n = a.size();
    
    auto* a_dbl = dynamic_cast<const DenseVector<double>*>(&a);
    auto* res_dbl = dynamic_cast<DenseVector<double>*>(&result);
    
    if (a_dbl && res_dbl) {
        const double* a_data = a_dbl->data();
        double* res_data = res_dbl->data();
        
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = a_data[i] * scalar;
        });
        return;
    }
    
    // Fallback
    ParallelFor(0, n, [&](size_t i) {
        if (res_dbl) {
            res_dbl->set(i, a.get_element_as_double(i) * scalar);
        }
    });
}

void CpuSolver::scalar_add_vector(const VectorBase& a, double scalar, VectorBase& result) {
    uint64_t n = a.size();
    
    auto* a_dbl = dynamic_cast<const DenseVector<double>*>(&a);
    auto* res_dbl = dynamic_cast<DenseVector<double>*>(&result);
    
    if (a_dbl && res_dbl) {
        const double* a_data = a_dbl->data();
        double* res_data = res_dbl->data();
        
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = a_data[i] + scalar;
        });
        return;
    }
    
    // Fallback
    ParallelFor(0, n, [&](size_t i) {
        if (res_dbl) {
            res_dbl->set(i, a.get_element_as_double(i) + scalar);
        }
    });
}

} // namespace pycauset

