#include "pycauset/compute/cpu/CpuSolver.hpp"
#include <complex>
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
// Force 64-bit integers for LAPACK to match the DLL expectation (likely ILP64)
// or at least to prevent buffer overflow if the DLL writes 64-bit integers.
// #define LAPACK_ILP64
#include <cblas.h>
#include <lapacke.h>
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/core/Float16.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/ComplexFloat16Matrix.hpp"
#include "pycauset/vector/DenseVector.hpp"
#include "pycauset/vector/ComplexFloat16Vector.hpp"
#include "pycauset/vector/UnitVector.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/matrix/SymmetricMatrix.hpp"
#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/matrix/IdentityMatrix.hpp"
#include "pycauset/math/Eigen.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/core/MemoryHints.hpp"
#include "pycauset/core/MemoryGovernor.hpp"
#include "pycauset/core/DebugTrace.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <bit>
#include <filesystem>
#include <vector>
#include <cstring>
#include <iostream>
#include <limits>

namespace pycauset {

namespace {
    inline int64_t checked_add_int64(int64_t a, int64_t b) {
        if ((b > 0 && a > (std::numeric_limits<int64_t>::max)() - b) ||
            (b < 0 && a < (std::numeric_limits<int64_t>::min)() - b)) {
            throw std::overflow_error("Integer matmul overflow: accumulator overflow");
        }
        return a + b;
    }

    template <typename T>
    constexpr bool is_std_complex_v = false;

    template <>
    constexpr bool is_std_complex_v<std::complex<float>> = true;

    template <>
    constexpr bool is_std_complex_v<std::complex<double>> = true;

    template <typename Op>
    void binary_op_complex16_impl(const MatrixBase& a, const MatrixBase& b, ComplexFloat16Matrix& out, Op op) {
        const uint64_t n = out.size();
        if (a.size() != n || b.size() != n) {
            throw std::invalid_argument("Dimension mismatch");
        }

        const bool t_c = out.is_transposed();
        float16_t* rdst = out.real_data();
        float16_t* idst = out.imag_data();

        ParallelFor(0, n, [&](size_t i) {
            for (size_t j = 0; j < n; ++j) {
                const std::complex<double> va = a.get_element_as_complex(i, j);
                const std::complex<double> vb = b.get_element_as_complex(i, j);
                const std::complex<double> vc = op(va, vb);
                const uint64_t idx = t_c ? (j * n + i) : (i * n + j);
                rdst[idx] = float16_t(vc.real());
                idst[idx] = float16_t(vc.imag());
            }
        });

        out.set_scalar(1.0);
    }

    // Forward declaration
    template <typename T>
    void matmul_streaming(const DenseMatrix<T>* a_dense, const DenseMatrix<T>* b_dense, DenseMatrix<T>* c_dense);

    // --- Helper: Direct Path Optimization ---
    // Tries to pin all matrices and run BLAS directly.
    // Returns true if successful, false if memory budget didn't allow it.
    template <typename T>
    bool attempt_direct_path(const DenseMatrix<T>* a_dense, const DenseMatrix<T>* b_dense, DenseMatrix<T>* c_dense) {
        // Only supported for Float/Double where we have BLAS
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
            uint64_t n = a_dense->size();
            size_t matrix_bytes = n * n * sizeof(T);
            size_t total_bytes = matrix_bytes * 3; // A + B + C
            
            auto& governor = pycauset::core::MemoryGovernor::instance();
            
            // Check 1: Does it fit in RAM?
            // Check 2: Do we have enough Pinned Memory Budget?
            if (governor.can_fit_in_ram(total_bytes) && governor.try_pin_memory(total_bytes)) {
                // Attempt to pin all three matrices
                bool pinned = true;
                // We use const_cast because pinning is a logical const operation (doesn't change data)
                // but might change internal OS handles.
                pinned &= const_cast<DenseMatrix<T>*>(a_dense)->pin_range(0, n * n);
                pinned &= const_cast<DenseMatrix<T>*>(b_dense)->pin_range(0, n * n);
                pinned &= c_dense->pin_range(0, n * n);
                
                if (pinned) {
                    const T* a_data = a_dense->data();
                    const T* b_data = b_dense->data();
                    T* c_data = c_dense->data();
                    
                    bool t_a = a_dense->is_transposed();
                    bool t_b = b_dense->is_transposed();
                    
                    if constexpr (std::is_same_v<T, double>) {
                        cblas_dgemm(
                            CblasRowMajor,
                            t_a ? CblasTrans : CblasNoTrans,
                            t_b ? CblasTrans : CblasNoTrans,
                            n, n, n,
                            1.0, a_data, n,
                            b_data, n,
                            0.0, c_data, n
                        );
                    } else {
                        cblas_sgemm(
                            CblasRowMajor,
                            t_a ? CblasTrans : CblasNoTrans,
                            t_b ? CblasTrans : CblasNoTrans,
                            n, n, n,
                            1.0f, a_data, n,
                            b_data, n,
                            0.0f, c_data, n
                        );
                    }
                    
                    // Cleanup
                    // Unpinning is important to release the "locked" status
                    const_cast<DenseMatrix<T>*>(a_dense)->unpin_range(0, n * n);
                    const_cast<DenseMatrix<T>*>(b_dense)->unpin_range(0, n * n);
                    c_dense->unpin_range(0, n * n);
                    
                    governor.unpin_memory(total_bytes);
                    
                    c_dense->set_scalar(a_dense->get_scalar() * b_dense->get_scalar());
                    return true;
                } else {
                    // Pinning failed (OS limit?), release budget and fall back
                    governor.unpin_memory(total_bytes);
                }
            }
        }
        return false;
    }

    template <typename T>
    void matmul_impl(const DenseMatrix<T>* a_dense, const DenseMatrix<T>* b_dense, DenseMatrix<T>* c_dense) {
        uint64_t n = a_dense->size();
        if (b_dense->size() != n || c_dense->size() != n) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        // Try Direct Path first (Generalized for Float/Double)
        if (attempt_direct_path(a_dense, b_dense, c_dense)) {
            return;
        }

        const T* a_data = a_dense->data();
        const T* b_data = b_dense->data();
        T* c_data = c_dense->data();

        bool t_a = a_dense->is_transposed();
        bool t_b = b_dense->is_transposed();

        // --- Integer path: int64 accumulator + overflow checks ---
        // Policy: never silently wrap, never silently widen output storage.
        // We accumulate in int64 to avoid intermediate overflow, then range-check into T.
        if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
            size_t block_size = 64;

            ParallelBlockMap(n, n, block_size, [&](size_t i_start, size_t i_end, size_t j_start, size_t j_end) {
                std::vector<int64_t> sums(j_end - j_start);

                for (size_t i = i_start; i < i_end; ++i) {
                    if (!t_a && !t_b) {
                        std::fill(sums.begin(), sums.end(), 0);
                        for (size_t k_start = 0; k_start < n; k_start += block_size) {
                            size_t k_end = std::min(k_start + block_size, (size_t)n);
                            for (size_t k = k_start; k < k_end; ++k) {
                                int64_t val_a = static_cast<int64_t>(a_data[i * n + k]);
                                if (val_a == 0) continue;
                                const T* b_ptr = b_data + k * n + j_start;
                                for (size_t jj = 0; jj < sums.size(); ++jj) {
                                    int64_t term = val_a * static_cast<int64_t>(b_ptr[jj]);
                                    sums[jj] = checked_add_int64(sums[jj], term);
                                }
                            }
                        }
                    } else if (!t_a && t_b) {
                        const T* a_ptr = a_data + i * n;
                        for (size_t jj = 0; jj < sums.size(); ++jj) {
                            int64_t sum = 0;
                            const T* b_row = b_data + (j_start + jj) * n;
                            for (size_t k_start = 0; k_start < n; k_start += block_size) {
                                size_t k_end = std::min(k_start + block_size, (size_t)n);
                                for (size_t k = k_start; k < k_end; ++k) {
                                    int64_t term = static_cast<int64_t>(a_ptr[k]) * static_cast<int64_t>(b_row[k]);
                                    sum = checked_add_int64(sum, term);
                                }
                            }
                            sums[jj] = sum;
                        }
                    } else if (t_a && !t_b) {
                        std::fill(sums.begin(), sums.end(), 0);
                        for (size_t k_start = 0; k_start < n; k_start += block_size) {
                            size_t k_end = std::min(k_start + block_size, (size_t)n);
                            for (size_t k = k_start; k < k_end; ++k) {
                                int64_t val_a = static_cast<int64_t>(a_data[k * n + i]);
                                if (val_a == 0) continue;
                                const T* b_ptr = b_data + k * n + j_start;
                                for (size_t jj = 0; jj < sums.size(); ++jj) {
                                    int64_t term = val_a * static_cast<int64_t>(b_ptr[jj]);
                                    sums[jj] = checked_add_int64(sums[jj], term);
                                }
                            }
                        }
                    } else {
                        for (size_t jj = 0; jj < sums.size(); ++jj) {
                            int64_t sum = 0;
                            const T* b_row = b_data + (j_start + jj) * n;
                            for (size_t k_start = 0; k_start < n; k_start += block_size) {
                                size_t k_end = std::min(k_start + block_size, (size_t)n);
                                for (size_t k = k_start; k < k_end; ++k) {
                                    int64_t term = static_cast<int64_t>(a_data[k * n + i]) * static_cast<int64_t>(b_row[k]);
                                    sum = checked_add_int64(sum, term);
                                }
                            }
                            sums[jj] = sum;
                        }
                    }

                    T* c_ptr = c_data + i * n;
                    for (size_t jj = 0; jj < sums.size(); ++jj) {
                        int64_t s = sums[jj];
                        if constexpr (std::is_signed_v<T>) {
                            const int64_t lo = static_cast<int64_t>((std::numeric_limits<T>::min)());
                            const int64_t hi = static_cast<int64_t>((std::numeric_limits<T>::max)());
                            if (s < lo || s > hi) {
                                throw std::overflow_error("Integer matmul overflow: result does not fit in output dtype");
                            }
                        } else {
                            const uint64_t hi = static_cast<uint64_t>((std::numeric_limits<T>::max)());
                            if (s < 0 || static_cast<uint64_t>(s) > hi) {
                                throw std::overflow_error("Integer matmul overflow: result does not fit in output dtype");
                            }
                        }
                        c_ptr[j_start + jj] = static_cast<T>(s);
                    }
                }
            });

            c_dense->set_scalar(a_dense->get_scalar() * b_dense->get_scalar());
            return;
        }

        // --- Optimization: Use OpenBLAS for double/float (Fallback if Direct Path failed) ---
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
            size_t total_bytes = 3 * n * n * sizeof(T);
            
            // Check if we should use Direct Path (OS Paging) or Streaming
            if (false && pycauset::core::MemoryGovernor::instance().should_use_direct_path(total_bytes)) {
                std::cout << "DEBUG: Using Direct Path (BLAS)" << std::endl;
                if constexpr (std::is_same_v<T, double>) {
                    cblas_dgemm(
                        CblasRowMajor,
                        t_a ? CblasTrans : CblasNoTrans,
                        t_b ? CblasTrans : CblasNoTrans,
                        n, n, n,
                        1.0, a_data, n,
                        b_data, n,
                        0.0, c_data, n
                    );
                } else {
                    cblas_sgemm(
                        CblasRowMajor,
                        t_a ? CblasTrans : CblasNoTrans,
                        t_b ? CblasTrans : CblasNoTrans,
                        n, n, n,
                        1.0f, a_data, n,
                        b_data, n,
                        0.0f, c_data, n
                    );
                }
                c_dense->set_scalar(a_dense->get_scalar() * b_dense->get_scalar());
                return;
            } else {
                // Fallback to Streaming (Out-of-Core)
                matmul_streaming(a_dense, b_dense, c_dense);
                return;
            }
        }
        
        // Initialize result to 0
        std::fill(c_data, c_data + n * n, static_cast<T>(0));
        
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

    // Specializations for Double/Float are now handled by the template above
    // which includes the Direct Path optimization.

    // --- Helper: Direct Path Inverse (LAPACK) ---
    template <typename T>
    void inverse_direct(const DenseMatrix<T>* in_dense, DenseMatrix<T>* out_dense) {
        uint64_t n = in_dense->size();
        
        // 1. Copy input to output (LAPACK inverts in-place)
        const T* src = in_dense->data();
        T* dst = out_dense->data();
        
        bool is_transposed = in_dense->is_transposed();
        
        if (is_transposed) {
            ParallelFor(0, n, [&](size_t i) {
                for (size_t j = 0; j < n; ++j) {
                    dst[i * n + j] = src[j * n + i];
                }
            });
        } else {
            // Direct copy
            std::copy(src, src + n * n, dst);
        }
        
        // 2. LU Factorization (dgetrf)
        std::vector<lapack_int> ipiv(n);
        lapack_int info = 0;
        
        if constexpr (std::is_same_v<T, double>) {
            info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, (lapack_int)n, (lapack_int)n, dst, (lapack_int)n, ipiv.data());
        } else {
            info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, (lapack_int)n, (lapack_int)n, dst, (lapack_int)n, ipiv.data());
        }
        
        if (info > 0) {
            throw std::runtime_error("Matrix is singular");
        } else if (info < 0) {
            throw std::runtime_error("Illegal argument in LAPACK dgetrf");
        }
        
        // 3. Inverse (dgetri)
        if constexpr (std::is_same_v<T, double>) {
            info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, (lapack_int)n, dst, (lapack_int)n, ipiv.data());
        } else {
            info = LAPACKE_sgetri(LAPACK_ROW_MAJOR, (lapack_int)n, dst, (lapack_int)n, ipiv.data());
        }
        
        if (info > 0) {
            throw std::runtime_error("Matrix is singular");
        } else if (info < 0) {
            throw std::runtime_error("Illegal argument in LAPACK dgetri");
        }
        
        // Handle scalar
        std::complex<double> s = in_dense->get_scalar();
        if (s != 1.0) {
            out_dense->set_scalar(1.0 / s);
        }
    }



    template <typename T>
    void inverse_impl(const DenseMatrix<T>* in_dense, DenseMatrix<T>* out_dense) {
        uint64_t n = in_dense->size();
        if (out_dense->size() != n) throw std::invalid_argument("Output matrix size mismatch");
        if (in_dense->get_scalar() == 0.0) throw std::runtime_error("Matrix scalar is 0, cannot invert");

        // --- Optimization: Direct Path (LAPACK) ---
        // If it fits in RAM, use LAPACK. It's orders of magnitude faster than manual Gauss-Jordan.
        // We need space for Input (already there) + Output (already there) + Pivot Array (negligible).
        // So we just check if Output fits in RAM (Input is likely already in RAM or mapped).
        // Actually, we need to ensure we don't cause thrashing.
        // Total footprint ~ 2 * N^2 * sizeof(T).
        size_t total_bytes = 2 * n * n * sizeof(T);
        
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
            // if (pycauset::core::MemoryGovernor::instance().should_use_direct_path(total_bytes)) {
                inverse_direct(in_dense, out_dense);
                return;
            // }
        }

        // --- Fallback: Out-of-Core Block Gauss-Jordan ---
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
        // Calculate optimal block size based on available RAM
        // We need to hold 2 * block_size * n * sizeof(T) in RAM (Panel for W and R)
        // Plus some overhead.
        size_t available_ram = pycauset::core::MemoryGovernor::instance().get_available_system_ram();
        size_t row_size = n * sizeof(T);
        // Use 50% of available RAM for the panel to be safe and leave room for OS cache/streaming
        size_t panel_budget = available_ram / 2;
        size_t max_block_size = panel_budget / (2 * row_size);
        
        // Clamp block size
        size_t block_size = std::clamp(max_block_size, (size_t)64, (size_t)16384);
        // Align to 64
        block_size = (block_size / 64) * 64;
        if (block_size == 0) block_size = 64;

        std::cout << "PyCauset: Out-of-Core Inverse using Block Size: " << block_size 
                  << " (Panel Size: " << (2 * block_size * row_size / 1024 / 1024) << " MB)" << std::endl;

        T epsilon = std::is_same_v<T, float> ? 1e-5f : 1e-12;
        T zero_threshold = std::is_same_v<T, float> ? 1e-7f : 1e-15;
        
        for (size_t k_start = 0; k_start < n; k_start += block_size) {
            size_t k_end = std::min(k_start + block_size, (size_t)n);
            size_t current_block_size = k_end - k_start;

            // Pin the current panel (rows k_start to k_end) for both W and R
            // This ensures the pivot rows stay in RAM while we stream the rest of the matrix against them.
            // Note: pin_range takes offset in elements, not bytes? No, usually bytes or elements.
            // DenseMatrix::pin_range takes (offset, length) in elements.
            // Rows are contiguous.
            work.pin_range(k_start * n, current_block_size * n);
            out_dense->pin_range(k_start * n, current_block_size * n);

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

            // Unpin the panel
            work.unpin_range(k_start * n, current_block_size * n);
            out_dense->unpin_range(k_start * n, current_block_size * n);
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

    inline void blas_gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                          const int M, const int N, const int K,
                          const double alpha, const double *A, const int lda,
                          const double *B, const int ldb, const double beta,
                          double *C, const int ldc) {
        cblas_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    inline void blas_gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                          const int M, const int N, const int K,
                          const float alpha, const float *A, const int lda,
                          const float *B, const int ldb, const float beta,
                          float *C, const int ldc) {
        cblas_sgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    template <typename T>
    void matmul_streaming(const DenseMatrix<T>* a_dense, const DenseMatrix<T>* b_dense, DenseMatrix<T>* c_dense) {
        size_t n = a_dense->rows();
        auto& governor = pycauset::core::MemoryGovernor::instance();

        // DYNAMIC BLOCK SIZING
        // Target 80% of available RAM
        size_t available_ram = governor.get_available_system_ram();
        size_t target_ram = static_cast<size_t>(available_ram * 0.8);
        
        // We need 3 blocks: A, B, C. 
        // 3 * (B^2) * sizeof(T) = RAM
        size_t optimal_block_size = static_cast<size_t>(std::sqrt(target_ram / (3.0 * sizeof(T))));
        
        // Clamp to reasonable limits
        // Min: 4096 elements (small)
        // Max: n (full matrix)
        size_t block_size = std::max(static_cast<size_t>(4096), optimal_block_size);
        if (block_size > n) block_size = n;
        
        // Ensure even multiple of 8 for SIMD alignment (optional but good)
        block_size = (block_size / 8) * 8;

        std::cout << "[PyCauset] Out-of-Core MatMul: N=" << n 
                  << ", BlockSize=" << block_size 
                  << " (" << (block_size*block_size*sizeof(T))/(1024.0*1024.0) << " MB/block)" << std::endl;

        // Ensure OpenBLAS threading
        // int num_threads = omp_get_max_threads();
        // openblas_set_num_threads(num_threads);

        const T* a_data = a_dense->data();
        const T* b_data = b_dense->data();
        T* c_data = c_dense->data();

        // Allocate buffers
        std::vector<T> a_block(block_size * block_size);
        std::vector<T> b_block(block_size * block_size);
        std::vector<T> c_block(block_size * block_size);

        // Tiled Matrix Multiplication
        // Loop Order: i, j, k (Standard)
        for (size_t i = 0; i < n; i += block_size) {
            size_t ib = std::min(block_size, n - i);
            
            for (size_t j = 0; j < n; j += block_size) {
                size_t jb = std::min(block_size, n - j);
                
                // Reset Accumulator
                std::fill(c_block.begin(), c_block.end(), static_cast<T>(0));

                for (size_t k = 0; k < n; k += block_size) {
                    size_t kb = std::min(block_size, n - k);

                    // Load A tile
                    for (size_t row = 0; row < ib; ++row) {
                        std::memcpy(&a_block[row * kb], &a_data[(i + row) * n + k], kb * sizeof(T));
                    }

                    // Load B tile
                    for (size_t row = 0; row < kb; ++row) {
                        std::memcpy(&b_block[row * jb], &b_data[(k + row) * n + j], jb * sizeof(T));
                    }

                    // Compute
                    T alpha = 1.0;
                    T beta = 1.0; // Accumulate
                    if constexpr (std::is_same_v<T, double>) {
                        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                                    ib, jb, kb, alpha, a_block.data(), kb, b_block.data(), jb, beta, c_block.data(), jb);
                    } else {
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                                    ib, jb, kb, alpha, a_block.data(), kb, b_block.data(), jb, beta, c_block.data(), jb);
                    }
                }

                // Write C tile
                for (size_t row = 0; row < ib; ++row) {
                    std::memcpy(&c_data[(i + row) * n + j], &c_block[row * jb], jb * sizeof(T));
                }
            }
        }
        
        c_dense->set_scalar(a_dense->get_scalar() * b_dense->get_scalar());
    }
} // namespace

void CpuSolver::matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    // Dispatch to specialized implementations
    
    // 1. Double Precision (Float64)
    auto* a_f64 = dynamic_cast<const DenseMatrix<double>*>(&a);
    auto* b_f64 = dynamic_cast<const DenseMatrix<double>*>(&b);
    auto* c_f64 = dynamic_cast<DenseMatrix<double>*>(&result);

    if (a_f64 && b_f64 && c_f64) {
        debug_trace::set_last("cpu.matmul.f64");
        matmul_impl(a_f64, b_f64, c_f64);
        return;
    }

    // 2. Single Precision (Float32)
    auto* a_f32 = dynamic_cast<const DenseMatrix<float>*>(&a);
    auto* b_f32 = dynamic_cast<const DenseMatrix<float>*>(&b);
    auto* c_f32 = dynamic_cast<DenseMatrix<float>*>(&result);

    if (a_f32 && b_f32 && c_f32) {
        debug_trace::set_last("cpu.matmul.f32");
        matmul_impl(a_f32, b_f32, c_f32);
        return;
    }

    // 2c. Complex32
    auto* a_c32 = dynamic_cast<const DenseMatrix<std::complex<float>>*>(&a);
    auto* b_c32 = dynamic_cast<const DenseMatrix<std::complex<float>>*>(&b);
    auto* c_c32 = dynamic_cast<DenseMatrix<std::complex<float>>*>(&result);

    if (a_c32 && b_c32 && c_c32) {
        debug_trace::set_last("cpu.matmul.c32");
        matmul_impl(a_c32, b_c32, c_c32);
        return;
    }

    // 2d. Complex64
    auto* a_c64 = dynamic_cast<const DenseMatrix<std::complex<double>>*>(&a);
    auto* b_c64 = dynamic_cast<const DenseMatrix<std::complex<double>>*>(&b);
    auto* c_c64 = dynamic_cast<DenseMatrix<std::complex<double>>*>(&result);

    if (a_c64 && b_c64 && c_c64) {
        debug_trace::set_last("cpu.matmul.c64");
        matmul_impl(a_c64, b_c64, c_c64);
        return;
    }

    // 2a. Half Precision (Float16) -> Float16 (compute in float32, store float16)
    auto* c_f16 = dynamic_cast<DenseMatrix<float16_t>*>(&result);
    if (c_f16 &&
        (a.get_data_type() == DataType::FLOAT16 || a.get_data_type() == DataType::FLOAT32 ||
         a.get_data_type() == DataType::FLOAT64) &&
        (b.get_data_type() == DataType::FLOAT16 || b.get_data_type() == DataType::FLOAT32 ||
         b.get_data_type() == DataType::FLOAT64)) {
        debug_trace::set_last("cpu.matmul.float16_fallback");
        const uint64_t n = a.size();
        if (b.size() != n || result.size() != n) {
            throw std::invalid_argument("Dimension mismatch");
        }

        const bool t_c = c_f16->is_transposed();
        float16_t* c_data = c_f16->data();
        ParallelFor(0, n, [&](size_t i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < n; ++k) {
                    sum += static_cast<float>(a.get_element_as_double(i, k)) *
                           static_cast<float>(b.get_element_as_double(k, j));
                }
                const float16_t out(sum);
                if (t_c) {
                    c_data[j * n + i] = out;
                } else {
                    c_data[i * n + j] = out;
                }
            }
        });
        c_f16->set_scalar(1.0);
        return;
    }

    // 2b. Mixed Float32/Float64 -> Float32 (generic fallback)
    if (c_f32 &&
        (a.get_data_type() == DataType::FLOAT32 || a.get_data_type() == DataType::FLOAT64) &&
        (b.get_data_type() == DataType::FLOAT32 || b.get_data_type() == DataType::FLOAT64)) {
        debug_trace::set_last("cpu.matmul.mixed_float_fallback_f32");
        uint64_t n = a.size();
        if (b.size() != n || result.size() != n) {
            throw std::invalid_argument("Dimension mismatch");
        }

        float* c_data = c_f32->data();
        ParallelFor(0, n, [&](size_t i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < n; ++k) {
                    sum += static_cast<float>(a.get_element_as_double(i, k)) *
                           static_cast<float>(b.get_element_as_double(k, j));
                }
                c_data[i * n + j] = sum;
            }
        });
        return;
    }

    // 2e. ComplexFloat16 result (compute in complex<double>, store float16 planes)
    if (auto* c_cf16 = dynamic_cast<ComplexFloat16Matrix*>(&result)) {
        debug_trace::set_last("cpu.matmul.cf16_fallback");
        const uint64_t n = a.size();
        if (b.size() != n || result.size() != n) {
            throw std::invalid_argument("Dimension mismatch");
        }

        const bool t_c = c_cf16->is_transposed();
        float16_t* rdst = c_cf16->real_data();
        float16_t* idst = c_cf16->imag_data();

        ParallelFor(0, n, [&](size_t i) {
            for (size_t j = 0; j < n; ++j) {
                std::complex<double> sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += a.get_element_as_complex(i, k) * b.get_element_as_complex(k, j);
                }
                const uint64_t idx = t_c ? (j * n + i) : (i * n + j);
                rdst[idx] = float16_t(sum.real());
                idst[idx] = float16_t(sum.imag());
            }
        });

        c_cf16->set_scalar(1.0);
        return;
    }

    // 3. Integer (Int32)
    auto* a_i32 = dynamic_cast<const DenseMatrix<int32_t>*>(&a);
    auto* b_i32 = dynamic_cast<const DenseMatrix<int32_t>*>(&b);
    auto* c_i32 = dynamic_cast<DenseMatrix<int32_t>*>(&result);

    if (a_i32 && b_i32 && c_i32) {
        debug_trace::set_last("cpu.matmul.i32");
        matmul_impl(a_i32, b_i32, c_i32);
        return;
    }

    // 3b. Integer (Int16)
    auto* a_i16 = dynamic_cast<const DenseMatrix<int16_t>*>(&a);
    auto* b_i16 = dynamic_cast<const DenseMatrix<int16_t>*>(&b);
    auto* c_i16 = dynamic_cast<DenseMatrix<int16_t>*>(&result);

    if (a_i16 && b_i16 && c_i16) {
        debug_trace::set_last("cpu.matmul.i16");
        matmul_impl(a_i16, b_i16, c_i16);
        return;
    }

    // 3d. Integer (Int8)
    auto* a_i8 = dynamic_cast<const DenseMatrix<int8_t>*>(&a);
    auto* b_i8 = dynamic_cast<const DenseMatrix<int8_t>*>(&b);
    auto* c_i8 = dynamic_cast<DenseMatrix<int8_t>*>(&result);

    if (a_i8 && b_i8 && c_i8) {
        debug_trace::set_last("cpu.matmul.i8");
        matmul_impl(a_i8, b_i8, c_i8);
        return;
    }

    // 3e. Integer (Int64)
    auto* a_i64 = dynamic_cast<const DenseMatrix<int64_t>*>(&a);
    auto* b_i64 = dynamic_cast<const DenseMatrix<int64_t>*>(&b);
    auto* c_i64 = dynamic_cast<DenseMatrix<int64_t>*>(&result);

    if (a_i64 && b_i64 && c_i64) {
        debug_trace::set_last("cpu.matmul.i64");
        matmul_impl(a_i64, b_i64, c_i64);
        return;
    }

    // 3f. Unsigned Integer (UInt8)
    auto* a_u8 = dynamic_cast<const DenseMatrix<uint8_t>*>(&a);
    auto* b_u8 = dynamic_cast<const DenseMatrix<uint8_t>*>(&b);
    auto* c_u8 = dynamic_cast<DenseMatrix<uint8_t>*>(&result);

    if (a_u8 && b_u8 && c_u8) {
        debug_trace::set_last("cpu.matmul.u8");
        matmul_impl(a_u8, b_u8, c_u8);
        return;
    }

    // 3g. Unsigned Integer (UInt16)
    auto* a_u16 = dynamic_cast<const DenseMatrix<uint16_t>*>(&a);
    auto* b_u16 = dynamic_cast<const DenseMatrix<uint16_t>*>(&b);
    auto* c_u16 = dynamic_cast<DenseMatrix<uint16_t>*>(&result);

    if (a_u16 && b_u16 && c_u16) {
        debug_trace::set_last("cpu.matmul.u16");
        matmul_impl(a_u16, b_u16, c_u16);
        return;
    }

    // 3h. Unsigned Integer (UInt32)
    auto* a_u32 = dynamic_cast<const DenseMatrix<uint32_t>*>(&a);
    auto* b_u32 = dynamic_cast<const DenseMatrix<uint32_t>*>(&b);
    auto* c_u32 = dynamic_cast<DenseMatrix<uint32_t>*>(&result);

    if (a_u32 && b_u32 && c_u32) {
        debug_trace::set_last("cpu.matmul.u32");
        matmul_impl(a_u32, b_u32, c_u32);
        return;
    }

    // 3i. Unsigned Integer (UInt64)
    auto* a_u64 = dynamic_cast<const DenseMatrix<uint64_t>*>(&a);
    auto* b_u64 = dynamic_cast<const DenseMatrix<uint64_t>*>(&b);
    auto* c_u64 = dynamic_cast<DenseMatrix<uint64_t>*>(&result);

    if (a_u64 && b_u64 && c_u64) {
        debug_trace::set_last("cpu.matmul.u64");
        matmul_impl(a_u64, b_u64, c_u64);
        return;
    }

    // 3c. Mixed Integer (Int16/Int32) -> Int32
    // PromotionResolver selects int32 if either operand is int32.
    if (c_i32 &&
        ((a.get_data_type() == DataType::INT16 || a.get_data_type() == DataType::INT32) &&
         (b.get_data_type() == DataType::INT16 || b.get_data_type() == DataType::INT32)) &&
        !(a_i32 && b_i32)) {
        debug_trace::set_last("cpu.matmul.mixed_int_fallback_i32");
        const uint64_t n = a.size();
        if (b.size() != n || result.size() != n) {
            throw std::invalid_argument("Dimension mismatch");
        }

        const bool t_a = a.is_transposed();
        const bool t_b = b.is_transposed();
        const bool t_c = c_i32->is_transposed();

        int32_t* c_data = c_i32->data();

        ParallelFor(0, n, [&](size_t i) {
            for (size_t j = 0; j < n; ++j) {
                int64_t sum = 0;
                for (size_t k = 0; k < n; ++k) {
                    int64_t va = 0;
                    int64_t vb = 0;

                    if (a_i32) {
                        const int32_t* a_data = a_i32->data();
                        va = static_cast<int64_t>(t_a ? a_data[k * n + i] : a_data[i * n + k]);
                    } else if (a_i16) {
                        const int16_t* a_data = a_i16->data();
                        va = static_cast<int64_t>(t_a ? a_data[k * n + i] : a_data[i * n + k]);
                    }

                    if (b_i32) {
                        const int32_t* b_data = b_i32->data();
                        vb = static_cast<int64_t>(t_b ? b_data[j * n + k] : b_data[k * n + j]);
                    } else if (b_i16) {
                        const int16_t* b_data = b_i16->data();
                        vb = static_cast<int64_t>(t_b ? b_data[j * n + k] : b_data[k * n + j]);
                    }

                    const int64_t term = va * vb;
                    sum = checked_add_int64(sum, term);
                }

                if (sum > static_cast<int64_t>((std::numeric_limits<int32_t>::max)()) ||
                    sum < static_cast<int64_t>((std::numeric_limits<int32_t>::min)())) {
                    throw std::overflow_error(
                        "Integer matmul overflow: mixed int16/int32 result does not fit in int32 output");
                }

                const int32_t out = static_cast<int32_t>(sum);
                if (t_c) {
                    c_data[j * n + i] = out;
                } else {
                    c_data[i * n + j] = out;
                }
            }
        });

        c_i32->set_scalar(a.get_scalar() * b.get_scalar());
        return;
    }

    // 4. Bit x {Float64, Float32, Int32} (scale-first: keep bit-packed inputs)
    auto* a_bit = dynamic_cast<const DenseMatrix<bool>*>(&a);
    auto* b_bit = dynamic_cast<const DenseMatrix<bool>*>(&b);

    if (a_bit && c_f64 && b_f64) {
        if (a.size() != b.size() || result.size() != a.size()) {
            throw std::invalid_argument("Dimension mismatch");
        }
        // For correctness, only use raw-data fast path when both are non-transposed.
        if (!a_bit->is_transposed() && !b_f64->is_transposed()) {
            debug_trace::set_last("cpu.matmul.bit_x_f64");
            const uint64_t n = a.size();
            const uint64_t* a_data = a_bit->data();
            const uint64_t words_per_row = a_bit->stride_bytes() / 8;
            const double* b_data = b_f64->data();
            double* c_data = c_f64->data();

            ParallelFor(0, n, [&](size_t i) {
                std::vector<double> acc(n, 0.0);
                const uint64_t* a_row = a_data + i * words_per_row;
                for (uint64_t w = 0; w < words_per_row; ++w) {
                    uint64_t word = a_row[w];
                    while (word) {
                        const uint64_t bit = static_cast<uint64_t>(std::countr_zero(word));
                        const uint64_t k = w * 64 + bit;
                        if (k < n) {
                            const double* b_row = b_data + k * n;
                            for (uint64_t j = 0; j < n; ++j) {
                                acc[j] += b_row[j];
                            }
                        }
                        word &= (word - 1);
                    }
                }
                double* c_row = c_data + i * n;
                for (uint64_t j = 0; j < n; ++j) {
                    c_row[j] = acc[j];
                }
            });

            c_f64->set_scalar(a_bit->get_scalar() * b_f64->get_scalar());
            return;
        }
    }

    if (a_bit && c_f32 && b_f32) {
        if (a.size() != b.size() || result.size() != a.size()) {
            throw std::invalid_argument("Dimension mismatch");
        }
        if (!a_bit->is_transposed() && !b_f32->is_transposed()) {
            debug_trace::set_last("cpu.matmul.bit_x_f32");
            const uint64_t n = a.size();
            const uint64_t* a_data = a_bit->data();
            const uint64_t words_per_row = a_bit->stride_bytes() / 8;
            const float* b_data = b_f32->data();
            float* c_data = c_f32->data();

            ParallelFor(0, n, [&](size_t i) {
                std::vector<float> acc(n, 0.0f);
                const uint64_t* a_row = a_data + i * words_per_row;
                for (uint64_t w = 0; w < words_per_row; ++w) {
                    uint64_t word = a_row[w];
                    while (word) {
                        const uint64_t bit = static_cast<uint64_t>(std::countr_zero(word));
                        const uint64_t k = w * 64 + bit;
                        if (k < n) {
                            const float* b_row = b_data + k * n;
                            for (uint64_t j = 0; j < n; ++j) {
                                acc[j] += b_row[j];
                            }
                        }
                        word &= (word - 1);
                    }
                }
                float* c_row = c_data + i * n;
                for (uint64_t j = 0; j < n; ++j) {
                    c_row[j] = acc[j];
                }
            });

            c_f32->set_scalar(a_bit->get_scalar() * b_f32->get_scalar());
            return;
        }
    }

    if (a_bit && c_i32 && b_i32) {
        if (a.size() != b.size() || result.size() != a.size()) {
            throw std::invalid_argument("Dimension mismatch");
        }
        if (!a_bit->is_transposed() && !b_i32->is_transposed()) {
            debug_trace::set_last("cpu.matmul.bit_x_i32");
            const uint64_t n = a.size();
            const uint64_t* a_data = a_bit->data();
            const uint64_t words_per_row = a_bit->stride_bytes() / 8;
            const int32_t* b_data = b_i32->data();
            int32_t* c_data = c_i32->data();

            ParallelFor(0, n, [&](size_t i) {
                std::vector<int64_t> acc(n, 0);
                const uint64_t* a_row = a_data + i * words_per_row;
                for (uint64_t w = 0; w < words_per_row; ++w) {
                    uint64_t word = a_row[w];
                    while (word) {
                        const uint64_t bit = static_cast<uint64_t>(std::countr_zero(word));
                        const uint64_t k = w * 64 + bit;
                        if (k < n) {
                            const int32_t* b_row = b_data + k * n;
                            for (uint64_t j = 0; j < n; ++j) {
                                acc[j] = checked_add_int64(acc[j], static_cast<int64_t>(b_row[j]));
                            }
                        }
                        word &= (word - 1);
                    }
                }

                int32_t* c_row = c_data + i * n;
                for (uint64_t j = 0; j < n; ++j) {
                    const int64_t v = acc[j];
                    if (v > static_cast<int64_t>((std::numeric_limits<int32_t>::max)()) ||
                        v < static_cast<int64_t>((std::numeric_limits<int32_t>::min)())) {
                        throw std::overflow_error("Integer matmul overflow: bit×int32 result does not fit in int32 output");
                    }
                    c_row[j] = static_cast<int32_t>(v);
                }
            });

            c_i32->set_scalar(a_bit->get_scalar() * b_i32->get_scalar());
            return;
        }
    }

    if (a_bit && c_i16 && b_i16) {
        if (a.size() != b.size() || result.size() != a.size()) {
            throw std::invalid_argument("Dimension mismatch");
        }
        if (!a_bit->is_transposed() && !b_i16->is_transposed()) {
            debug_trace::set_last("cpu.matmul.bit_x_i16");
            const uint64_t n = a.size();
            const uint64_t* a_data = a_bit->data();
            const uint64_t words_per_row = a_bit->stride_bytes() / 8;
            const int16_t* b_data = b_i16->data();
            int16_t* c_data = c_i16->data();

            ParallelFor(0, n, [&](size_t i) {
                std::vector<int64_t> acc(n, 0);
                const uint64_t* a_row = a_data + i * words_per_row;
                for (uint64_t w = 0; w < words_per_row; ++w) {
                    uint64_t word = a_row[w];
                    while (word) {
                        const uint64_t bit = static_cast<uint64_t>(std::countr_zero(word));
                        const uint64_t k = w * 64 + bit;
                        if (k < n) {
                            const int16_t* b_row = b_data + k * n;
                            for (uint64_t j = 0; j < n; ++j) {
                                acc[j] = checked_add_int64(acc[j], static_cast<int64_t>(b_row[j]));
                            }
                        }
                        word &= (word - 1);
                    }
                }

                int16_t* c_row = c_data + i * n;
                for (uint64_t j = 0; j < n; ++j) {
                    const int64_t v = acc[j];
                    if (v > static_cast<int64_t>((std::numeric_limits<int16_t>::max)()) ||
                        v < static_cast<int64_t>((std::numeric_limits<int16_t>::min)())) {
                        throw std::overflow_error("Integer matmul overflow: bit×int16 result does not fit in int16 output");
                    }
                    c_row[j] = static_cast<int16_t>(v);
                }
            });

            c_i16->set_scalar(a_bit->get_scalar() * b_i16->get_scalar());
            return;
        }
    }

    // 5. BitMatrix Support (bit×bit -> int32)
    auto* c_int = dynamic_cast<DenseMatrix<int32_t>*>(&result);
    if (a_bit && b_bit && c_int) {
        debug_trace::set_last("cpu.matmul.bitbit_popcount");
        uint64_t n = a.size();

        const uint64_t* a_data = a_bit->data();
        uint64_t stride_bytes = a_bit->stride_bytes();
        uint64_t words_per_row = stride_bytes / 8;

        auto b_transposed_mat = std::make_unique<DenseBitMatrix>(n, "");
        b_transposed_mat->set_temporary(true);
        uint64_t* b_transposed_data = b_transposed_mat->data();

        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t j = 0; j < n; ++j) {
                if (b_bit->get(i, j)) {
                    uint64_t word_idx = i / 64;
                    uint64_t bit_idx = i % 64;
                    b_transposed_data[j * words_per_row + word_idx] |= (1ULL << bit_idx);
                }
            }
        }

        ParallelFor(0, n, [&](size_t i) {
            const uint64_t* a_row = a_data + i * words_per_row;
            for (size_t j = 0; j < n; ++j) {
                const uint64_t* b_col = b_transposed_data + j * words_per_row;

                int64_t dot_product = 0;
                for (size_t k = 0; k < words_per_row; ++k) {
                    dot_product = checked_add_int64(
                        dot_product,
                        static_cast<int64_t>(std::popcount(a_row[k] & b_col[k])));
                }
                if (dot_product > static_cast<int64_t>((std::numeric_limits<int32_t>::max)())) {
                    throw std::overflow_error("Integer matmul overflow: bit×bit result does not fit in int32 output");
                }
                c_int->set(i, j, static_cast<int32_t>(dot_product));
            }
        });

        c_int->set_scalar(a_bit->get_scalar() * b_bit->get_scalar());
        return;
    }

    // 5. Triangular Matrix Support (Double)
    auto* a_tri = dynamic_cast<const TriangularMatrix<double>*>(&a);
    auto* b_tri = dynamic_cast<const TriangularMatrix<double>*>(&b);
    auto* c_tri = dynamic_cast<TriangularMatrix<double>*>(&result);

    if (a_tri && b_tri && c_tri) {
        debug_trace::set_last("cpu.matmul.tri_f64");
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
        debug_trace::set_last("cpu.matmul.diag_f64");
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
        debug_trace::set_last("cpu.matmul.diag_x_dense_f64");
        uint64_t n = a.size();
        const double* a_data = a_diag->data();
        const double* b_data = b_f64->data();
        double* c_data = c_dense_dbl->data();
        
        // Scale rows of B by A's diagonal
        ParallelFor(0, n, [&](size_t i) {
            double scale = (a_data[i] * a_diag->get_scalar()).real();
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
        debug_trace::set_last("cpu.matmul.dense_x_diag_f64");
        uint64_t n = a.size();
        const double* a_data = a_f64->data();
        const double* b_data = b_diag->data();
        double* c_data = c_dense_dbl->data();
        
        // Scale columns of A by B's diagonal
        ParallelFor(0, n, [&](size_t i) {
            const double* a_row = a_data + i * n;
            double* c_row = c_data + i * n;
            for (size_t j = 0; j < n; ++j) {
                c_row[j] = (a_row[j] * b_data[j] * b_diag->get_scalar()).real();
            }
        });
        c_dense_dbl->set_scalar(a_f64->get_scalar());
        return;
    }

    // Fallback for generic types (Parallelized)
    uint64_t n = a.size();

    // Generic float32 result fallback for mixed-kind inputs.
    // NOTE: this computes through get_element_as_double (includes scalars) and stores raw values
    // into a float32 result with scalar=1.
    if (c_f32) {
        debug_trace::set_last("cpu.matmul.generic_fallback_f32");
        if (b.size() != n || result.size() != n) {
            throw std::invalid_argument("Dimension mismatch");
        }

        float* c_data = c_f32->data();
        const bool t_c = c_f32->is_transposed();

        ParallelFor(0, n, [&](size_t i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < n; ++k) {
                    sum += static_cast<float>(a.get_element_as_double(i, k)) *
                           static_cast<float>(b.get_element_as_double(k, j));
                }
                if (t_c) {
                    c_data[j * n + i] = sum;
                } else {
                    c_data[i * n + j] = sum;
                }
            }
        });
        return;
    }

    auto* res_dense = dynamic_cast<DenseMatrix<double>*>(&result);
    if (!res_dense) {
        debug_trace::set_last("cpu.matmul.generic_fallback_result_not_dense_f64");
        // If result is not dense double, we might need other handlers or throw
        // For now, let's assume result is dense double as per original CpuDevice logic
        throw std::runtime_error("CpuSolver::matmul currently only supports DenseMatrix<double> result for generic inputs");
    }

    debug_trace::set_last("cpu.matmul.generic_fallback_f64");
    
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

void CpuSolver::batch_gemv(const MatrixBase& A, const double* x_data, double* y_data, size_t b) {
    uint64_t n = A.size();
    
    // Check for optimized types
    auto* f32 = dynamic_cast<const DenseMatrix<float>*>(&A);
    auto* f64 = dynamic_cast<const DenseMatrix<double>*>(&A);

    ParallelFor(0, n, [&](size_t i) {
        // Accumulator for row i of Y (size b)
        std::vector<double> acc(b, 0.0);
        
        // Scan row i of A
        for (size_t j = 0; j < n; ++j) {
            double a_val;
            if (f32) a_val = (double)f32->get(i, j);
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
                    T val_a;
                    T val_b;
                    if constexpr (is_std_complex_v<T>) {
                        val_a = static_cast<T>(a.get_element_as_complex(i, j));
                        val_b = static_cast<T>(b.get_element_as_complex(i, j));
                    } else {
                        val_a = pycauset::scalar::from_double<T>(a.get_element_as_double(i, j));
                        val_b = pycauset::scalar::from_double<T>(b.get_element_as_double(i, j));
                    }
                    res_dense->set(i, j, op(val_a, val_b));
                }
            });
            return;
        }

        if constexpr (!is_std_complex_v<T>) {
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

            // 4. SymmetricMatrix Result
            if (auto* res_sym = dynamic_cast<SymmetricMatrix<T>*>(&result)) {
                ParallelFor(0, n, [&](size_t i) {
                    // SymmetricMatrix stores upper triangle (including diagonal)
                    for (uint64_t j = i; j < n; ++j) {
                        T val = op(static_cast<T>(a.get_element_as_double(i, j)),
                                   static_cast<T>(b.get_element_as_double(i, j)));
                        res_sym->set(i, j, val);
                    }
                });
                return;
            }
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
            T s;
            if constexpr (is_std_complex_v<T>) {
                using V = typename T::value_type;
                s = T{static_cast<V>(scalar), static_cast<V>(0)};
            } else {
                s = pycauset::scalar::from_double<T>(scalar);
            }
            ParallelFor(0, n * n, [&](size_t i) {
                res_data[i] = a_data[i] * s;
            });
            return;
        }
        
        ParallelFor(0, n, [&](size_t i) {
            for (size_t j = 0; j < n; ++j) {
                T val_a;
                T s;
                if constexpr (is_std_complex_v<T>) {
                    val_a = static_cast<T>(a.get_element_as_complex(i, j));
                    using V = typename T::value_type;
                    s = T{static_cast<V>(scalar), static_cast<V>(0)};
                } else {
                    val_a = pycauset::scalar::from_double<T>(a.get_element_as_double(i, j));
                    s = pycauset::scalar::from_double<T>(scalar);
                }
                res->set(i, j, val_a * s);
            }
        });
    }
}

void CpuSolver::add(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    DataType dtype = result.get_data_type();
    if (dtype == DataType::COMPLEX_FLOAT16) {
        if (auto* out = dynamic_cast<ComplexFloat16Matrix*>(&result)) {
            binary_op_complex16_impl(a, b, *out, std::plus<>());
            return;
        }
        throw std::runtime_error("CpuSolver::add complex_float16 result type mismatch");
    }
    if (dtype == DataType::COMPLEX_FLOAT32) {
        binary_op_impl<std::complex<float>>(a, b, result, std::plus<>());
        return;
    }
    if (dtype == DataType::COMPLEX_FLOAT64) {
        binary_op_impl<std::complex<double>>(a, b, result, std::plus<>());
        return;
    }

    if (dtype == DataType::FLOAT64) {
        binary_op_impl<double>(a, b, result, std::plus<>());
    } else if (dtype == DataType::FLOAT16) {
        binary_op_impl<float16_t>(a, b, result, std::plus<>());
    } else if (dtype == DataType::FLOAT32) {
        binary_op_impl<float>(a, b, result, std::plus<>());
    } else if (dtype == DataType::INT8) {
        binary_op_impl<int8_t>(a, b, result, std::plus<>());
    } else if (dtype == DataType::INT16) {
        binary_op_impl<int16_t>(a, b, result, std::plus<>());
    } else if (dtype == DataType::INT32) {
        binary_op_impl<int32_t>(a, b, result, std::plus<>());
    } else if (dtype == DataType::INT64) {
        binary_op_impl<int64_t>(a, b, result, std::plus<>());
    } else if (dtype == DataType::UINT8) {
        binary_op_impl<uint8_t>(a, b, result, std::plus<>());
    } else if (dtype == DataType::UINT16) {
        binary_op_impl<uint16_t>(a, b, result, std::plus<>());
    } else if (dtype == DataType::UINT32) {
        binary_op_impl<uint32_t>(a, b, result, std::plus<>());
    } else if (dtype == DataType::UINT64) {
        binary_op_impl<uint64_t>(a, b, result, std::plus<>());
    } else {
        throw std::runtime_error("CpuSolver::add result data type not supported");
    }
}

void CpuSolver::subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    DataType dtype = result.get_data_type();
    if (dtype == DataType::COMPLEX_FLOAT16) {
        if (auto* out = dynamic_cast<ComplexFloat16Matrix*>(&result)) {
            binary_op_complex16_impl(a, b, *out, std::minus<>());
            return;
        }
        throw std::runtime_error("CpuSolver::subtract complex_float16 result type mismatch");
    }
    if (dtype == DataType::COMPLEX_FLOAT32) {
        binary_op_impl<std::complex<float>>(a, b, result, std::minus<>());
        return;
    }
    if (dtype == DataType::COMPLEX_FLOAT64) {
        binary_op_impl<std::complex<double>>(a, b, result, std::minus<>());
        return;
    }

    if (dtype == DataType::FLOAT64) {
        binary_op_impl<double>(a, b, result, std::minus<>());
    } else if (dtype == DataType::FLOAT16) {
        binary_op_impl<float16_t>(a, b, result, std::minus<>());
    } else if (dtype == DataType::FLOAT32) {
        binary_op_impl<float>(a, b, result, std::minus<>());
    } else if (dtype == DataType::INT8) {
        binary_op_impl<int8_t>(a, b, result, std::minus<>());
    } else if (dtype == DataType::INT16) {
        binary_op_impl<int16_t>(a, b, result, std::minus<>());
    } else if (dtype == DataType::INT32) {
        binary_op_impl<int32_t>(a, b, result, std::minus<>());
    } else if (dtype == DataType::INT64) {
        binary_op_impl<int64_t>(a, b, result, std::minus<>());
    } else if (dtype == DataType::UINT8) {
        binary_op_impl<uint8_t>(a, b, result, std::minus<>());
    } else if (dtype == DataType::UINT16) {
        binary_op_impl<uint16_t>(a, b, result, std::minus<>());
    } else if (dtype == DataType::UINT32) {
        binary_op_impl<uint32_t>(a, b, result, std::minus<>());
    } else if (dtype == DataType::UINT64) {
        binary_op_impl<uint64_t>(a, b, result, std::minus<>());
    } else {
        throw std::runtime_error("CpuSolver::subtract result data type not supported");
    }
}

void CpuSolver::elementwise_multiply(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    DataType dtype = result.get_data_type();
    if (dtype == DataType::COMPLEX_FLOAT16) {
        if (auto* out = dynamic_cast<ComplexFloat16Matrix*>(&result)) {
            binary_op_complex16_impl(a, b, *out, std::multiplies<>());
            return;
        }
        throw std::runtime_error("CpuSolver::elementwise_multiply complex_float16 result type mismatch");
    }
    if (dtype == DataType::COMPLEX_FLOAT32) {
        binary_op_impl<std::complex<float>>(a, b, result, std::multiplies<>());
        return;
    }
    if (dtype == DataType::COMPLEX_FLOAT64) {
        binary_op_impl<std::complex<double>>(a, b, result, std::multiplies<>());
        return;
    }

    if (dtype == DataType::BIT) {
        auto* out = dynamic_cast<DenseBitMatrix*>(&result);
        if (!out) {
            throw std::runtime_error("CpuSolver::elementwise_multiply bit result type mismatch");
        }
        const uint64_t n = out->size();
        debug_trace::set_last("cpu.elementwise_multiply.bit");
        ParallelFor(0, n, [&](size_t i) {
            for (uint64_t j = 0; j < n; ++j) {
                const bool va = (a.get_element_as_double(i, j) != 0.0);
                const bool vb = (b.get_element_as_double(i, j) != 0.0);
                out->set(i, j, va && vb);
            }
        });
        out->set_scalar(1.0);
        return;
    }

    if (dtype == DataType::FLOAT64) {
        binary_op_impl<double>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::FLOAT16) {
        binary_op_impl<float16_t>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::FLOAT32) {
        binary_op_impl<float>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::INT8) {
        binary_op_impl<int8_t>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::INT16) {
        binary_op_impl<int16_t>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::INT32) {
        binary_op_impl<int32_t>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::INT64) {
        binary_op_impl<int64_t>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::UINT8) {
        binary_op_impl<uint8_t>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::UINT16) {
        binary_op_impl<uint16_t>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::UINT32) {
        binary_op_impl<uint32_t>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::UINT64) {
        binary_op_impl<uint64_t>(a, b, result, std::multiplies<>());
    } else {
        throw std::runtime_error("CpuSolver::elementwise_multiply result data type not supported");
    }
}

void CpuSolver::multiply_scalar(const MatrixBase& a, double scalar, MatrixBase& result) {
    DataType dtype = result.get_data_type();
    if (dtype == DataType::COMPLEX_FLOAT16) {
        if (auto* out = dynamic_cast<ComplexFloat16Matrix*>(&result)) {
            const uint64_t n = out->size();
            ParallelFor(0, n, [&](size_t i) {
                for (size_t j = 0; j < n; ++j) {
                    out->set(i, j, a.get_element_as_complex(i, j) * scalar);
                }
            });
            out->set_scalar(1.0);
            return;
        }
        throw std::runtime_error("CpuSolver::multiply_scalar complex_float16 result type mismatch");
    }
    if (dtype == DataType::COMPLEX_FLOAT32) {
        if (auto* r = dynamic_cast<DenseMatrix<std::complex<float>>*>(&result)) {
            scalar_op_impl(a, scalar, r);
            return;
        }
        throw std::runtime_error("CpuSolver::multiply_scalar complex_float32 result type mismatch");
    }
    if (dtype == DataType::COMPLEX_FLOAT64) {
        if (auto* r = dynamic_cast<DenseMatrix<std::complex<double>>*>(&result)) {
            scalar_op_impl(a, scalar, r);
            return;
        }
        throw std::runtime_error("CpuSolver::multiply_scalar complex_float64 result type mismatch");
    }

    if (auto* r = dynamic_cast<DenseMatrix<double>*>(&result)) {
        scalar_op_impl(a, scalar, r);
    } else if (auto* r = dynamic_cast<DenseMatrix<float>*>(&result)) {
        scalar_op_impl(a, scalar, r);
    } else if (auto* r = dynamic_cast<DenseMatrix<float16_t>*>(&result)) {
        scalar_op_impl(a, scalar, r);
    } else {
        throw std::runtime_error("CpuSolver::multiply_scalar result type not supported");
    }
}

double CpuSolver::dot(const VectorBase& a, const VectorBase& b) {
    uint64_t n = a.size();
    if (b.size() != n) throw std::invalid_argument("Vector dimensions mismatch");

    const auto dt_a = a.get_data_type();
    const auto dt_b = b.get_data_type();
    if (dt_a == DataType::COMPLEX_FLOAT16 || dt_a == DataType::COMPLEX_FLOAT32 || dt_a == DataType::COMPLEX_FLOAT64 ||
        dt_b == DataType::COMPLEX_FLOAT16 || dt_b == DataType::COMPLEX_FLOAT32 || dt_b == DataType::COMPLEX_FLOAT64) {
        throw std::runtime_error("CpuSolver::dot not implemented for complex vectors");
    }

    // 1. Dense Double * Dense Double
    auto* a_dbl = dynamic_cast<const DenseVector<double>*>(&a);
    auto* b_dbl = dynamic_cast<const DenseVector<double>*>(&b);

    if (a_dbl && b_dbl) {
        debug_trace::set_last("cpu.dot.f64");
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
        debug_trace::set_last("cpu.dot.bitbit_popcount");
        const uint64_t* a_data = a_bit->data();
        const uint64_t* b_data = b_bit->data();
        uint64_t num_words = (n + 63) / 64;
        
        int64_t count = 0;
        #pragma omp parallel for reduction(+:count)
        for (int64_t i = 0; i < (int64_t)num_words; ++i) {
            count += std::popcount(a_data[i] & b_data[i]);
        }
        const double scale = a.get_scalar().real() * b.get_scalar().real();
        return static_cast<double>(count) * scale;
    }

    // Fallback
    debug_trace::set_last("cpu.dot.generic_fallback");
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
        const DataType res_dt = result.get_data_type();
        if (res_dt == DataType::COMPLEX_FLOAT16 || res_dt == DataType::COMPLEX_FLOAT32 || res_dt == DataType::COMPLEX_FLOAT64) {
            debug_trace::set_last("cpu.matvec.identity.complex");
            const std::complex<double> scalar = m.get_scalar();
            if (auto* out16 = dynamic_cast<ComplexFloat16Vector*>(&result)) {
                auto* rdst = out16->real_data();
                auto* idst = out16->imag_data();
                ParallelFor(0, n, [&](size_t i) {
                    const std::complex<double> z = v.get_element_as_complex(i) * scalar;
                    rdst[i] = float16_t(z.real());
                    idst[i] = float16_t(z.imag());
                });
                out16->set_scalar(1.0);
                return;
            }
            if (auto* out32 = dynamic_cast<DenseVector<std::complex<float>>*>(&result)) {
                auto* dst = out32->data();
                ParallelFor(0, n, [&](size_t i) {
                    const std::complex<double> z = v.get_element_as_complex(i) * scalar;
                    dst[i] = std::complex<float>(static_cast<float>(z.real()), static_cast<float>(z.imag()));
                });
                out32->set_scalar(1.0);
                return;
            }
            if (auto* out64 = dynamic_cast<DenseVector<std::complex<double>>*>(&result)) {
                auto* dst = out64->data();
                ParallelFor(0, n, [&](size_t i) { dst[i] = v.get_element_as_complex(i) * scalar; });
                out64->set_scalar(1.0);
                return;
            }
            throw std::runtime_error("Unsupported complex result type for matrix_vector_multiply");
        }

        double scalar = m.get_scalar().real();
        if (auto* res_int = dynamic_cast<DenseVector<int32_t>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_int->set(i, (int32_t)(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_i8 = dynamic_cast<DenseVector<int8_t>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_i8->set(i, static_cast<int8_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_i16 = dynamic_cast<DenseVector<int16_t>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_i16->set(i, static_cast<int16_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_i64 = dynamic_cast<DenseVector<int64_t>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_i64->set(i, static_cast<int64_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_u8 = dynamic_cast<DenseVector<uint8_t>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_u8->set(i, static_cast<uint8_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_u16 = dynamic_cast<DenseVector<uint16_t>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_u16->set(i, static_cast<uint16_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_u32 = dynamic_cast<DenseVector<uint32_t>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_u32->set(i, static_cast<uint32_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_u64 = dynamic_cast<DenseVector<uint64_t>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_u64->set(i, static_cast<uint64_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_f16 = dynamic_cast<DenseVector<float16_t>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_f16->set(i, float16_t(static_cast<float>(v.get_element_as_double(i) * scalar)));
            });
        } else if (auto* res_f32 = dynamic_cast<DenseVector<float>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_f32->set(i, static_cast<float>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_dbl = dynamic_cast<DenseVector<double>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_dbl->set(i, v.get_element_as_double(i) * scalar);
            });
        }
        return;
    }

    // Complex path (covers complex matrices/vectors and mixed real/complex producing complex result)
    const DataType res_dt = result.get_data_type();
    if (res_dt == DataType::COMPLEX_FLOAT16 || res_dt == DataType::COMPLEX_FLOAT32 || res_dt == DataType::COMPLEX_FLOAT64) {
        debug_trace::set_last("cpu.matvec.complex.generic");
        if (auto* out16 = dynamic_cast<ComplexFloat16Vector*>(&result)) {
            auto* rdst = out16->real_data();
            auto* idst = out16->imag_data();
            ParallelFor(0, n, [&](size_t i) {
                std::complex<double> sum = 0.0;
                for (uint64_t j = 0; j < n; ++j) {
                    sum += m.get_element_as_complex(i, j) * v.get_element_as_complex(j);
                }
                rdst[i] = float16_t(sum.real());
                idst[i] = float16_t(sum.imag());
            });
            out16->set_scalar(1.0);
            return;
        }
        if (auto* out32 = dynamic_cast<DenseVector<std::complex<float>>*>(&result)) {
            auto* dst = out32->data();
            ParallelFor(0, n, [&](size_t i) {
                std::complex<double> sum = 0.0;
                for (uint64_t j = 0; j < n; ++j) {
                    sum += m.get_element_as_complex(i, j) * v.get_element_as_complex(j);
                }
                dst[i] = std::complex<float>(static_cast<float>(sum.real()), static_cast<float>(sum.imag()));
            });
            out32->set_scalar(1.0);
            return;
        }
        if (auto* out64 = dynamic_cast<DenseVector<std::complex<double>>*>(&result)) {
            auto* dst = out64->data();
            ParallelFor(0, n, [&](size_t i) {
                std::complex<double> sum = 0.0;
                for (uint64_t j = 0; j < n; ++j) {
                    sum += m.get_element_as_complex(i, j) * v.get_element_as_complex(j);
                }
                dst[i] = sum;
            });
            out64->set_scalar(1.0);
            return;
        }
        throw std::runtime_error("Unsupported complex result type for matrix_vector_multiply");
    }

    // BitMatrix * BitVector Optimization
    auto* m_bit = dynamic_cast<const DenseMatrix<bool>*>(&m);
    auto* v_bit = dynamic_cast<const DenseVector<bool>*>(&v);
    auto* res_int = dynamic_cast<DenseVector<int32_t>*>(&result);
    
    if (m_bit && v_bit && res_int) {
        debug_trace::set_last("cpu.matvec.bit_x_bit_popcount");
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

        res_int->set_scalar(m_bit->get_scalar() * v_bit->get_scalar());
        return;
    }

    // BitMatrix * DenseVector<int32_t>
    auto* v_i32 = dynamic_cast<const DenseVector<int32_t>*>(&v);
    auto* res_i32 = dynamic_cast<DenseVector<int32_t>*>(&result);
    if (m_bit && v_i32 && res_i32) {
        if (m_bit->is_transposed()) {
            // Fall through to generic path for correctness.
        } else {
            debug_trace::set_last("cpu.matvec.bit_x_i32");
            const uint64_t* m_data = m_bit->data();
            const int32_t* v_data = v_i32->data();
            int32_t* res_data = res_i32->data();

            const uint64_t words_per_row = m_bit->stride_bytes() / 8;
            ParallelFor(0, n, [&](size_t i) {
                const uint64_t* row_ptr = m_data + i * words_per_row;
                int64_t sum = 0;
                for (uint64_t w = 0; w < words_per_row; ++w) {
                    uint64_t word = row_ptr[w];
                    while (word) {
                        const uint64_t bit = static_cast<uint64_t>(std::countr_zero(word));
                        const uint64_t k = w * 64 + bit;
                        if (k < n) {
                            sum = checked_add_int64(sum, static_cast<int64_t>(v_data[k]));
                        }
                        word &= (word - 1);
                    }
                }
                if (sum > static_cast<int64_t>((std::numeric_limits<int32_t>::max)()) ||
                    sum < static_cast<int64_t>((std::numeric_limits<int32_t>::min)())) {
                    throw std::overflow_error("Integer matvec overflow: bit×int32 result does not fit in int32 output");
                }
                res_data[i] = static_cast<int32_t>(sum);
            });

            res_i32->set_scalar(m_bit->get_scalar() * v_i32->get_scalar());
            return;
        }
    }

    // BitMatrix * DenseVector<int16_t>
    auto* v_i16 = dynamic_cast<const DenseVector<int16_t>*>(&v);
    auto* res_i16 = dynamic_cast<DenseVector<int16_t>*>(&result);
    if (m_bit && v_i16 && res_i16) {
        if (m_bit->is_transposed()) {
            // Fall through to generic path for correctness.
        } else {
            debug_trace::set_last("cpu.matvec.bit_x_i16");
            const uint64_t* m_data = m_bit->data();
            const int16_t* v_data = v_i16->data();
            int16_t* res_data = res_i16->data();

            const uint64_t words_per_row = m_bit->stride_bytes() / 8;
            ParallelFor(0, n, [&](size_t i) {
                const uint64_t* row_ptr = m_data + i * words_per_row;
                int64_t sum = 0;
                for (uint64_t w = 0; w < words_per_row; ++w) {
                    uint64_t word = row_ptr[w];
                    while (word) {
                        const uint64_t bit = static_cast<uint64_t>(std::countr_zero(word));
                        const uint64_t k = w * 64 + bit;
                        if (k < n) {
                            sum = checked_add_int64(sum, static_cast<int64_t>(v_data[k]));
                        }
                        word &= (word - 1);
                    }
                }
                if (sum > static_cast<int64_t>((std::numeric_limits<int16_t>::max)()) ||
                    sum < static_cast<int64_t>((std::numeric_limits<int16_t>::min)())) {
                    throw std::overflow_error("Integer matvec overflow: bit×int16 result does not fit in int16 output");
                }
                res_data[i] = static_cast<int16_t>(sum);
            });

            res_i16->set_scalar(m_bit->get_scalar() * v_i16->get_scalar());
            return;
        }
    }

    // BitMatrix * DenseVector<double>
    auto* v_f64 = dynamic_cast<const DenseVector<double>*>(&v);
    auto* res_f64 = dynamic_cast<DenseVector<double>*>(&result);
    if (m_bit && v_f64 && res_f64) {
        if (m_bit->is_transposed()) {
            // Fall through to generic path for correctness.
        } else {
            debug_trace::set_last("cpu.matvec.bit_x_f64");
            const uint64_t* m_data = m_bit->data();
            const double* v_data = v_f64->data();
            double* res_data = res_f64->data();

            const uint64_t words_per_row = m_bit->stride_bytes() / 8;
            ParallelFor(0, n, [&](size_t i) {
                const uint64_t* row_ptr = m_data + i * words_per_row;
                double sum = 0.0;
                for (uint64_t w = 0; w < words_per_row; ++w) {
                    uint64_t word = row_ptr[w];
                    while (word) {
                        const uint64_t bit = static_cast<uint64_t>(std::countr_zero(word));
                        const uint64_t k = w * 64 + bit;
                        if (k < n) {
                            sum += v_data[k];
                        }
                        word &= (word - 1);
                    }
                }
                res_data[i] = sum;
            });

            res_f64->set_scalar(m_bit->get_scalar() * v_f64->get_scalar());
            return;
        }
    }

    // Generic Fallback
    if (auto* res_int = dynamic_cast<DenseVector<int32_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            int64_t sum = 0;
            for (uint64_t j = 0; j < n; ++j) {
                sum += static_cast<int64_t>(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_int->set(i, static_cast<int32_t>(sum));
        });
    } else if (auto* res_i8_fb = dynamic_cast<DenseVector<int8_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            int64_t sum = 0;
            for (uint64_t j = 0; j < n; ++j) {
                sum += static_cast<int64_t>(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_i8_fb->set(i, static_cast<int8_t>(sum));
        });
    } else if (auto* res_i16_fb = dynamic_cast<DenseVector<int16_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            int64_t sum = 0;
            for (uint64_t j = 0; j < n; ++j) {
                sum += static_cast<int64_t>(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_i16_fb->set(i, static_cast<int16_t>(sum));
        });
    } else if (auto* res_i64_fb = dynamic_cast<DenseVector<int64_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            int64_t sum = 0;
            for (uint64_t j = 0; j < n; ++j) {
                sum += static_cast<int64_t>(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_i64_fb->set(i, sum);
        });
    } else if (auto* res_u8_fb = dynamic_cast<DenseVector<uint8_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            uint64_t sum = 0;
            for (uint64_t j = 0; j < n; ++j) {
                sum += static_cast<uint64_t>(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_u8_fb->set(i, static_cast<uint8_t>(sum));
        });
    } else if (auto* res_u16_fb = dynamic_cast<DenseVector<uint16_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            uint64_t sum = 0;
            for (uint64_t j = 0; j < n; ++j) {
                sum += static_cast<uint64_t>(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_u16_fb->set(i, static_cast<uint16_t>(sum));
        });
    } else if (auto* res_u32_fb = dynamic_cast<DenseVector<uint32_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            uint64_t sum = 0;
            for (uint64_t j = 0; j < n; ++j) {
                sum += static_cast<uint64_t>(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_u32_fb->set(i, static_cast<uint32_t>(sum));
        });
    } else if (auto* res_u64_fb = dynamic_cast<DenseVector<uint64_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            uint64_t sum = 0;
            for (uint64_t j = 0; j < n; ++j) {
                sum += static_cast<uint64_t>(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_u64_fb->set(i, sum);
        });
    } else if (auto* res_dbl = dynamic_cast<DenseVector<double>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            double sum = 0.0;
            for (uint64_t j = 0; j < n; ++j) {
                sum += m.get_element_as_double(i, j) * v.get_element_as_double(j);
            }
            res_dbl->set(i, sum);
        });
    } else if (auto* res_f32 = dynamic_cast<DenseVector<float>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            float sum = 0.0f;
            for (uint64_t j = 0; j < n; ++j) {
                sum += static_cast<float>(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_f32->set(i, sum);
        });
    } else if (auto* res_f16 = dynamic_cast<DenseVector<float16_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            float sum = 0.0f;
            for (uint64_t j = 0; j < n; ++j) {
                sum += static_cast<float>(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_f16->set(i, float16_t(sum));
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
        const DataType res_dt = result.get_data_type();
        if (res_dt == DataType::COMPLEX_FLOAT16 || res_dt == DataType::COMPLEX_FLOAT32 || res_dt == DataType::COMPLEX_FLOAT64) {
            debug_trace::set_last("cpu.vecmat.identity.complex");
            const std::complex<double> scalar = m.get_scalar();
            if (auto* out16 = dynamic_cast<ComplexFloat16Vector*>(&result)) {
                auto* rdst = out16->real_data();
                auto* idst = out16->imag_data();
                ParallelFor(0, n, [&](size_t i) {
                    const std::complex<double> z = v.get_element_as_complex(i) * scalar;
                    rdst[i] = float16_t(z.real());
                    idst[i] = float16_t(z.imag());
                });
                out16->set_scalar(1.0);
                result.set_transposed(true);
                return;
            }
            if (auto* out32 = dynamic_cast<DenseVector<std::complex<float>>*>(&result)) {
                auto* dst = out32->data();
                ParallelFor(0, n, [&](size_t i) {
                    const std::complex<double> z = v.get_element_as_complex(i) * scalar;
                    dst[i] = std::complex<float>(static_cast<float>(z.real()), static_cast<float>(z.imag()));
                });
                out32->set_scalar(1.0);
                result.set_transposed(true);
                return;
            }
            if (auto* out64 = dynamic_cast<DenseVector<std::complex<double>>*>(&result)) {
                auto* dst = out64->data();
                ParallelFor(0, n, [&](size_t i) { dst[i] = v.get_element_as_complex(i) * scalar; });
                out64->set_scalar(1.0);
                result.set_transposed(true);
                return;
            }
            throw std::runtime_error("Unsupported complex result type for vector_matrix_multiply");
        }

        double scalar = m.get_scalar().real();
        if (auto* res_int = dynamic_cast<DenseVector<int32_t>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_int->set(i, (int32_t)(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_i8 = dynamic_cast<DenseVector<int8_t>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_i8->set(i, static_cast<int8_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_i16 = dynamic_cast<DenseVector<int16_t>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_i16->set(i, static_cast<int16_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_i64 = dynamic_cast<DenseVector<int64_t>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_i64->set(i, static_cast<int64_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_u8 = dynamic_cast<DenseVector<uint8_t>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_u8->set(i, static_cast<uint8_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_u16 = dynamic_cast<DenseVector<uint16_t>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_u16->set(i, static_cast<uint16_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_u32 = dynamic_cast<DenseVector<uint32_t>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_u32->set(i, static_cast<uint32_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_u64 = dynamic_cast<DenseVector<uint64_t>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_u64->set(i, static_cast<uint64_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_f16 = dynamic_cast<DenseVector<float16_t>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_f16->set(i, float16_t(static_cast<float>(v.get_element_as_double(i) * scalar)));
            });
        } else if (auto* res_f32 = dynamic_cast<DenseVector<float>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_f32->set(i, static_cast<float>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_dbl = dynamic_cast<DenseVector<double>*>(&result)) {
            ParallelFor(0, n, [&](size_t i) {
                res_dbl->set(i, v.get_element_as_double(i) * scalar);
            });
        }
        result.set_transposed(true);
        return;
    }

    // Complex path (covers complex matrices/vectors and mixed real/complex producing complex result)
    const DataType res_dt = result.get_data_type();
    if (res_dt == DataType::COMPLEX_FLOAT16 || res_dt == DataType::COMPLEX_FLOAT32 || res_dt == DataType::COMPLEX_FLOAT64) {
        debug_trace::set_last("cpu.vecmat.complex.generic");
        if (auto* out16 = dynamic_cast<ComplexFloat16Vector*>(&result)) {
            auto* rdst = out16->real_data();
            auto* idst = out16->imag_data();
            ParallelFor(0, n, [&](size_t j) {
                std::complex<double> sum = 0.0;
                for (uint64_t i = 0; i < n; ++i) {
                    sum += v.get_element_as_complex(i) * m.get_element_as_complex(i, j);
                }
                rdst[j] = float16_t(sum.real());
                idst[j] = float16_t(sum.imag());
            });
            out16->set_scalar(1.0);
            return;
        }
        if (auto* out32 = dynamic_cast<DenseVector<std::complex<float>>*>(&result)) {
            auto* dst = out32->data();
            ParallelFor(0, n, [&](size_t j) {
                std::complex<double> sum = 0.0;
                for (uint64_t i = 0; i < n; ++i) {
                    sum += v.get_element_as_complex(i) * m.get_element_as_complex(i, j);
                }
                dst[j] = std::complex<float>(static_cast<float>(sum.real()), static_cast<float>(sum.imag()));
            });
            out32->set_scalar(1.0);
            return;
        }
        if (auto* out64 = dynamic_cast<DenseVector<std::complex<double>>*>(&result)) {
            auto* dst = out64->data();
            ParallelFor(0, n, [&](size_t j) {
                std::complex<double> sum = 0.0;
                for (uint64_t i = 0; i < n; ++i) {
                    sum += v.get_element_as_complex(i) * m.get_element_as_complex(i, j);
                }
                dst[j] = sum;
            });
            out64->set_scalar(1.0);
            return;
        }
        throw std::runtime_error("Unsupported complex result type for vector_matrix_multiply");
    }

    // Generic Fallback
    if (auto* res_int = dynamic_cast<DenseVector<int32_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t j) {
            int64_t sum = 0;
            for (uint64_t i = 0; i < n; ++i) {
                sum += static_cast<int64_t>(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_int->set(j, static_cast<int32_t>(sum));
        });
    } else if (auto* res_i8_fb = dynamic_cast<DenseVector<int8_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t j) {
            int64_t sum = 0;
            for (uint64_t i = 0; i < n; ++i) {
                sum += static_cast<int64_t>(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_i8_fb->set(j, static_cast<int8_t>(sum));
        });
    } else if (auto* res_i16_fb = dynamic_cast<DenseVector<int16_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t j) {
            int64_t sum = 0;
            for (uint64_t i = 0; i < n; ++i) {
                sum += static_cast<int64_t>(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_i16_fb->set(j, static_cast<int16_t>(sum));
        });
    } else if (auto* res_i64_fb = dynamic_cast<DenseVector<int64_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t j) {
            int64_t sum = 0;
            for (uint64_t i = 0; i < n; ++i) {
                sum += static_cast<int64_t>(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_i64_fb->set(j, sum);
        });
    } else if (auto* res_u8_fb = dynamic_cast<DenseVector<uint8_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t j) {
            uint64_t sum = 0;
            for (uint64_t i = 0; i < n; ++i) {
                sum += static_cast<uint64_t>(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_u8_fb->set(j, static_cast<uint8_t>(sum));
        });
    } else if (auto* res_u16_fb = dynamic_cast<DenseVector<uint16_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t j) {
            uint64_t sum = 0;
            for (uint64_t i = 0; i < n; ++i) {
                sum += static_cast<uint64_t>(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_u16_fb->set(j, static_cast<uint16_t>(sum));
        });
    } else if (auto* res_u32_fb = dynamic_cast<DenseVector<uint32_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t j) {
            uint64_t sum = 0;
            for (uint64_t i = 0; i < n; ++i) {
                sum += static_cast<uint64_t>(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_u32_fb->set(j, static_cast<uint32_t>(sum));
        });
    } else if (auto* res_u64_fb = dynamic_cast<DenseVector<uint64_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t j) {
            uint64_t sum = 0;
            for (uint64_t i = 0; i < n; ++i) {
                sum += static_cast<uint64_t>(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_u64_fb->set(j, sum);
        });
    } else if (auto* res_dbl = dynamic_cast<DenseVector<double>*>(&result)) {
        ParallelFor(0, n, [&](size_t j) {
            double sum = 0.0;
            for (uint64_t i = 0; i < n; ++i) {
                sum += v.get_element_as_double(i) * m.get_element_as_double(i, j);
            }
            res_dbl->set(j, sum);
        });
    } else if (auto* res_f32 = dynamic_cast<DenseVector<float>*>(&result)) {
        ParallelFor(0, n, [&](size_t j) {
            float sum = 0.0f;
            for (uint64_t i = 0; i < n; ++i) {
                sum += static_cast<float>(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_f32->set(j, sum);
        });
    } else if (auto* res_f16 = dynamic_cast<DenseVector<float16_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t j) {
            float sum = 0.0f;
            for (uint64_t i = 0; i < n; ++i) {
                sum += static_cast<float>(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_f16->set(j, float16_t(sum));
        });
    } else {
        throw std::runtime_error("Unsupported result type for vector_matrix_multiply");
    }
}

void CpuSolver::outer_product(const VectorBase& a, const VectorBase& b, MatrixBase& result) {
    if (a.size() != b.size()) throw std::invalid_argument("Vectors must be same size");
    uint64_t n = a.size();

    const DataType res_dt = result.get_data_type();
    if (res_dt == DataType::COMPLEX_FLOAT16) {
        auto* out = dynamic_cast<ComplexFloat16Matrix*>(&result);
        if (!out) {
            throw std::runtime_error("CpuSolver::outer_product complex_float16 result type mismatch");
        }
        debug_trace::set_last("cpu.outer_product.cf16");
        auto* rdst = out->real_data();
        auto* idst = out->imag_data();
        const bool t = out->is_transposed();
        ParallelFor(0, n, [&](size_t i) {
            const std::complex<double> va = a.get_element_as_complex(i);
            for (uint64_t j = 0; j < n; ++j) {
                const std::complex<double> vb = b.get_element_as_complex(j);
                const std::complex<double> vc = va * vb;
                const uint64_t idx = t ? (j * n + i) : (i * n + j);
                rdst[idx] = float16_t(vc.real());
                idst[idx] = float16_t(vc.imag());
            }
        });
        out->set_scalar(1.0);
        return;
    }
    if (res_dt == DataType::COMPLEX_FLOAT32) {
        auto* out = dynamic_cast<DenseMatrix<std::complex<float>>*>(&result);
        if (!out) {
            throw std::runtime_error("CpuSolver::outer_product complex_float32 result type mismatch");
        }
        debug_trace::set_last("cpu.outer_product.c32");
        ParallelFor(0, n, [&](size_t i) {
            const std::complex<double> va = a.get_element_as_complex(i);
            for (uint64_t j = 0; j < n; ++j) {
                const std::complex<double> vc = va * b.get_element_as_complex(j);
                out->set(i, j, std::complex<float>(static_cast<float>(vc.real()), static_cast<float>(vc.imag())));
            }
        });
        out->set_scalar(1.0);
        return;
    }
    if (res_dt == DataType::COMPLEX_FLOAT64) {
        auto* out = dynamic_cast<DenseMatrix<std::complex<double>>*>(&result);
        if (!out) {
            throw std::runtime_error("CpuSolver::outer_product complex_float64 result type mismatch");
        }
        debug_trace::set_last("cpu.outer_product.c64");
        ParallelFor(0, n, [&](size_t i) {
            const std::complex<double> va = a.get_element_as_complex(i);
            for (uint64_t j = 0; j < n; ++j) {
                out->set(i, j, va * b.get_element_as_complex(j));
            }
        });
        out->set_scalar(1.0);
        return;
    }

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
    } else if (auto* m = dynamic_cast<DenseMatrix<int64_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            int64_t val_a = static_cast<int64_t>(a.get_element_as_double(i));
            for (uint64_t j = 0; j < n; ++j) {
                m->set(i, j, val_a * static_cast<int64_t>(b.get_element_as_double(j)));
            }
        });
    } else if (auto* m = dynamic_cast<DenseMatrix<int16_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            int16_t val_a = static_cast<int16_t>(a.get_element_as_double(i));
            for (uint64_t j = 0; j < n; ++j) {
                m->set(i, j, static_cast<int16_t>(val_a * static_cast<int16_t>(b.get_element_as_double(j))));
            }
        });
    } else if (auto* m = dynamic_cast<DenseMatrix<int8_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            int8_t val_a = static_cast<int8_t>(a.get_element_as_double(i));
            for (uint64_t j = 0; j < n; ++j) {
                m->set(i, j, static_cast<int8_t>(val_a * static_cast<int8_t>(b.get_element_as_double(j))));
            }
        });
    } else if (auto* m = dynamic_cast<DenseMatrix<uint64_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            uint64_t val_a = static_cast<uint64_t>(a.get_element_as_double(i));
            for (uint64_t j = 0; j < n; ++j) {
                m->set(i, j, val_a * static_cast<uint64_t>(b.get_element_as_double(j)));
            }
        });
    } else if (auto* m = dynamic_cast<DenseMatrix<uint32_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            uint32_t val_a = static_cast<uint32_t>(a.get_element_as_double(i));
            for (uint64_t j = 0; j < n; ++j) {
                m->set(i, j, static_cast<uint32_t>(val_a * static_cast<uint32_t>(b.get_element_as_double(j))));
            }
        });
    } else if (auto* m = dynamic_cast<DenseMatrix<uint16_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            uint16_t val_a = static_cast<uint16_t>(a.get_element_as_double(i));
            for (uint64_t j = 0; j < n; ++j) {
                m->set(i, j, static_cast<uint16_t>(val_a * static_cast<uint16_t>(b.get_element_as_double(j))));
            }
        });
    } else if (auto* m = dynamic_cast<DenseMatrix<uint8_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            uint8_t val_a = static_cast<uint8_t>(a.get_element_as_double(i));
            for (uint64_t j = 0; j < n; ++j) {
                m->set(i, j, static_cast<uint8_t>(val_a * static_cast<uint8_t>(b.get_element_as_double(j))));
            }
        });
    } else if (auto* m = dynamic_cast<DenseMatrix<float>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            const float val_a = static_cast<float>(a.get_element_as_double(i));
            for (uint64_t j = 0; j < n; ++j) {
                m->set(i, j, static_cast<float>(val_a * static_cast<float>(b.get_element_as_double(j))));
            }
        });
        m->set_scalar(1.0);
    } else if (auto* m = dynamic_cast<DenseMatrix<float16_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            const float val_a = static_cast<float>(a.get_element_as_double(i));
            for (uint64_t j = 0; j < n; ++j) {
                const float v = val_a * static_cast<float>(b.get_element_as_double(j));
                m->set(i, j, float16_t(v));
            }
        });
        m->set_scalar(1.0);
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

    const DataType dtype = result.get_data_type();
    if (dtype == DataType::COMPLEX_FLOAT16) {
        auto* out = dynamic_cast<ComplexFloat16Vector*>(&result);
        if (!out) {
            throw std::runtime_error("CpuSolver::add_vector complex_float16 result type mismatch");
        }
        debug_trace::set_last("cpu.add_vector.cf16");
        auto* rdst = out->real_data();
        auto* idst = out->imag_data();
        ParallelFor(0, n, [&](size_t i) {
            const std::complex<double> v = a.get_element_as_complex(i) + b.get_element_as_complex(i);
            rdst[i] = float16_t(v.real());
            idst[i] = float16_t(v.imag());
        });
        out->set_scalar(1.0);
        return;
    }
    if (dtype == DataType::COMPLEX_FLOAT32) {
        auto* out = dynamic_cast<DenseVector<std::complex<float>>*>(&result);
        if (!out) {
            throw std::runtime_error("CpuSolver::add_vector complex_float32 result type mismatch");
        }
        debug_trace::set_last("cpu.add_vector.c32");
        auto* dst = out->data();
        ParallelFor(0, n, [&](size_t i) {
            const std::complex<double> v = a.get_element_as_complex(i) + b.get_element_as_complex(i);
            dst[i] = std::complex<float>(static_cast<float>(v.real()), static_cast<float>(v.imag()));
        });
        out->set_scalar(1.0);
        return;
    }
    if (dtype == DataType::COMPLEX_FLOAT64) {
        auto* out = dynamic_cast<DenseVector<std::complex<double>>*>(&result);
        if (!out) {
            throw std::runtime_error("CpuSolver::add_vector complex_float64 result type mismatch");
        }
        debug_trace::set_last("cpu.add_vector.c64");
        auto* dst = out->data();
        ParallelFor(0, n, [&](size_t i) {
            dst[i] = a.get_element_as_complex(i) + b.get_element_as_complex(i);
        });
        out->set_scalar(1.0);
        return;
    }

    if (dtype == DataType::FLOAT16) {
        auto* out = dynamic_cast<DenseVector<float16_t>*>(&result);
        if (!out) {
            throw std::runtime_error("CpuSolver::add_vector float16 result type mismatch");
        }
        debug_trace::set_last("cpu.add_vector.f16");
        auto* dst = out->data();
        ParallelFor(0, n, [&](size_t i) {
            dst[i] = float16_t(a.get_element_as_double(i) + b.get_element_as_double(i));
        });
        out->set_scalar(1.0);
        return;
    }
    if (dtype == DataType::FLOAT32) {
        auto* out = dynamic_cast<DenseVector<float>*>(&result);
        if (!out) {
            throw std::runtime_error("CpuSolver::add_vector float32 result type mismatch");
        }
        debug_trace::set_last("cpu.add_vector.f32");
        auto* dst = out->data();
        ParallelFor(0, n, [&](size_t i) {
            dst[i] = static_cast<float>(a.get_element_as_double(i) + b.get_element_as_double(i));
        });
        out->set_scalar(1.0);
        return;
    }
    
    auto* a_dbl = dynamic_cast<const DenseVector<double>*>(&a);
    auto* b_dbl = dynamic_cast<const DenseVector<double>*>(&b);
    auto* res_dbl = dynamic_cast<DenseVector<double>*>(&result);
    
    if (a_dbl && b_dbl && res_dbl) {
        const bool scalar_ok =
            (a.get_scalar() == std::complex<double>(1.0, 0.0)) &&
            (b.get_scalar() == std::complex<double>(1.0, 0.0)) &&
            (result.get_scalar() == std::complex<double>(1.0, 0.0));
        if (scalar_ok) {
            const double* a_data = a_dbl->data();
            const double* b_data = b_dbl->data();
            double* res_data = res_dbl->data();

            ParallelFor(0, n, [&](size_t i) {
                res_data[i] = a_data[i] + b_data[i];
            });
            return;
        }
    }

    auto* a_i16 = dynamic_cast<const DenseVector<int16_t>*>(&a);
    auto* b_i16 = dynamic_cast<const DenseVector<int16_t>*>(&b);
    auto* res_i16 = dynamic_cast<DenseVector<int16_t>*>(&result);
    if (a_i16 && b_i16 && res_i16) {
        const int16_t* a_data = a_i16->data();
        const int16_t* b_data = b_i16->data();
        int16_t* res_data = res_i16->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = static_cast<int16_t>(a_data[i] + b_data[i]);
        });
        return;
    }

    auto* a_i32 = dynamic_cast<const DenseVector<int32_t>*>(&a);
    auto* b_i32 = dynamic_cast<const DenseVector<int32_t>*>(&b);
    auto* res_i32 = dynamic_cast<DenseVector<int32_t>*>(&result);
    if (a_i32 && b_i32 && res_i32) {
        const int32_t* a_data = a_i32->data();
        const int32_t* b_data = b_i32->data();
        int32_t* res_data = res_i32->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = a_data[i] + b_data[i];
        });
        return;
    }

    auto* a_i8 = dynamic_cast<const DenseVector<int8_t>*>(&a);
    auto* b_i8 = dynamic_cast<const DenseVector<int8_t>*>(&b);
    auto* res_i8 = dynamic_cast<DenseVector<int8_t>*>(&result);
    if (a_i8 && b_i8 && res_i8) {
        const int8_t* a_data = a_i8->data();
        const int8_t* b_data = b_i8->data();
        int8_t* res_data = res_i8->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = static_cast<int8_t>(a_data[i] + b_data[i]);
        });
        return;
    }

    auto* a_i64 = dynamic_cast<const DenseVector<int64_t>*>(&a);
    auto* b_i64 = dynamic_cast<const DenseVector<int64_t>*>(&b);
    auto* res_i64 = dynamic_cast<DenseVector<int64_t>*>(&result);
    if (a_i64 && b_i64 && res_i64) {
        const int64_t* a_data = a_i64->data();
        const int64_t* b_data = b_i64->data();
        int64_t* res_data = res_i64->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = a_data[i] + b_data[i];
        });
        return;
    }

    auto* a_u8 = dynamic_cast<const DenseVector<uint8_t>*>(&a);
    auto* b_u8 = dynamic_cast<const DenseVector<uint8_t>*>(&b);
    auto* res_u8 = dynamic_cast<DenseVector<uint8_t>*>(&result);
    if (a_u8 && b_u8 && res_u8) {
        const uint8_t* a_data = a_u8->data();
        const uint8_t* b_data = b_u8->data();
        uint8_t* res_data = res_u8->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = static_cast<uint8_t>(a_data[i] + b_data[i]);
        });
        return;
    }

    auto* a_u16 = dynamic_cast<const DenseVector<uint16_t>*>(&a);
    auto* b_u16 = dynamic_cast<const DenseVector<uint16_t>*>(&b);
    auto* res_u16 = dynamic_cast<DenseVector<uint16_t>*>(&result);
    if (a_u16 && b_u16 && res_u16) {
        const uint16_t* a_data = a_u16->data();
        const uint16_t* b_data = b_u16->data();
        uint16_t* res_data = res_u16->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = static_cast<uint16_t>(a_data[i] + b_data[i]);
        });
        return;
    }

    auto* a_u32 = dynamic_cast<const DenseVector<uint32_t>*>(&a);
    auto* b_u32 = dynamic_cast<const DenseVector<uint32_t>*>(&b);
    auto* res_u32 = dynamic_cast<DenseVector<uint32_t>*>(&result);
    if (a_u32 && b_u32 && res_u32) {
        const uint32_t* a_data = a_u32->data();
        const uint32_t* b_data = b_u32->data();
        uint32_t* res_data = res_u32->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = a_data[i] + b_data[i];
        });
        return;
    }

    auto* a_u64 = dynamic_cast<const DenseVector<uint64_t>*>(&a);
    auto* b_u64 = dynamic_cast<const DenseVector<uint64_t>*>(&b);
    auto* res_u64 = dynamic_cast<DenseVector<uint64_t>*>(&result);
    if (a_u64 && b_u64 && res_u64) {
        const uint64_t* a_data = a_u64->data();
        const uint64_t* b_data = b_u64->data();
        uint64_t* res_data = res_u64->data();
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
                 return;
             }

             // Try int16
             auto* res_i16_fb = dynamic_cast<DenseVector<int16_t>*>(&result);
             if (res_i16_fb) {
                 res_i16_fb->set(i, static_cast<int16_t>(a.get_element_as_double(i) + b.get_element_as_double(i)));
                 return;
             }

             auto* res_i8_fb = dynamic_cast<DenseVector<int8_t>*>(&result);
             if (res_i8_fb) {
                 res_i8_fb->set(i, static_cast<int8_t>(a.get_element_as_double(i) + b.get_element_as_double(i)));
                 return;
             }

             auto* res_i64_fb = dynamic_cast<DenseVector<int64_t>*>(&result);
             if (res_i64_fb) {
                 res_i64_fb->set(i, static_cast<int64_t>(a.get_element_as_double(i) + b.get_element_as_double(i)));
                 return;
             }

             auto* res_u8_fb = dynamic_cast<DenseVector<uint8_t>*>(&result);
             if (res_u8_fb) {
                 res_u8_fb->set(i, static_cast<uint8_t>(a.get_element_as_double(i) + b.get_element_as_double(i)));
                 return;
             }

             auto* res_u16_fb = dynamic_cast<DenseVector<uint16_t>*>(&result);
             if (res_u16_fb) {
                 res_u16_fb->set(i, static_cast<uint16_t>(a.get_element_as_double(i) + b.get_element_as_double(i)));
                 return;
             }

             auto* res_u32_fb = dynamic_cast<DenseVector<uint32_t>*>(&result);
             if (res_u32_fb) {
                 res_u32_fb->set(i, static_cast<uint32_t>(a.get_element_as_double(i) + b.get_element_as_double(i)));
                 return;
             }

             auto* res_u64_fb = dynamic_cast<DenseVector<uint64_t>*>(&result);
             if (res_u64_fb) {
                 res_u64_fb->set(i, static_cast<uint64_t>(a.get_element_as_double(i) + b.get_element_as_double(i)));
                 return;
             }
        }
    });
}

void CpuSolver::subtract_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) {
    uint64_t n = a.size();

    const DataType dtype = result.get_data_type();
    if (dtype == DataType::COMPLEX_FLOAT16) {
        auto* out = dynamic_cast<ComplexFloat16Vector*>(&result);
        if (!out) {
            throw std::runtime_error("CpuSolver::subtract_vector complex_float16 result type mismatch");
        }
        debug_trace::set_last("cpu.subtract_vector.cf16");
        auto* rdst = out->real_data();
        auto* idst = out->imag_data();
        ParallelFor(0, n, [&](size_t i) {
            const std::complex<double> v = a.get_element_as_complex(i) - b.get_element_as_complex(i);
            rdst[i] = float16_t(v.real());
            idst[i] = float16_t(v.imag());
        });
        out->set_scalar(1.0);
        return;
    }
    if (dtype == DataType::COMPLEX_FLOAT32) {
        auto* out = dynamic_cast<DenseVector<std::complex<float>>*>(&result);
        if (!out) {
            throw std::runtime_error("CpuSolver::subtract_vector complex_float32 result type mismatch");
        }
        debug_trace::set_last("cpu.subtract_vector.c32");
        auto* dst = out->data();
        ParallelFor(0, n, [&](size_t i) {
            const std::complex<double> v = a.get_element_as_complex(i) - b.get_element_as_complex(i);
            dst[i] = std::complex<float>(static_cast<float>(v.real()), static_cast<float>(v.imag()));
        });
        out->set_scalar(1.0);
        return;
    }
    if (dtype == DataType::COMPLEX_FLOAT64) {
        auto* out = dynamic_cast<DenseVector<std::complex<double>>*>(&result);
        if (!out) {
            throw std::runtime_error("CpuSolver::subtract_vector complex_float64 result type mismatch");
        }
        debug_trace::set_last("cpu.subtract_vector.c64");
        auto* dst = out->data();
        ParallelFor(0, n, [&](size_t i) {
            dst[i] = a.get_element_as_complex(i) - b.get_element_as_complex(i);
        });
        out->set_scalar(1.0);
        return;
    }

    if (dtype == DataType::FLOAT16) {
        auto* out = dynamic_cast<DenseVector<float16_t>*>(&result);
        if (!out) {
            throw std::runtime_error("CpuSolver::subtract_vector float16 result type mismatch");
        }
        debug_trace::set_last("cpu.subtract_vector.f16");
        auto* dst = out->data();
        ParallelFor(0, n, [&](size_t i) {
            dst[i] = float16_t(a.get_element_as_double(i) - b.get_element_as_double(i));
        });
        out->set_scalar(1.0);
        return;
    }
    if (dtype == DataType::FLOAT32) {
        auto* out = dynamic_cast<DenseVector<float>*>(&result);
        if (!out) {
            throw std::runtime_error("CpuSolver::subtract_vector float32 result type mismatch");
        }
        debug_trace::set_last("cpu.subtract_vector.f32");
        auto* dst = out->data();
        ParallelFor(0, n, [&](size_t i) {
            dst[i] = static_cast<float>(a.get_element_as_double(i) - b.get_element_as_double(i));
        });
        out->set_scalar(1.0);
        return;
    }
    
    auto* a_dbl = dynamic_cast<const DenseVector<double>*>(&a);
    auto* b_dbl = dynamic_cast<const DenseVector<double>*>(&b);
    auto* res_dbl = dynamic_cast<DenseVector<double>*>(&result);
    
    if (a_dbl && b_dbl && res_dbl) {
        const bool scalar_ok =
            (a.get_scalar() == std::complex<double>(1.0, 0.0)) &&
            (b.get_scalar() == std::complex<double>(1.0, 0.0)) &&
            (result.get_scalar() == std::complex<double>(1.0, 0.0));
        if (scalar_ok) {
            const double* a_data = a_dbl->data();
            const double* b_data = b_dbl->data();
            double* res_data = res_dbl->data();

            ParallelFor(0, n, [&](size_t i) {
                res_data[i] = a_data[i] - b_data[i];
            });
            return;
        }
    }

    auto* a_i16 = dynamic_cast<const DenseVector<int16_t>*>(&a);
    auto* b_i16 = dynamic_cast<const DenseVector<int16_t>*>(&b);
    auto* res_i16 = dynamic_cast<DenseVector<int16_t>*>(&result);
    if (a_i16 && b_i16 && res_i16) {
        const int16_t* a_data = a_i16->data();
        const int16_t* b_data = b_i16->data();
        int16_t* res_data = res_i16->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = static_cast<int16_t>(a_data[i] - b_data[i]);
        });
        return;
    }

    auto* a_i32 = dynamic_cast<const DenseVector<int32_t>*>(&a);
    auto* b_i32 = dynamic_cast<const DenseVector<int32_t>*>(&b);
    auto* res_i32 = dynamic_cast<DenseVector<int32_t>*>(&result);
    if (a_i32 && b_i32 && res_i32) {
        const int32_t* a_data = a_i32->data();
        const int32_t* b_data = b_i32->data();
        int32_t* res_data = res_i32->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = a_data[i] - b_data[i];
        });
        return;
    }

    auto* a_i8 = dynamic_cast<const DenseVector<int8_t>*>(&a);
    auto* b_i8 = dynamic_cast<const DenseVector<int8_t>*>(&b);
    auto* res_i8 = dynamic_cast<DenseVector<int8_t>*>(&result);
    if (a_i8 && b_i8 && res_i8) {
        const int8_t* a_data = a_i8->data();
        const int8_t* b_data = b_i8->data();
        int8_t* res_data = res_i8->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = static_cast<int8_t>(a_data[i] - b_data[i]);
        });
        return;
    }

    auto* a_i64 = dynamic_cast<const DenseVector<int64_t>*>(&a);
    auto* b_i64 = dynamic_cast<const DenseVector<int64_t>*>(&b);
    auto* res_i64 = dynamic_cast<DenseVector<int64_t>*>(&result);
    if (a_i64 && b_i64 && res_i64) {
        const int64_t* a_data = a_i64->data();
        const int64_t* b_data = b_i64->data();
        int64_t* res_data = res_i64->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = a_data[i] - b_data[i];
        });
        return;
    }

    auto* a_u8 = dynamic_cast<const DenseVector<uint8_t>*>(&a);
    auto* b_u8 = dynamic_cast<const DenseVector<uint8_t>*>(&b);
    auto* res_u8 = dynamic_cast<DenseVector<uint8_t>*>(&result);
    if (a_u8 && b_u8 && res_u8) {
        const uint8_t* a_data = a_u8->data();
        const uint8_t* b_data = b_u8->data();
        uint8_t* res_data = res_u8->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = static_cast<uint8_t>(a_data[i] - b_data[i]);
        });
        return;
    }

    auto* a_u16 = dynamic_cast<const DenseVector<uint16_t>*>(&a);
    auto* b_u16 = dynamic_cast<const DenseVector<uint16_t>*>(&b);
    auto* res_u16 = dynamic_cast<DenseVector<uint16_t>*>(&result);
    if (a_u16 && b_u16 && res_u16) {
        const uint16_t* a_data = a_u16->data();
        const uint16_t* b_data = b_u16->data();
        uint16_t* res_data = res_u16->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = static_cast<uint16_t>(a_data[i] - b_data[i]);
        });
        return;
    }

    auto* a_u32 = dynamic_cast<const DenseVector<uint32_t>*>(&a);
    auto* b_u32 = dynamic_cast<const DenseVector<uint32_t>*>(&b);
    auto* res_u32 = dynamic_cast<DenseVector<uint32_t>*>(&result);
    if (a_u32 && b_u32 && res_u32) {
        const uint32_t* a_data = a_u32->data();
        const uint32_t* b_data = b_u32->data();
        uint32_t* res_data = res_u32->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = a_data[i] - b_data[i];
        });
        return;
    }

    auto* a_u64 = dynamic_cast<const DenseVector<uint64_t>*>(&a);
    auto* b_u64 = dynamic_cast<const DenseVector<uint64_t>*>(&b);
    auto* res_u64 = dynamic_cast<DenseVector<uint64_t>*>(&result);
    if (a_u64 && b_u64 && res_u64) {
        const uint64_t* a_data = a_u64->data();
        const uint64_t* b_data = b_u64->data();
        uint64_t* res_data = res_u64->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = a_data[i] - b_data[i];
        });
        return;
    }
    // Fallback
    ParallelFor(0, n, [&](size_t i) {
        if (res_dbl) {
            res_dbl->set(i, a.get_element_as_double(i) - b.get_element_as_double(i));
            return;
        }

        // Try int32
        auto* res_int = dynamic_cast<DenseVector<int32_t>*>(&result);
        if (res_int) {
            res_int->set(i, static_cast<int32_t>(a.get_element_as_double(i) - b.get_element_as_double(i)));
            return;
        }

        auto* res_i16_fb = dynamic_cast<DenseVector<int16_t>*>(&result);
        if (res_i16_fb) {
            res_i16_fb->set(i, static_cast<int16_t>(a.get_element_as_double(i) - b.get_element_as_double(i)));
            return;
        }

        auto* res_i8_fb = dynamic_cast<DenseVector<int8_t>*>(&result);
        if (res_i8_fb) {
            res_i8_fb->set(i, static_cast<int8_t>(a.get_element_as_double(i) - b.get_element_as_double(i)));
            return;
        }

        auto* res_i64_fb = dynamic_cast<DenseVector<int64_t>*>(&result);
        if (res_i64_fb) {
            res_i64_fb->set(i, static_cast<int64_t>(a.get_element_as_double(i) - b.get_element_as_double(i)));
            return;
        }

        auto* res_u8_fb = dynamic_cast<DenseVector<uint8_t>*>(&result);
        if (res_u8_fb) {
            res_u8_fb->set(i, static_cast<uint8_t>(a.get_element_as_double(i) - b.get_element_as_double(i)));
            return;
        }

        auto* res_u16_fb = dynamic_cast<DenseVector<uint16_t>*>(&result);
        if (res_u16_fb) {
            res_u16_fb->set(i, static_cast<uint16_t>(a.get_element_as_double(i) - b.get_element_as_double(i)));
            return;
        }

        auto* res_u32_fb = dynamic_cast<DenseVector<uint32_t>*>(&result);
        if (res_u32_fb) {
            res_u32_fb->set(i, static_cast<uint32_t>(a.get_element_as_double(i) - b.get_element_as_double(i)));
            return;
        }

        auto* res_u64_fb = dynamic_cast<DenseVector<uint64_t>*>(&result);
        if (res_u64_fb) {
            res_u64_fb->set(i, static_cast<uint64_t>(a.get_element_as_double(i) - b.get_element_as_double(i)));
            return;
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

    auto* a_i16 = dynamic_cast<const DenseVector<int16_t>*>(&a);
    auto* res_i16 = dynamic_cast<DenseVector<int16_t>*>(&result);
    if (a_i16 && res_i16) {
        const int16_t* a_data = a_i16->data();
        int16_t* res_data = res_i16->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = static_cast<int16_t>(static_cast<double>(a_data[i]) * scalar);
        });
        return;
    }
    
    // Fallback
    ParallelFor(0, n, [&](size_t i) {
        if (res_dbl) {
            res_dbl->set(i, a.get_element_as_double(i) * scalar);
            return;
        }

        // Try int32
        auto* res_int = dynamic_cast<DenseVector<int32_t>*>(&result);
        if (res_int) {
            res_int->set(i, static_cast<int32_t>(a.get_element_as_double(i) * scalar));
            return;
        }

        auto* res_i16_fb = dynamic_cast<DenseVector<int16_t>*>(&result);
        if (res_i16_fb) {
            res_i16_fb->set(i, static_cast<int16_t>(a.get_element_as_double(i) * scalar));
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

    auto* a_i16 = dynamic_cast<const DenseVector<int16_t>*>(&a);
    auto* res_i16 = dynamic_cast<DenseVector<int16_t>*>(&result);
    if (a_i16 && res_i16) {
        const int16_t* a_data = a_i16->data();
        int16_t* res_data = res_i16->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = static_cast<int16_t>(static_cast<double>(a_data[i]) + scalar);
        });
        return;
    }
    
    // Fallback
    ParallelFor(0, n, [&](size_t i) {
        if (res_dbl) {
            res_dbl->set(i, a.get_element_as_double(i) + scalar);
            return;
        }

        // Try int32
        auto* res_int = dynamic_cast<DenseVector<int32_t>*>(&result);
        if (res_int) {
            res_int->set(i, static_cast<int32_t>(a.get_element_as_double(i) + scalar));
            return;
        }

        auto* res_i16_fb = dynamic_cast<DenseVector<int16_t>*>(&result);
        if (res_i16_fb) {
            res_i16_fb->set(i, static_cast<int16_t>(a.get_element_as_double(i) + scalar));
        }
    });
}

} // namespace pycauset

