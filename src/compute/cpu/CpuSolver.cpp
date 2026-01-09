#include "pycauset/compute/cpu/CpuSolver.hpp"
#include <complex>
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
// Force 64-bit integers for LAPACK to match the DLL expectation (likely ILP64)
// or at least to prevent buffer overflow if the DLL writes 64-bit integers.
// #define LAPACK_ILP64

#ifdef _WIN32
#include <cblas.h>
#include <lapacke.h>
#elif defined(__APPLE__)
// On macOS, we can use Accelerate (native) or OpenBLAS (brew). 
// Our CMake prefers OpenBLAS now.
// If OpenBLAS is used, we need its specific headers.
// However, the include path might be <openblas/cblas.h> or just <cblas.h>
#include <cblas.h>
// Check if lapacke.h exists or implicit
#if __has_include(<lapacke.h>)
#include <lapacke.h>
#else
// Fallback for Accelerate or weird paths
#include <openblas/lapacke.h> 
#endif
#else
// Linux
#include <cblas.h>
#include <lapacke.h>
#endif
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/core/Float16.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/ComplexFloat16Matrix.hpp"
#include "pycauset/vector/DenseVector.hpp"
#include "pycauset/vector/ComplexFloat16Vector.hpp"
#include "pycauset/vector/UnitVector.hpp"
#include "pycauset/matrix/TriangularBitMatrix.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/matrix/SymmetricMatrix.hpp"
#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/matrix/IdentityMatrix.hpp"
#include "pycauset/math/Eigen.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/core/MemoryHints.hpp"
#include "pycauset/core/MemoryGovernor.hpp"
#include "pycauset/core/DebugTrace.hpp"
#include "pycauset/core/Nvtx.hpp"
#include <Eigen/Dense>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <bit>
#include <filesystem>
#include <vector>
#include <cstring>
#include <iostream>
#include <limits>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#ifdef _WIN32
#include <intrin.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

namespace pycauset {

namespace {
    // Runtime CPU detection for AVX-512 VPOPCNTDQ
    bool has_avx512_vpopcntdq() {
#if defined(__x86_64__) || defined(_M_X64)
        static bool checked = false;
        static bool available = false;
        if (checked) return available;

        // Check for AVX-512 Foundation (bit 16 of EBX in leaf 7, subleaf 0)
        // Check for VPOPCNTDQ (bit 14 of ECX in leaf 7, subleaf 0)
        
        int cpuInfo[4];
#ifdef _WIN32
        __cpuid(cpuInfo, 0);
        int nIds = cpuInfo[0];
        if (nIds >= 7) {
            __cpuidex(cpuInfo, 7, 0);
            bool avx512f = (cpuInfo[1] & (1 << 16)) != 0;
            bool avx512vpopcntdq = (cpuInfo[2] & (1 << 14)) != 0;
            available = avx512f && avx512vpopcntdq;
        }
#else
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid_max(0, &eax) >= 7) {
            __cpuid_count(7, 0, eax, ebx, ecx, edx);
            bool avx512f = (ebx & (1 << 16)) != 0;
            bool avx512vpopcntdq = (ecx & (1 << 14)) != 0;
            available = avx512f && avx512vpopcntdq;
        }
#endif
        checked = true;
        return available;
#else
        return false;
#endif
    }

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64))
    __attribute__((target("avx512f,avx512vpopcntdq")))
#endif
#if defined(__x86_64__) || defined(_M_X64)
    int64_t dot_product_avx512(const uint64_t* a_row, const uint64_t* b_col, uint64_t count) {
        __m512i vsum = _mm512_setzero_si512();
        for (uint64_t w = 0; w < count; w += 8) {
            __m512i va = _mm512_loadu_si512((const void*)&a_row[w]);
            __m512i vb = _mm512_loadu_si512((const void*)&b_col[w]);
            __m512i vand = _mm512_and_si512(va, vb);
            __m512i vpop = _mm512_popcnt_epi64(vand);
            vsum = _mm512_add_epi64(vsum, vpop);
        }
        return _mm512_reduce_add_epi64(vsum);
    }
#endif
}

namespace {
std::vector<double> to_memory_flat_real_square(const MatrixBase& m) {
    const uint64_t rows = m.rows();
    const uint64_t cols = m.cols();
    if (rows != cols) {
        throw std::runtime_error("Operation requires square matrix");
    }

    const uint64_t n = rows;
    std::vector<double> mat(n * n);
    ParallelFor(0, n, [&](size_t i) {
        for (size_t j = 0; j < n; ++j) {
            mat[i * n + j] = m.get_element_as_double(static_cast<uint64_t>(i), static_cast<uint64_t>(j));
        }
    });
    return mat;
}

std::vector<std::complex<double>> to_memory_flat_complex_square(const MatrixBase& m) {
    const uint64_t rows = m.rows();
    const uint64_t cols = m.cols();
    if (rows != cols) {
        throw std::runtime_error("Operation requires square matrix");
    }

    const uint64_t n = rows;
    std::vector<std::complex<double>> mat(n * n);
    ParallelFor(0, n, [&](size_t i) {
        for (size_t j = 0; j < n; ++j) {
            mat[i * n + j] = m.get_element_as_complex(static_cast<uint64_t>(i), static_cast<uint64_t>(j));
        }
    });
    return mat;
}
} // namespace

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

    inline bool is_broadcast_compatible_dim(uint64_t in_dim, uint64_t out_dim) {
        return (in_dim == out_dim) || (in_dim == 1);
    }

    inline uint64_t broadcast_index(uint64_t out_index, uint64_t in_dim) {
        return (in_dim == 1) ? 0 : out_index;
    }

    template <typename Op>
    void binary_op_complex16_impl(const MatrixBase& a, const MatrixBase& b, ComplexFloat16Matrix& out, Op op) {
        const uint64_t rows = out.rows();
        const uint64_t cols = out.cols();
        if (!is_broadcast_compatible_dim(a.rows(), rows) ||
            !is_broadcast_compatible_dim(a.cols(), cols) ||
            !is_broadcast_compatible_dim(b.rows(), rows) ||
            !is_broadcast_compatible_dim(b.cols(), cols)) {
            throw std::invalid_argument("operands could not be broadcast together");
        }

        const bool t_c = out.is_transposed();
        const uint64_t storage_cols = out.base_cols();
        float16_t* rdst = out.real_data();
        float16_t* idst = out.imag_data();

        ParallelFor(0, rows, [&](size_t i) {
            for (size_t j = 0; j < cols; ++j) {
                const uint64_t ia = broadcast_index(static_cast<uint64_t>(i), a.rows());
                const uint64_t ja = broadcast_index(static_cast<uint64_t>(j), a.cols());
                const uint64_t ib = broadcast_index(static_cast<uint64_t>(i), b.rows());
                const uint64_t jb = broadcast_index(static_cast<uint64_t>(j), b.cols());
                const std::complex<double> va = a.get_element_as_complex(ia, ja);
                const std::complex<double> vb = b.get_element_as_complex(ib, jb);
                const std::complex<double> vc = op(va, vb);
                const uint64_t idx = t_c ? (static_cast<uint64_t>(j) * storage_cols + static_cast<uint64_t>(i))
                                        : (static_cast<uint64_t>(i) * storage_cols + static_cast<uint64_t>(j));
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
        NVTX_RANGE("CpuSolver::attempt_direct_path");
        // Only supported for Float/Double where we have BLAS
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
            const uint64_t a_elems = a_dense->base_rows() * a_dense->base_cols();
            const uint64_t b_elems = b_dense->base_rows() * b_dense->base_cols();
            const uint64_t c_elems = c_dense->base_rows() * c_dense->base_cols();
            const size_t total_bytes = static_cast<size_t>(a_elems + b_elems + c_elems) * sizeof(T);
            
            auto& governor = pycauset::core::MemoryGovernor::instance();
            
            // Check 1: Does it fit in RAM? (Anti-Nanny Logic)
            // Check 2: Do we have enough Pinned Memory Budget?
            if (governor.should_use_direct_path(total_bytes) && governor.try_pin_memory(total_bytes)) {
                // Attempt to pin all three matrices
                bool pinned = true;
                // We use const_cast because pinning is a logical const operation (doesn't change data)
                // but might change internal OS handles.
                pinned &= const_cast<DenseMatrix<T>*>(a_dense)->pin_range(0, a_elems);
                pinned &= const_cast<DenseMatrix<T>*>(b_dense)->pin_range(0, b_elems);
                pinned &= c_dense->pin_range(0, c_elems);
                
                if (pinned) {
                    const T* a_data = a_dense->data();
                    const T* b_data = b_dense->data();
                    T* c_data = c_dense->data();
                    
                    bool t_a = a_dense->is_transposed();
                    bool t_b = b_dense->is_transposed();

                    const int M = static_cast<int>(a_dense->rows());
                    const int N = static_cast<int>(b_dense->cols());
                    const int K = static_cast<int>(a_dense->cols());
                    const int lda = static_cast<int>(a_dense->base_cols());
                    const int ldb = static_cast<int>(b_dense->base_cols());
                    const int ldc = static_cast<int>(c_dense->base_cols());
                    
                    if constexpr (std::is_same_v<T, double>) {
                        cblas_dgemm(
                            CblasRowMajor,
                            t_a ? CblasTrans : CblasNoTrans,
                            t_b ? CblasTrans : CblasNoTrans,
                            M, N, K,
                            1.0, a_data, lda,
                            b_data, ldb,
                            0.0, c_data, ldc
                        );
                    } else {
                        cblas_sgemm(
                            CblasRowMajor,
                            t_a ? CblasTrans : CblasNoTrans,
                            t_b ? CblasTrans : CblasNoTrans,
                            M, N, K,
                            1.0f, a_data, lda,
                            b_data, ldb,
                            0.0f, c_data, ldc
                        );
                    }
                    
                    // Cleanup
                    // Unpinning is important to release the "locked" status
                    const_cast<DenseMatrix<T>*>(a_dense)->unpin_range(0, a_elems);
                    const_cast<DenseMatrix<T>*>(b_dense)->unpin_range(0, b_elems);
                    c_dense->unpin_range(0, c_elems);
                    
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
        NVTX_RANGE("CpuSolver::matmul_impl");
        const uint64_t m = a_dense->rows();
        const uint64_t k = a_dense->cols();
        if (b_dense->rows() != k) {
            throw std::invalid_argument("Matrix dimensions must match");
        }
        const uint64_t n = b_dense->cols();
        if (c_dense->rows() != m || c_dense->cols() != n) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        // Try Direct Path first (Generalized for Float/Double)
        if (attempt_direct_path(a_dense, b_dense, c_dense)) {
            return;
        }

        const T* a_data = a_dense->data();
        const T* b_data = b_dense->data();
        T* c_data = c_dense->data();

        const bool t_a = a_dense->is_transposed();
        const bool t_b = b_dense->is_transposed();
        const bool t_c = c_dense->is_transposed();

        const uint64_t lda = a_dense->base_cols();
        const uint64_t ldb = b_dense->base_cols();
        const uint64_t ldc = c_dense->base_cols();

        auto idx_a = [&](uint64_t i, uint64_t kk) -> uint64_t {
            return t_a ? (kk * lda + i) : (i * lda + kk);
        };
        auto idx_b = [&](uint64_t kk, uint64_t j) -> uint64_t {
            return t_b ? (j * ldb + kk) : (kk * ldb + j);
        };
        auto idx_c = [&](uint64_t i, uint64_t j) -> uint64_t {
            return t_c ? (j * ldc + i) : (i * ldc + j);
        };

        // --- Integer path: int64 accumulator + overflow checks ---
        // Policy: never silently wrap, never silently widen output storage.
        // We accumulate in int64 to avoid intermediate overflow, then range-check into T.
        if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
            size_t block_size = 64;

            ParallelBlockMap(m, n, block_size, [&](size_t i_start, size_t i_end, size_t j_start, size_t j_end) {
                std::vector<int64_t> sums(j_end - j_start);

                for (size_t i = i_start; i < i_end; ++i) {
                    for (size_t jj = 0; jj < sums.size(); ++jj) {
                        sums[jj] = 0;
                    }
                    for (size_t k_start = 0; k_start < k; k_start += block_size) {
                        const size_t k_end = std::min(k_start + block_size, static_cast<size_t>(k));
                        for (size_t kk = k_start; kk < k_end; ++kk) {
                            const int64_t val_a = static_cast<int64_t>(a_data[idx_a(static_cast<uint64_t>(i), static_cast<uint64_t>(kk))]);
                            if (val_a == 0) continue;
                            for (size_t jj = 0; jj < sums.size(); ++jj) {
                                const uint64_t j = static_cast<uint64_t>(j_start + jj);
                                const int64_t val_b = static_cast<int64_t>(b_data[idx_b(static_cast<uint64_t>(kk), j)]);
                                const int64_t term = val_a * val_b;
                                sums[jj] = checked_add_int64(sums[jj], term);
                            }
                        }
                    }

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
                        c_data[idx_c(static_cast<uint64_t>(i), static_cast<uint64_t>(j_start + jj))] = static_cast<T>(s);
                    }
                }
            });

            c_dense->set_scalar(a_dense->get_scalar() * b_dense->get_scalar());
            return;
        }

        // --- Optimization: Use OpenBLAS for double/float (Fallback if Direct Path failed) ---
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
            // If C is a transposed view, fall back to scalar loops.
            if (!t_c) {
                const int M = static_cast<int>(m);
                const int N = static_cast<int>(n);
                const int K = static_cast<int>(k);
                const int lda_i = static_cast<int>(lda);
                const int ldb_i = static_cast<int>(ldb);
                const int ldc_i = static_cast<int>(ldc);
                if constexpr (std::is_same_v<T, double>) {
                    cblas_dgemm(
                        CblasRowMajor,
                        t_a ? CblasTrans : CblasNoTrans,
                        t_b ? CblasTrans : CblasNoTrans,
                        M, N, K,
                        1.0,
                        a_data, lda_i,
                        b_data, ldb_i,
                        0.0,
                        c_data, ldc_i);
                } else {
                    cblas_sgemm(
                        CblasRowMajor,
                        t_a ? CblasTrans : CblasNoTrans,
                        t_b ? CblasTrans : CblasNoTrans,
                        M, N, K,
                        1.0f,
                        a_data, lda_i,
                        b_data, ldb_i,
                        0.0f,
                        c_data, ldc_i);
                }
                c_dense->set_scalar(a_dense->get_scalar() * b_dense->get_scalar());
                return;
            }
        }
        
        // Initialize result to 0
        for (uint64_t i = 0; i < m; ++i) {
            for (uint64_t j = 0; j < n; ++j) {
                c_data[idx_c(i, j)] = static_cast<T>(0);
            }
        }
        
        // --- Lookahead Protocol: Send Memory Hints ---
        using namespace pycauset::core;
        size_t total_bytes = m * n * sizeof(T);
        size_t stride_bytes = n * sizeof(T);
        size_t block_bytes = sizeof(T);
        size_t block_size = 64;

        ParallelBlockMap(m, n, block_size, [&](size_t i_start, size_t i_end, size_t j_start, size_t j_end) {
            for (size_t k_start = 0; k_start < k; k_start += block_size) {
                const size_t k_end = std::min(k_start + block_size, static_cast<size_t>(k));
                for (size_t i = i_start; i < i_end; ++i) {
                    for (size_t kk = k_start; kk < k_end; ++kk) {
                        const T val_a = a_data[idx_a(static_cast<uint64_t>(i), static_cast<uint64_t>(kk))];
                        if (val_a == static_cast<T>(0)) continue;
                        for (size_t j = j_start; j < j_end; ++j) {
                            const uint64_t jj = static_cast<uint64_t>(j);
                            c_data[idx_c(static_cast<uint64_t>(i), jj)] +=
                                val_a * b_data[idx_b(static_cast<uint64_t>(kk), jj)];
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
        const uint64_t n = in_dense->rows();
        if (in_dense->cols() != n) {
            throw std::invalid_argument("inverse_direct requires square matrix");
        }
        if (out_dense->rows() != n || out_dense->cols() != n) {
            throw std::invalid_argument("Output matrix size mismatch");
        }
        
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
        const uint64_t n = in_dense->rows();
        if (in_dense->cols() != n) throw std::invalid_argument("inverse_impl requires square matrix");
        if (out_dense->rows() != n || out_dense->cols() != n) throw std::invalid_argument("Output matrix size mismatch");
        if (in_dense->get_scalar() == 0.0) throw std::runtime_error("Matrix scalar is 0, cannot invert");

        // --- Optimization: Direct Path (LAPACK) ---
        // If it fits in RAM, use LAPACK. It's orders of magnitude faster than manual Gauss-Jordan.
        // We need space for Input (already there) + Output (already there) + Pivot Array (negligible).
        // So we just check if Output fits in RAM (Input is likely already in RAM or mapped).
        // Actually, we need to ensure we don't cause thrashing.
        // Total footprint ~ 2 * N^2 * sizeof(T).
        size_t total_bytes = 2 * n * n * sizeof(T);
        
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
            if (pycauset::core::MemoryGovernor::instance().should_use_direct_path(total_bytes)) {
                inverse_direct(in_dense, out_dense);
                return;
            }
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
        const uint64_t m = a.rows();
        const uint64_t k = a.cols();
        const uint64_t n = b.cols();
        if (b.rows() != k || result.rows() != m || result.cols() != n) {
            throw std::invalid_argument("Dimension mismatch");
        }

        const bool t_c = c_f16->is_transposed();
        const uint64_t storage_cols = c_f16->base_cols();
        float16_t* c_data = c_f16->data();
        ParallelFor(0, m, [&](size_t i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t kk = 0; kk < k; ++kk) {
                    sum += static_cast<float>(a.get_element_as_double(static_cast<uint64_t>(i), static_cast<uint64_t>(kk))) *
                           static_cast<float>(b.get_element_as_double(static_cast<uint64_t>(kk), static_cast<uint64_t>(j)));
                }
                const float16_t out(sum);
                if (t_c) {
                    c_data[static_cast<uint64_t>(j) * storage_cols + static_cast<uint64_t>(i)] = out;
                } else {
                    c_data[static_cast<uint64_t>(i) * storage_cols + static_cast<uint64_t>(j)] = out;
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
        const uint64_t m = a.rows();
        const uint64_t k = a.cols();
        const uint64_t n = b.cols();
        if (b.rows() != k || result.rows() != m || result.cols() != n) {
            throw std::invalid_argument("Dimension mismatch");
        }

        float* c_data = c_f32->data();
        const uint64_t ldc = c_f32->base_cols();
        const bool t_c = c_f32->is_transposed();
        ParallelFor(0, m, [&](size_t i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t kk = 0; kk < k; ++kk) {
                    sum += static_cast<float>(a.get_element_as_double(static_cast<uint64_t>(i), static_cast<uint64_t>(kk))) *
                           static_cast<float>(b.get_element_as_double(static_cast<uint64_t>(kk), static_cast<uint64_t>(j)));
                }
                const uint64_t idx = t_c ? (static_cast<uint64_t>(j) * ldc + static_cast<uint64_t>(i))
                                        : (static_cast<uint64_t>(i) * ldc + static_cast<uint64_t>(j));
                c_data[idx] = sum;
            }
        });
        return;
    }

    // 2e. ComplexFloat16 result (compute in complex<double>, store float16 planes)
    if (auto* c_cf16 = dynamic_cast<ComplexFloat16Matrix*>(&result)) {
        debug_trace::set_last("cpu.matmul.cf16_fallback");
        const uint64_t m = a.rows();
        const uint64_t k = a.cols();
        const uint64_t n = b.cols();
        if (b.rows() != k || result.rows() != m || result.cols() != n) {
            throw std::invalid_argument("Dimension mismatch");
        }

        const bool t_c = c_cf16->is_transposed();
        const uint64_t storage_cols = c_cf16->base_cols();
        float16_t* rdst = c_cf16->real_data();
        float16_t* idst = c_cf16->imag_data();

        ParallelFor(0, m, [&](size_t i) {
            for (size_t j = 0; j < n; ++j) {
                std::complex<double> sum = 0.0;
                for (size_t kk = 0; kk < k; ++kk) {
                    sum += a.get_element_as_complex(static_cast<uint64_t>(i), static_cast<uint64_t>(kk)) *
                           b.get_element_as_complex(static_cast<uint64_t>(kk), static_cast<uint64_t>(j));
                }
                const uint64_t idx = t_c ? (static_cast<uint64_t>(j) * storage_cols + static_cast<uint64_t>(i))
                                        : (static_cast<uint64_t>(i) * storage_cols + static_cast<uint64_t>(j));
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
        const uint64_t m = a.rows();
        const uint64_t k = a.cols();
        const uint64_t n = b.cols();
        if (b.rows() != k || result.rows() != m || result.cols() != n) {
            throw std::invalid_argument("Dimension mismatch");
        }

        const bool t_a = a.is_transposed();
        const bool t_b = b.is_transposed();
        const bool t_c = c_i32->is_transposed();

        const uint64_t lda = a_i32 ? a_i32->base_cols() : (a_i16 ? a_i16->base_cols() : a.cols());
        const uint64_t ldb = b_i32 ? b_i32->base_cols() : (b_i16 ? b_i16->base_cols() : b.cols());
        const uint64_t ldc = c_i32->base_cols();

        int32_t* c_data = c_i32->data();

        ParallelFor(0, m, [&](size_t i) {
            for (size_t j = 0; j < n; ++j) {
                int64_t sum = 0;
                for (size_t kk = 0; kk < k; ++kk) {
                    int64_t va = 0;
                    int64_t vb = 0;

                    if (a_i32) {
                        const int32_t* a_data = a_i32->data();
                        const uint64_t idx = t_a ? (static_cast<uint64_t>(kk) * lda + static_cast<uint64_t>(i))
                                                 : (static_cast<uint64_t>(i) * lda + static_cast<uint64_t>(kk));
                        va = static_cast<int64_t>(a_data[idx]);
                    } else if (a_i16) {
                        const int16_t* a_data = a_i16->data();
                        const uint64_t idx = t_a ? (static_cast<uint64_t>(kk) * lda + static_cast<uint64_t>(i))
                                                 : (static_cast<uint64_t>(i) * lda + static_cast<uint64_t>(kk));
                        va = static_cast<int64_t>(a_data[idx]);
                    }

                    if (b_i32) {
                        const int32_t* b_data = b_i32->data();
                        const uint64_t idx = t_b ? (static_cast<uint64_t>(j) * ldb + static_cast<uint64_t>(kk))
                                                 : (static_cast<uint64_t>(kk) * ldb + static_cast<uint64_t>(j));
                        vb = static_cast<int64_t>(b_data[idx]);
                    } else if (b_i16) {
                        const int16_t* b_data = b_i16->data();
                        const uint64_t idx = t_b ? (static_cast<uint64_t>(j) * ldb + static_cast<uint64_t>(kk))
                                                 : (static_cast<uint64_t>(kk) * ldb + static_cast<uint64_t>(j));
                        vb = static_cast<int64_t>(b_data[idx]);
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
                const uint64_t idx = t_c ? (static_cast<uint64_t>(j) * ldc + static_cast<uint64_t>(i))
                                        : (static_cast<uint64_t>(i) * ldc + static_cast<uint64_t>(j));
                c_data[idx] = out;
            }
        });

        c_i32->set_scalar(a.get_scalar() * b.get_scalar());
        return;
    }

    // 4. Bit x {Float64, Float32, Int32} (scale-first: keep bit-packed inputs)
    auto* a_bit = dynamic_cast<const DenseMatrix<bool>*>(&a);
    auto* b_bit = dynamic_cast<const DenseMatrix<bool>*>(&b);

    if (a_bit && c_f64 && b_f64) {
        const uint64_t m = a.rows();
        const uint64_t k = a.cols();
        const uint64_t n = b.cols();
        if (b.rows() != k || result.rows() != m || result.cols() != n) {
            throw std::invalid_argument("Dimension mismatch");
        }

        // Raw-data fast path: only when both are non-transposed.
        if (!a_bit->is_transposed() && !b_f64->is_transposed() && !c_f64->is_transposed()) {
            debug_trace::set_last("cpu.matmul.bit_x_f64");
            const uint64_t* a_data = a_bit->data();
            const uint64_t words_per_row = a_bit->stride_bytes() / 8;
            const double* b_data = b_f64->data();
            double* c_data = c_f64->data();

            ParallelFor(0, m, [&](size_t i) {
                std::vector<double> acc(n, 0.0);
                const uint64_t* a_row = a_data + i * words_per_row;
                for (uint64_t w = 0; w < words_per_row; ++w) {
                    uint64_t word = a_row[w];
                    while (word) {
                        const uint64_t bit = static_cast<uint64_t>(std::countr_zero(word));
                        const uint64_t kk = w * 64 + bit;
                        if (kk < k) {
                            const double* b_row = b_data + kk * n;
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

        // Correct generic path (supports transpose flags).
        debug_trace::set_last("cpu.matmul.bit_x_f64");
        const bool t_c = c_f64->is_transposed();
        const uint64_t ldc = c_f64->base_cols();
        double* c_data = c_f64->data();
        ParallelFor(0, m, [&](size_t i) {
            for (uint64_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (uint64_t kk = 0; kk < k; ++kk) {
                    if (a_bit->get(static_cast<uint64_t>(i), kk)) {
                        sum += b_f64->get(kk, j);
                    }
                }
                const uint64_t idx = t_c ? (j * ldc + static_cast<uint64_t>(i))
                                        : (static_cast<uint64_t>(i) * ldc + j);
                c_data[idx] = sum;
            }
        });
        c_f64->set_scalar(a_bit->get_scalar() * b_f64->get_scalar());
        return;
    }

    if (a_bit && c_f32 && b_f32) {
        const uint64_t m = a.rows();
        const uint64_t k = a.cols();
        const uint64_t n = b.cols();
        if (b.rows() != k || result.rows() != m || result.cols() != n) {
            throw std::invalid_argument("Dimension mismatch");
        }
        if (!a_bit->is_transposed() && !b_f32->is_transposed() && !c_f32->is_transposed()) {
            debug_trace::set_last("cpu.matmul.bit_x_f32");
            const uint64_t* a_data = a_bit->data();
            const uint64_t words_per_row = a_bit->stride_bytes() / 8;
            const float* b_data = b_f32->data();
            float* c_data = c_f32->data();

            ParallelFor(0, m, [&](size_t i) {
                std::vector<float> acc(n, 0.0f);
                const uint64_t* a_row = a_data + i * words_per_row;
                for (uint64_t w = 0; w < words_per_row; ++w) {
                    uint64_t word = a_row[w];
                    while (word) {
                        const uint64_t bit = static_cast<uint64_t>(std::countr_zero(word));
                        const uint64_t kk = w * 64 + bit;
                        if (kk < k) {
                            const float* b_row = b_data + kk * n;
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

        debug_trace::set_last("cpu.matmul.bit_x_f32");
        const bool t_c = c_f32->is_transposed();
        const uint64_t ldc = c_f32->base_cols();
        float* c_data = c_f32->data();
        ParallelFor(0, m, [&](size_t i) {
            for (uint64_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (uint64_t kk = 0; kk < k; ++kk) {
                    if (a_bit->get(static_cast<uint64_t>(i), kk)) {
                        sum += b_f32->get(kk, j);
                    }
                }
                const uint64_t idx = t_c ? (j * ldc + static_cast<uint64_t>(i))
                                        : (static_cast<uint64_t>(i) * ldc + j);
                c_data[idx] = sum;
            }
        });
        c_f32->set_scalar(a_bit->get_scalar() * b_f32->get_scalar());
        return;
    }

    if (a_bit && c_i32 && b_i32) {
        const uint64_t m = a.rows();
        const uint64_t k = a.cols();
        const uint64_t n = b.cols();
        if (b.rows() != k || result.rows() != m || result.cols() != n) {
            throw std::invalid_argument("Dimension mismatch");
        }
        if (!a_bit->is_transposed() && !b_i32->is_transposed() && !c_i32->is_transposed()) {
            debug_trace::set_last("cpu.matmul.bit_x_i32");
            const uint64_t* a_data = a_bit->data();
            const uint64_t words_per_row = a_bit->stride_bytes() / 8;
            const int32_t* b_data = b_i32->data();
            int32_t* c_data = c_i32->data();

            ParallelFor(0, m, [&](size_t i) {
                std::vector<int64_t> acc(n, 0);
                const uint64_t* a_row = a_data + i * words_per_row;
                for (uint64_t w = 0; w < words_per_row; ++w) {
                    uint64_t word = a_row[w];
                    while (word) {
                        const uint64_t bit = static_cast<uint64_t>(std::countr_zero(word));
                        const uint64_t kk = w * 64 + bit;
                        if (kk < k) {
                            const int32_t* b_row = b_data + kk * n;
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

        debug_trace::set_last("cpu.matmul.bit_x_i32");
        const bool t_c = c_i32->is_transposed();
        const uint64_t ldc = c_i32->base_cols();
        int32_t* c_data = c_i32->data();
        ParallelFor(0, m, [&](size_t i) {
            for (uint64_t j = 0; j < n; ++j) {
                int64_t sum = 0;
                for (uint64_t kk = 0; kk < k; ++kk) {
                    if (a_bit->get(static_cast<uint64_t>(i), kk)) {
                        sum = checked_add_int64(sum, static_cast<int64_t>(b_i32->get(kk, j)));
                    }
                }
                if (sum > static_cast<int64_t>((std::numeric_limits<int32_t>::max)()) ||
                    sum < static_cast<int64_t>((std::numeric_limits<int32_t>::min)())) {
                    throw std::overflow_error("Integer matmul overflow: bit×int32 result does not fit in int32 output");
                }
                const uint64_t idx = t_c ? (j * ldc + static_cast<uint64_t>(i))
                                        : (static_cast<uint64_t>(i) * ldc + j);
                c_data[idx] = static_cast<int32_t>(sum);
            }
        });
        c_i32->set_scalar(a_bit->get_scalar() * b_i32->get_scalar());
        return;
    }

    if (a_bit && c_i16 && b_i16) {
        const uint64_t m = a.rows();
        const uint64_t k = a.cols();
        const uint64_t n = b.cols();
        if (b.rows() != k || result.rows() != m || result.cols() != n) {
            throw std::invalid_argument("Dimension mismatch");
        }
        if (!a_bit->is_transposed() && !b_i16->is_transposed() && !c_i16->is_transposed()) {
            debug_trace::set_last("cpu.matmul.bit_x_i16");
            const uint64_t* a_data = a_bit->data();
            const uint64_t words_per_row = a_bit->stride_bytes() / 8;
            const int16_t* b_data = b_i16->data();
            int16_t* c_data = c_i16->data();

            ParallelFor(0, m, [&](size_t i) {
                std::vector<int64_t> acc(n, 0);
                const uint64_t* a_row = a_data + i * words_per_row;
                for (uint64_t w = 0; w < words_per_row; ++w) {
                    uint64_t word = a_row[w];
                    while (word) {
                        const uint64_t bit = static_cast<uint64_t>(std::countr_zero(word));
                        const uint64_t kk = w * 64 + bit;
                        if (kk < k) {
                            const int16_t* b_row = b_data + kk * n;
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

        debug_trace::set_last("cpu.matmul.bit_x_i16");
        const bool t_c = c_i16->is_transposed();
        const uint64_t ldc = c_i16->base_cols();
        int16_t* c_data = c_i16->data();
        ParallelFor(0, m, [&](size_t i) {
            for (uint64_t j = 0; j < n; ++j) {
                int64_t sum = 0;
                for (uint64_t kk = 0; kk < k; ++kk) {
                    if (a_bit->get(static_cast<uint64_t>(i), kk)) {
                        sum = checked_add_int64(sum, static_cast<int64_t>(b_i16->get(kk, j)));
                    }
                }
                if (sum > static_cast<int64_t>((std::numeric_limits<int16_t>::max)()) ||
                    sum < static_cast<int64_t>((std::numeric_limits<int16_t>::min)())) {
                    throw std::overflow_error("Integer matmul overflow: bit×int16 result does not fit in int16 output");
                }
                const uint64_t idx = t_c ? (j * ldc + static_cast<uint64_t>(i))
                                        : (static_cast<uint64_t>(i) * ldc + j);
                c_data[idx] = static_cast<int16_t>(sum);
            }
        });
        c_i16->set_scalar(a_bit->get_scalar() * b_i16->get_scalar());
        return;
    }

    // 5. BitMatrix Support (bit×bit -> int32)
    auto* c_int = dynamic_cast<DenseMatrix<int32_t>*>(&result);
    if (a_bit && b_bit && c_int) {
        debug_trace::set_last("cpu.matmul.bitbit_popcount");
        const uint64_t m = a.rows();
        const uint64_t k = a.cols();
        const uint64_t n = b.cols();
        if (b.rows() != k || result.rows() != m || result.cols() != n) {
            throw std::invalid_argument("Dimension mismatch");
        }

        // Optimized popcount kernel: requires non-transposed inputs to safely use raw packed rows.
        if (!a_bit->is_transposed() && !b_bit->is_transposed()) {
            const uint64_t* a_data = a_bit->data();
            const uint64_t words_per_row = a_bit->stride_bytes() / 8;
            const uint64_t rem_bits = k % 64u;
            const uint64_t last_mask = (rem_bits == 0u) ? ~0ull : ((1ull << rem_bits) - 1ull);

            auto b_transposed_mat = std::make_unique<DenseBitMatrix>(n, k, "");
            b_transposed_mat->set_temporary(true);
            uint64_t* b_transposed_data = b_transposed_mat->data();

            // Build bit-packed columns of B as rows of B^T.
            for (uint64_t i = 0; i < k; ++i) {
                for (uint64_t j = 0; j < n; ++j) {
                    if (b_bit->get(i, j)) {
                        const uint64_t word_idx = i / 64;
                        const uint64_t bit_idx = i % 64;
                        b_transposed_data[j * words_per_row + word_idx] |= (1ULL << bit_idx);
                    }
                }
            }

            const bool use_avx512 = has_avx512_vpopcntdq();

            ParallelFor(0, m, [&](size_t i) {
                const uint64_t* a_row = a_data + i * words_per_row;
                for (uint64_t j = 0; j < n; ++j) {
                    const uint64_t* b_col = b_transposed_data + j * words_per_row;

                    int64_t dot_product = 0;
                    uint64_t w = 0;

                    // AVX-512 Loop (8 words = 512 bits per iteration)
                    if (use_avx512) {
#if defined(__x86_64__) || defined(_M_X64)
                        const uint64_t avx_limit = words_per_row & ~7ULL; // Multiple of 8
                        dot_product = dot_product_avx512(a_row, b_col, avx_limit);
                        w = avx_limit;
#endif
                    }

                    // Scalar Loop (Tail)
                    for (; w < words_per_row; ++w) {
                        uint64_t aw = a_row[w];
                        uint64_t bw = b_col[w];
                        if (w == words_per_row - 1 && rem_bits != 0u) {
                            aw &= last_mask;
                            bw &= last_mask;
                        }
                        dot_product = checked_add_int64(dot_product, static_cast<int64_t>(std::popcount(aw & bw)));
                    }
                    if (dot_product > static_cast<int64_t>((std::numeric_limits<int32_t>::max)())) {
                        throw std::overflow_error("Integer matmul overflow: bit×bit result does not fit in int32 output");
                    }
                    c_int->set(static_cast<uint64_t>(i), j, static_cast<int32_t>(dot_product));
                }
            });

            c_int->set_scalar(a_bit->get_scalar() * b_bit->get_scalar());
            return;
        }

        // Correct generic fallback (supports transpose flags).
        ParallelFor(0, m, [&](size_t i) {
            for (uint64_t j = 0; j < n; ++j) {
                int64_t dot_product = 0;
                for (uint64_t kk = 0; kk < k; ++kk) {
                    if (a_bit->get(static_cast<uint64_t>(i), kk) && b_bit->get(kk, j)) {
                        dot_product = checked_add_int64(dot_product, 1);
                    }
                }
                if (dot_product > static_cast<int64_t>((std::numeric_limits<int32_t>::max)())) {
                    throw std::overflow_error("Integer matmul overflow: bit×bit result does not fit in int32 output");
                }
                c_int->set(static_cast<uint64_t>(i), j, static_cast<int32_t>(dot_product));
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
        uint64_t n = a.rows();
        
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
        uint64_t n = a.rows();
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
        uint64_t n = a.rows();
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
        uint64_t n = a.rows();
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
    const uint64_t m = a.rows();
    const uint64_t k = a.cols();
    const uint64_t n = b.cols();

    // Generic float32 result fallback for mixed-kind inputs.
    // NOTE: this computes through get_element_as_double (includes scalars) and stores raw values
    // into a float32 result with scalar=1.
    if (c_f32) {
        debug_trace::set_last("cpu.matmul.generic_fallback_f32");
        if (b.rows() != k || result.rows() != m || result.cols() != n) {
            throw std::invalid_argument("Dimension mismatch");
        }

        float* c_data = c_f32->data();
        const bool t_c = c_f32->is_transposed();

        const uint64_t storage_cols = c_f32->base_cols();
        ParallelFor(0, m, [&](size_t i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t kk = 0; kk < k; ++kk) {
                    sum += static_cast<float>(a.get_element_as_double(static_cast<uint64_t>(i), static_cast<uint64_t>(kk))) *
                           static_cast<float>(b.get_element_as_double(static_cast<uint64_t>(kk), static_cast<uint64_t>(j)));
                }
                if (t_c) {
                    c_data[static_cast<uint64_t>(j) * storage_cols + static_cast<uint64_t>(i)] = sum;
                } else {
                    c_data[static_cast<uint64_t>(i) * storage_cols + static_cast<uint64_t>(j)] = sum;
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
    
    ParallelFor(0, m, [&](size_t i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t kk = 0; kk < k; ++kk) {
                sum += a.get_element_as_double(static_cast<uint64_t>(i), static_cast<uint64_t>(kk)) *
                       b.get_element_as_double(static_cast<uint64_t>(kk), static_cast<uint64_t>(j));
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
        
        uint64_t n = a_tri->rows();
        
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

void CpuSolver::inverse(const MatrixBase& matrix_in, MatrixBase& matrix_out) {
    // Triangular
    const auto* in_tri = dynamic_cast<const TriangularMatrix<double>*>(&matrix_in);
    if (in_tri != nullptr) {
        auto* out_tri = dynamic_cast<TriangularMatrix<double>*>(&matrix_out);
        if (out_tri != nullptr) {
            invert_triangular(in_tri, out_tri);
            return;
        }
    }

    // Double
    const auto* in_dense_f64 = dynamic_cast<const DenseMatrix<double>*>(&matrix_in);
    if (in_dense_f64 != nullptr) {
        auto* out_dense_f64 = dynamic_cast<DenseMatrix<double>*>(&matrix_out);
        if (out_dense_f64 != nullptr) {
            inverse_impl(in_dense_f64, out_dense_f64);
            return;
        }
    }

    // Float
    const auto* in_dense_f32 = dynamic_cast<const DenseMatrix<float>*>(&matrix_in);
    if (in_dense_f32 != nullptr) {
        auto* out_dense_f32 = dynamic_cast<DenseMatrix<float>*>(&matrix_out);
        if (out_dense_f32 != nullptr) {
            inverse_impl(in_dense_f32, out_dense_f32);
            return;
        }
    }

    // Diagonal
    const auto* in_diag = dynamic_cast<const DiagonalMatrix<double>*>(&matrix_in);
    if (in_diag != nullptr) {
        auto* out_diag = dynamic_cast<DiagonalMatrix<double>*>(&matrix_out);
        if (out_diag != nullptr) {
            uint64_t n = matrix_in.rows();
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
    uint64_t n = A.rows();
    if (A.cols() != n) {
        throw std::invalid_argument("batch_gemv requires square matrix");
    }
    
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
        const uint64_t rows = result.rows();
        const uint64_t cols = result.cols();

        if (!is_broadcast_compatible_dim(a.rows(), rows) ||
            !is_broadcast_compatible_dim(a.cols(), cols) ||
            !is_broadcast_compatible_dim(b.rows(), rows) ||
            !is_broadcast_compatible_dim(b.cols(), cols)) {
            throw std::invalid_argument("operands could not be broadcast together");
        }

        const bool a_full = (a.rows() == rows) && (a.cols() == cols);
        const bool b_full = (b.rows() == rows) && (b.cols() == cols);
        
        // 1. DenseMatrix Result
        if (auto* res_dense = dynamic_cast<DenseMatrix<T>*>(&result)) {
            auto* a_dense = dynamic_cast<const DenseMatrix<T>*>(&a);
            auto* b_dense = dynamic_cast<const DenseMatrix<T>*>(&b);
            
            // Fast path: All dense, no transpose, scalar=1
            if (a_dense && b_dense && a_full && b_full &&
                res_dense->get_scalar() == 1.0 && !res_dense->is_transposed() &&
                a_dense->get_scalar() == 1.0 && !a_dense->is_transposed() &&
                b_dense->get_scalar() == 1.0 && !b_dense->is_transposed()) {
                
                const T* a_data = a_dense->data();
                const T* b_data = b_dense->data();
                T* res_data = res_dense->data();
                
                ParallelFor(0, rows * cols, [&](size_t i) {
                    res_data[i] = op(a_data[i], b_data[i]);
                });
                return;
            }
            
            // Generic Dense path
            ParallelFor(0, rows, [&](size_t i) {
                for (size_t j = 0; j < cols; ++j) {
                    T val_a;
                    T val_b;
                    const uint64_t ia = broadcast_index(static_cast<uint64_t>(i), a.rows());
                    const uint64_t ja = broadcast_index(static_cast<uint64_t>(j), a.cols());
                    const uint64_t ib = broadcast_index(static_cast<uint64_t>(i), b.rows());
                    const uint64_t jb = broadcast_index(static_cast<uint64_t>(j), b.cols());
                    if constexpr (is_std_complex_v<T>) {
                        val_a = static_cast<T>(a.get_element_as_complex(ia, ja));
                        val_b = static_cast<T>(b.get_element_as_complex(ib, jb));
                    } else {
                        val_a = pycauset::scalar::from_double<T>(a.get_element_as_double(ia, ja));
                        val_b = pycauset::scalar::from_double<T>(b.get_element_as_double(ib, jb));
                    }
                    res_dense->set(i, j, op(val_a, val_b));
                }
            });
            return;
        }

        if constexpr (!is_std_complex_v<T>) {
            // 2. TriangularMatrix Result
            if (auto* res_tri = dynamic_cast<TriangularMatrix<T>*>(&result)) {
                if (rows != cols) {
                    throw std::invalid_argument("Triangular result requires square dimensions");
                }
                ParallelFor(0, rows, [&](size_t i) {
                    for (uint64_t j = static_cast<uint64_t>(i) + 1; j < rows; ++j) {
                        const uint64_t ia = broadcast_index(static_cast<uint64_t>(i), a.rows());
                        const uint64_t ja = broadcast_index(static_cast<uint64_t>(j), a.cols());
                        const uint64_t ib = broadcast_index(static_cast<uint64_t>(i), b.rows());
                        const uint64_t jb = broadcast_index(static_cast<uint64_t>(j), b.cols());
                        T val = op(static_cast<T>(a.get_element_as_double(ia, ja)),
                                   static_cast<T>(b.get_element_as_double(ib, jb)));
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
                     if (rows > 0 && cols > 0) {
                         const uint64_t ia = broadcast_index(0, a.rows());
                         const uint64_t ja = broadcast_index(0, a.cols());
                         const uint64_t ib = broadcast_index(0, b.rows());
                         const uint64_t jb = broadcast_index(0, b.cols());
                         T val = op(static_cast<T>(a.get_element_as_double(ia, ja)),
                                    static_cast<T>(b.get_element_as_double(ib, jb)));
                         result.set_scalar(static_cast<double>(val));
                     }
                     return;
                }

                if (rows != cols) {
                    throw std::invalid_argument("Diagonal result requires square dimensions");
                }

                ParallelFor(0, rows, [&](size_t i) {
                    const uint64_t ia = broadcast_index(static_cast<uint64_t>(i), a.rows());
                    const uint64_t ja = broadcast_index(static_cast<uint64_t>(i), a.cols());
                    const uint64_t ib = broadcast_index(static_cast<uint64_t>(i), b.rows());
                    const uint64_t jb = broadcast_index(static_cast<uint64_t>(i), b.cols());
                    T val = op(static_cast<T>(a.get_element_as_double(ia, ja)),
                               static_cast<T>(b.get_element_as_double(ib, jb)));
                    res_diag->set(i, i, val);
                });
                return;
            }

            // 4. SymmetricMatrix Result
            if (auto* res_sym = dynamic_cast<SymmetricMatrix<T>*>(&result)) {
                if (rows != cols) {
                    throw std::invalid_argument("Symmetric result requires square dimensions");
                }
                ParallelFor(0, rows, [&](size_t i) {
                    // SymmetricMatrix stores upper triangle (including diagonal)
                    for (uint64_t j = static_cast<uint64_t>(i); j < rows; ++j) {
                        const uint64_t ia = broadcast_index(static_cast<uint64_t>(i), a.rows());
                        const uint64_t ja = broadcast_index(static_cast<uint64_t>(j), a.cols());
                        const uint64_t ib = broadcast_index(static_cast<uint64_t>(i), b.rows());
                        const uint64_t jb = broadcast_index(static_cast<uint64_t>(j), b.cols());
                        T val = op(static_cast<T>(a.get_element_as_double(ia, ja)),
                                   static_cast<T>(b.get_element_as_double(ib, jb)));
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
        const uint64_t rows = res->rows();
        const uint64_t cols = res->cols();
        if (a.rows() != rows || a.cols() != cols) {
            throw std::invalid_argument("Dimension mismatch");
        }
        auto* a_dense = dynamic_cast<const DenseMatrix<T>*>(&a);
        
        if (a_dense &&
            res->get_scalar() == 1.0 && !res->is_transposed() &&
            a_dense->get_scalar() == 1.0 && !a_dense->is_transposed()) {
            const T* a_data = a_dense->data();
            T* res_data = res->data();
            T s;
            if constexpr (is_std_complex_v<T>) {
                using V = typename T::value_type;
                s = T{static_cast<V>(scalar), static_cast<V>(0)};
            } else {
                s = pycauset::scalar::from_double<T>(scalar);
            }
            ParallelFor(0, rows * cols, [&](size_t i) {
                res_data[i] = a_data[i] * s;
            });
            return;
        }
        
        ParallelFor(0, rows, [&](size_t i) {
            for (size_t j = 0; j < cols; ++j) {
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
        debug_trace::set_last("cpu.add.cf16");
        if (auto* out = dynamic_cast<ComplexFloat16Matrix*>(&result)) {
            binary_op_complex16_impl(a, b, *out, std::plus<>());
            return;
        }
        throw std::runtime_error("CpuSolver::add complex_float16 result type mismatch");
    }
    if (dtype == DataType::COMPLEX_FLOAT32) {
        debug_trace::set_last("cpu.add.c32");
        binary_op_impl<std::complex<float>>(a, b, result, std::plus<>());
        return;
    }
    if (dtype == DataType::COMPLEX_FLOAT64) {
        debug_trace::set_last("cpu.add.c64");
        binary_op_impl<std::complex<double>>(a, b, result, std::plus<>());
        return;
    }

    if (dtype == DataType::FLOAT64) {
        debug_trace::set_last("cpu.add.f64");
        binary_op_impl<double>(a, b, result, std::plus<>());
    } else if (dtype == DataType::FLOAT16) {
        debug_trace::set_last("cpu.add.f16");
        binary_op_impl<float16_t>(a, b, result, std::plus<>());
    } else if (dtype == DataType::FLOAT32) {
        debug_trace::set_last("cpu.add.f32");
        binary_op_impl<float>(a, b, result, std::plus<>());
    } else if (dtype == DataType::INT8) {
        debug_trace::set_last("cpu.add.i8");
        binary_op_impl<int8_t>(a, b, result, std::plus<>());
    } else if (dtype == DataType::INT16) {
        debug_trace::set_last("cpu.add.i16");
        binary_op_impl<int16_t>(a, b, result, std::plus<>());
    } else if (dtype == DataType::INT32) {
        debug_trace::set_last("cpu.add.i32");
        binary_op_impl<int32_t>(a, b, result, std::plus<>());
    } else if (dtype == DataType::INT64) {
        debug_trace::set_last("cpu.add.i64");
        binary_op_impl<int64_t>(a, b, result, std::plus<>());
    } else if (dtype == DataType::UINT8) {
        debug_trace::set_last("cpu.add.u8");
        binary_op_impl<uint8_t>(a, b, result, std::plus<>());
    } else if (dtype == DataType::UINT16) {
        debug_trace::set_last("cpu.add.u16");
        binary_op_impl<uint16_t>(a, b, result, std::plus<>());
    } else if (dtype == DataType::UINT32) {
        debug_trace::set_last("cpu.add.u32");
        binary_op_impl<uint32_t>(a, b, result, std::plus<>());
    } else if (dtype == DataType::UINT64) {
        debug_trace::set_last("cpu.add.u64");
        binary_op_impl<uint64_t>(a, b, result, std::plus<>());
    } else {
        throw std::runtime_error("CpuSolver::add result data type not supported");
    }
}

void CpuSolver::subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    DataType dtype = result.get_data_type();
    if (dtype == DataType::COMPLEX_FLOAT16) {
        debug_trace::set_last("cpu.subtract.cf16");
        if (auto* out = dynamic_cast<ComplexFloat16Matrix*>(&result)) {
            binary_op_complex16_impl(a, b, *out, std::minus<>());
            return;
        }
        throw std::runtime_error("CpuSolver::subtract complex_float16 result type mismatch");
    }
    if (dtype == DataType::COMPLEX_FLOAT32) {
        debug_trace::set_last("cpu.subtract.c32");
        binary_op_impl<std::complex<float>>(a, b, result, std::minus<>());
        return;
    }
    if (dtype == DataType::COMPLEX_FLOAT64) {
        debug_trace::set_last("cpu.subtract.c64");
        binary_op_impl<std::complex<double>>(a, b, result, std::minus<>());
        return;
    }

    if (dtype == DataType::FLOAT64) {
        debug_trace::set_last("cpu.subtract.f64");
        binary_op_impl<double>(a, b, result, std::minus<>());
    } else if (dtype == DataType::FLOAT16) {
        debug_trace::set_last("cpu.subtract.f16");
        binary_op_impl<float16_t>(a, b, result, std::minus<>());
    } else if (dtype == DataType::FLOAT32) {
        debug_trace::set_last("cpu.subtract.f32");
        binary_op_impl<float>(a, b, result, std::minus<>());
    } else if (dtype == DataType::INT8) {
        debug_trace::set_last("cpu.subtract.i8");
        binary_op_impl<int8_t>(a, b, result, std::minus<>());
    } else if (dtype == DataType::INT16) {
        debug_trace::set_last("cpu.subtract.i16");
        binary_op_impl<int16_t>(a, b, result, std::minus<>());
    } else if (dtype == DataType::INT32) {
        debug_trace::set_last("cpu.subtract.i32");
        binary_op_impl<int32_t>(a, b, result, std::minus<>());
    } else if (dtype == DataType::INT64) {
        debug_trace::set_last("cpu.subtract.i64");
        binary_op_impl<int64_t>(a, b, result, std::minus<>());
    } else if (dtype == DataType::UINT8) {
        debug_trace::set_last("cpu.subtract.u8");
        binary_op_impl<uint8_t>(a, b, result, std::minus<>());
    } else if (dtype == DataType::UINT16) {
        debug_trace::set_last("cpu.subtract.u16");
        binary_op_impl<uint16_t>(a, b, result, std::minus<>());
    } else if (dtype == DataType::UINT32) {
        debug_trace::set_last("cpu.subtract.u32");
        binary_op_impl<uint32_t>(a, b, result, std::minus<>());
    } else if (dtype == DataType::UINT64) {
        debug_trace::set_last("cpu.subtract.u64");
        binary_op_impl<uint64_t>(a, b, result, std::minus<>());
    } else {
        throw std::runtime_error("CpuSolver::subtract result data type not supported");
    }
}

void CpuSolver::elementwise_multiply(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    DataType dtype = result.get_data_type();
    if (dtype == DataType::COMPLEX_FLOAT16) {
        debug_trace::set_last("cpu.mul.cf16");
        if (auto* out = dynamic_cast<ComplexFloat16Matrix*>(&result)) {
            binary_op_complex16_impl(a, b, *out, std::multiplies<>());
            return;
        }
        throw std::runtime_error("CpuSolver::elementwise_multiply complex_float16 result type mismatch");
    }
    if (dtype == DataType::COMPLEX_FLOAT32) {
        debug_trace::set_last("cpu.mul.c32");
        binary_op_impl<std::complex<float>>(a, b, result, std::multiplies<>());
        return;
    }
    if (dtype == DataType::COMPLEX_FLOAT64) {
        debug_trace::set_last("cpu.mul.c64");
        binary_op_impl<std::complex<double>>(a, b, result, std::multiplies<>());
        return;
    }

    if (dtype == DataType::BIT) {
        auto* out = dynamic_cast<DenseBitMatrix*>(&result);
        if (!out) {
            throw std::runtime_error("CpuSolver::elementwise_multiply bit result type mismatch");
        }
        const uint64_t rows = out->rows();
        const uint64_t cols = out->cols();
        if (!is_broadcast_compatible_dim(a.rows(), rows) ||
            !is_broadcast_compatible_dim(a.cols(), cols) ||
            !is_broadcast_compatible_dim(b.rows(), rows) ||
            !is_broadcast_compatible_dim(b.cols(), cols)) {
            throw std::invalid_argument("operands could not be broadcast together");
        }
        debug_trace::set_last("cpu.elementwise_multiply.bit");
        ParallelFor(0, rows, [&](size_t i) {
            for (uint64_t j = 0; j < cols; ++j) {
                const uint64_t ia = broadcast_index(static_cast<uint64_t>(i), a.rows());
                const uint64_t ja = broadcast_index(static_cast<uint64_t>(j), a.cols());
                const uint64_t ib = broadcast_index(static_cast<uint64_t>(i), b.rows());
                const uint64_t jb = broadcast_index(static_cast<uint64_t>(j), b.cols());
                const bool va = (a.get_element_as_double(ia, ja) != 0.0);
                const bool vb = (b.get_element_as_double(ib, jb) != 0.0);
                out->set(i, j, va && vb);
            }
        });
        out->set_scalar(1.0);
        return;
    }

    if (dtype == DataType::FLOAT64) {
        debug_trace::set_last("cpu.mul.f64");
        binary_op_impl<double>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::FLOAT16) {
        debug_trace::set_last("cpu.mul.f16");
        binary_op_impl<float16_t>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::FLOAT32) {
        debug_trace::set_last("cpu.mul.f32");
        binary_op_impl<float>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::INT8) {
        debug_trace::set_last("cpu.mul.i8");
        binary_op_impl<int8_t>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::INT16) {
        debug_trace::set_last("cpu.mul.i16");
        binary_op_impl<int16_t>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::INT32) {
        debug_trace::set_last("cpu.mul.i32");
        binary_op_impl<int32_t>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::INT64) {
        debug_trace::set_last("cpu.mul.i64");
        binary_op_impl<int64_t>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::UINT8) {
        debug_trace::set_last("cpu.mul.u8");
        binary_op_impl<uint8_t>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::UINT16) {
        debug_trace::set_last("cpu.mul.u16");
        binary_op_impl<uint16_t>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::UINT32) {
        debug_trace::set_last("cpu.mul.u32");
        binary_op_impl<uint32_t>(a, b, result, std::multiplies<>());
    } else if (dtype == DataType::UINT64) {
        debug_trace::set_last("cpu.mul.u64");
        binary_op_impl<uint64_t>(a, b, result, std::multiplies<>());
    } else {
        throw std::runtime_error("CpuSolver::elementwise_multiply result data type not supported");
    }
}

void CpuSolver::elementwise_divide(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    DataType dtype = result.get_data_type();
    if (dtype == DataType::COMPLEX_FLOAT16) {
        debug_trace::set_last("cpu.div.cf16");
        if (auto* out = dynamic_cast<ComplexFloat16Matrix*>(&result)) {
            binary_op_complex16_impl(a, b, *out, [](auto x, auto y) { return x / y; });
            return;
        }
        throw std::runtime_error("CpuSolver::elementwise_divide complex_float16 result type mismatch");
    }
    if (dtype == DataType::COMPLEX_FLOAT32) {
        debug_trace::set_last("cpu.div.c32");
        binary_op_impl<std::complex<float>>(a, b, result, [](auto x, auto y) { return x / y; });
        return;
    }
    if (dtype == DataType::COMPLEX_FLOAT64) {
        debug_trace::set_last("cpu.div.c64");
        binary_op_impl<std::complex<double>>(a, b, result, [](auto x, auto y) { return x / y; });
        return;
    }

    if (dtype == DataType::BIT) {
        throw std::runtime_error("CpuSolver::elementwise_divide bit result type not supported");
    }

    if (dtype == DataType::FLOAT64) {
        debug_trace::set_last("cpu.div.f64");
        binary_op_impl<double>(a, b, result, [](auto x, auto y) { return x / y; });
    } else if (dtype == DataType::FLOAT16) {
        debug_trace::set_last("cpu.div.f16");
        binary_op_impl<float16_t>(a, b, result, [](auto x, auto y) { return x / y; });
    } else if (dtype == DataType::FLOAT32) {
        debug_trace::set_last("cpu.div.f32");
        binary_op_impl<float>(a, b, result, [](auto x, auto y) { return x / y; });
    } else if (dtype == DataType::INT8) {
        debug_trace::set_last("cpu.div.i8");
        binary_op_impl<int8_t>(a, b, result, [](auto x, auto y) { return x / y; });
    } else if (dtype == DataType::INT16) {
        debug_trace::set_last("cpu.div.i16");
        binary_op_impl<int16_t>(a, b, result, [](auto x, auto y) { return x / y; });
    } else if (dtype == DataType::INT32) {
        debug_trace::set_last("cpu.div.i32");
        binary_op_impl<int32_t>(a, b, result, [](auto x, auto y) { return x / y; });
    } else if (dtype == DataType::INT64) {
        debug_trace::set_last("cpu.div.i64");
        binary_op_impl<int64_t>(a, b, result, [](auto x, auto y) { return x / y; });
    } else if (dtype == DataType::UINT8) {
        debug_trace::set_last("cpu.div.u8");
        binary_op_impl<uint8_t>(a, b, result, [](auto x, auto y) { return x / y; });
    } else if (dtype == DataType::UINT16) {
        debug_trace::set_last("cpu.div.u16");
        binary_op_impl<uint16_t>(a, b, result, [](auto x, auto y) { return x / y; });
    } else if (dtype == DataType::UINT32) {
        debug_trace::set_last("cpu.div.u32");
        binary_op_impl<uint32_t>(a, b, result, [](auto x, auto y) { return x / y; });
    } else if (dtype == DataType::UINT64) {
        debug_trace::set_last("cpu.div.u64");
        binary_op_impl<uint64_t>(a, b, result, [](auto x, auto y) { return x / y; });
    } else {
        throw std::runtime_error("CpuSolver::elementwise_divide result data type not supported");
    }
}

void CpuSolver::multiply_scalar(const MatrixBase& a, double scalar, MatrixBase& result) {
    DataType dtype = result.get_data_type();
    if (dtype == DataType::COMPLEX_FLOAT16) {
        if (auto* out = dynamic_cast<ComplexFloat16Matrix*>(&result)) {
            const uint64_t rows = out->rows();
            const uint64_t cols = out->cols();
            ParallelFor(0, rows, [&](size_t i) {
                for (size_t j = 0; j < cols; ++j) {
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

std::complex<double> CpuSolver::dot_complex(const VectorBase& a, const VectorBase& b) {
    const uint64_t n = a.size();
    if (b.size() != n) {
        throw std::invalid_argument("Vector dimensions mismatch");
    }

    // Semantics: simple sum_i a[i] * b[i] (no conjugation).
    debug_trace::set_last("cpu.dot.complex_fallback");
    double re = 0.0;
    double im = 0.0;

    #pragma omp parallel for reduction(+:re, im)
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        const std::complex<double> prod = a.get_element_as_complex(static_cast<uint64_t>(i)) *
                                          b.get_element_as_complex(static_cast<uint64_t>(i));
        re += prod.real();
        im += prod.imag();
    }
    return {re, im};
}

std::complex<double> CpuSolver::sum(const VectorBase& v) {
    const uint64_t n = v.size();
    if (n == 0) {
        return {0.0, 0.0};
    }

    const auto dt = v.get_data_type();
    const bool complex =
        (dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64);

    if (complex) {
        debug_trace::set_last("cpu.sum.vector.complex");
        double re = 0.0;
        double im = 0.0;
        #pragma omp parallel for reduction(+:re, im)
        for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
            const std::complex<double> z = v.get_element_as_complex(static_cast<uint64_t>(i));
            re += z.real();
            im += z.imag();
        }
        return {re, im};
    }

    debug_trace::set_last("cpu.sum.vector.real");
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        sum += v.get_element_as_double(static_cast<uint64_t>(i));
    }
    return {sum, 0.0};
}

double CpuSolver::l2_norm(const VectorBase& v) {
    const uint64_t n = v.size();
    if (n == 0) return 0.0;

    const auto dt = v.get_data_type();
    if (dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64) {
        debug_trace::set_last("cpu.l2_norm.complex");
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
            const std::complex<double> z = v.get_element_as_complex(static_cast<uint64_t>(i));
            sum += std::norm(z);
        }
        return std::sqrt(sum);
    }

    debug_trace::set_last("cpu.l2_norm.real");
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        const double x = v.get_element_as_double(static_cast<uint64_t>(i));
        sum += x * x;
    }
    return std::sqrt(sum);
}

void CpuSolver::matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, VectorBase& result) {
    if (m.cols() != v.size()) throw std::invalid_argument("Dimension mismatch");
    const uint64_t rows = m.rows();
    const uint64_t cols = m.cols();
    if (result.size() != rows) throw std::invalid_argument("Dimension mismatch");

    // Identity Optimization
    if (m.get_matrix_type() == MatrixType::IDENTITY) {
        if (rows != cols) throw std::invalid_argument("Dimension mismatch");
        const DataType res_dt = result.get_data_type();
        if (res_dt == DataType::COMPLEX_FLOAT16 || res_dt == DataType::COMPLEX_FLOAT32 || res_dt == DataType::COMPLEX_FLOAT64) {
            debug_trace::set_last("cpu.matvec.identity.complex");
            const std::complex<double> scalar = m.get_scalar();
            if (auto* out16 = dynamic_cast<ComplexFloat16Vector*>(&result)) {
                auto* rdst = out16->real_data();
                auto* idst = out16->imag_data();
                ParallelFor(0, rows, [&](size_t i) {
                    const std::complex<double> z = v.get_element_as_complex(i) * scalar;
                    rdst[i] = float16_t(z.real());
                    idst[i] = float16_t(z.imag());
                });
                out16->set_scalar(1.0);
                return;
            }
            if (auto* out32 = dynamic_cast<DenseVector<std::complex<float>>*>(&result)) {
                auto* dst = out32->data();
                ParallelFor(0, rows, [&](size_t i) {
                    const std::complex<double> z = v.get_element_as_complex(i) * scalar;
                    dst[i] = std::complex<float>(static_cast<float>(z.real()), static_cast<float>(z.imag()));
                });
                out32->set_scalar(1.0);
                return;
            }
            if (auto* out64 = dynamic_cast<DenseVector<std::complex<double>>*>(&result)) {
                auto* dst = out64->data();
                ParallelFor(0, rows, [&](size_t i) { dst[i] = v.get_element_as_complex(i) * scalar; });
                out64->set_scalar(1.0);
                return;
            }
            throw std::runtime_error("Unsupported complex result type for matrix_vector_multiply");
        }

        double scalar = m.get_scalar().real();
        if (auto* res_int = dynamic_cast<DenseVector<int32_t>*>(&result)) {
            ParallelFor(0, rows, [&](size_t i) {
                res_int->set(i, (int32_t)(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_i8 = dynamic_cast<DenseVector<int8_t>*>(&result)) {
            ParallelFor(0, rows, [&](size_t i) {
                res_i8->set(i, static_cast<int8_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_i16 = dynamic_cast<DenseVector<int16_t>*>(&result)) {
            ParallelFor(0, rows, [&](size_t i) {
                res_i16->set(i, static_cast<int16_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_i64 = dynamic_cast<DenseVector<int64_t>*>(&result)) {
            ParallelFor(0, rows, [&](size_t i) {
                res_i64->set(i, static_cast<int64_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_u8 = dynamic_cast<DenseVector<uint8_t>*>(&result)) {
            ParallelFor(0, rows, [&](size_t i) {
                res_u8->set(i, static_cast<uint8_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_u16 = dynamic_cast<DenseVector<uint16_t>*>(&result)) {
            ParallelFor(0, rows, [&](size_t i) {
                res_u16->set(i, static_cast<uint16_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_u32 = dynamic_cast<DenseVector<uint32_t>*>(&result)) {
            ParallelFor(0, rows, [&](size_t i) {
                res_u32->set(i, static_cast<uint32_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_u64 = dynamic_cast<DenseVector<uint64_t>*>(&result)) {
            ParallelFor(0, rows, [&](size_t i) {
                res_u64->set(i, static_cast<uint64_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_f16 = dynamic_cast<DenseVector<float16_t>*>(&result)) {
            ParallelFor(0, rows, [&](size_t i) {
                res_f16->set(i, float16_t(static_cast<float>(v.get_element_as_double(i) * scalar)));
            });
        } else if (auto* res_f32 = dynamic_cast<DenseVector<float>*>(&result)) {
            ParallelFor(0, rows, [&](size_t i) {
                res_f32->set(i, static_cast<float>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_dbl = dynamic_cast<DenseVector<double>*>(&result)) {
            ParallelFor(0, rows, [&](size_t i) {
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
            ParallelFor(0, rows, [&](size_t i) {
                std::complex<double> sum = 0.0;
                for (uint64_t j = 0; j < cols; ++j) {
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
            ParallelFor(0, rows, [&](size_t i) {
                std::complex<double> sum = 0.0;
                for (uint64_t j = 0; j < cols; ++j) {
                    sum += m.get_element_as_complex(i, j) * v.get_element_as_complex(j);
                }
                dst[i] = std::complex<float>(static_cast<float>(sum.real()), static_cast<float>(sum.imag()));
            });
            out32->set_scalar(1.0);
            return;
        }
        if (auto* out64 = dynamic_cast<DenseVector<std::complex<double>>*>(&result)) {
            auto* dst = out64->data();
            ParallelFor(0, rows, [&](size_t i) {
                std::complex<double> sum = 0.0;
                for (uint64_t j = 0; j < cols; ++j) {
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
        if (m_bit->is_transposed()) {
            // Fall through to generic path for correctness.
        } else {
            debug_trace::set_last("cpu.matvec.bit_x_bit_popcount");
            const uint64_t* m_data = m_bit->data();
            const uint64_t* v_data = v_bit->data();
            int32_t* res_data = res_int->data();
            
            const uint64_t stride_bytes = m_bit->stride_bytes();
            const uint64_t words_per_row = stride_bytes / 8;
            const uint64_t words_for_cols = (cols + 63u) / 64u;
            const uint64_t rem_bits = cols % 64u;
            const uint64_t last_mask = (rem_bits == 0u) ? ~0ull : ((1ull << rem_bits) - 1ull);
            
            ParallelFor(0, rows, [&](size_t i) {
                const uint64_t* row_ptr = m_data + i * words_per_row;
                int32_t sum = 0;
                for (uint64_t w = 0; w < words_for_cols; ++w) {
                    uint64_t mw = row_ptr[w];
                    uint64_t vw = v_data[w];
                    if (w == words_for_cols - 1 && rem_bits != 0u) {
                        mw &= last_mask;
                        vw &= last_mask;
                    }
                    sum += std::popcount(mw & vw);
                }
                res_data[i] = sum;
            });

            res_int->set_scalar(m_bit->get_scalar() * v_bit->get_scalar());
            return;
        }
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
            const uint64_t words_for_cols = (cols + 63u) / 64u;
            const uint64_t rem_bits = cols % 64u;
            const uint64_t last_mask = (rem_bits == 0u) ? ~0ull : ((1ull << rem_bits) - 1ull);
            ParallelFor(0, rows, [&](size_t i) {
                const uint64_t* row_ptr = m_data + i * words_per_row;
                int64_t sum = 0;
                for (uint64_t w = 0; w < words_for_cols; ++w) {
                    uint64_t word = row_ptr[w];
                    if (w == words_for_cols - 1 && rem_bits != 0u) word &= last_mask;
                    while (word) {
                        const uint64_t bit = static_cast<uint64_t>(std::countr_zero(word));
                        const uint64_t k = w * 64 + bit;
                        if (k < cols) {
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
            const uint64_t words_for_cols = (cols + 63u) / 64u;
            const uint64_t rem_bits = cols % 64u;
            const uint64_t last_mask = (rem_bits == 0u) ? ~0ull : ((1ull << rem_bits) - 1ull);
            ParallelFor(0, rows, [&](size_t i) {
                const uint64_t* row_ptr = m_data + i * words_per_row;
                int64_t sum = 0;
                for (uint64_t w = 0; w < words_for_cols; ++w) {
                    uint64_t word = row_ptr[w];
                    if (w == words_for_cols - 1 && rem_bits != 0u) word &= last_mask;
                    while (word) {
                        const uint64_t bit = static_cast<uint64_t>(std::countr_zero(word));
                        const uint64_t k = w * 64 + bit;
                        if (k < cols) {
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
            const uint64_t words_for_cols = (cols + 63u) / 64u;
            const uint64_t rem_bits = cols % 64u;
            const uint64_t last_mask = (rem_bits == 0u) ? ~0ull : ((1ull << rem_bits) - 1ull);
            ParallelFor(0, rows, [&](size_t i) {
                const uint64_t* row_ptr = m_data + i * words_per_row;
                double sum = 0.0;
                for (uint64_t w = 0; w < words_for_cols; ++w) {
                    uint64_t word = row_ptr[w];
                    if (w == words_for_cols - 1 && rem_bits != 0u) word &= last_mask;
                    while (word) {
                        const uint64_t bit = static_cast<uint64_t>(std::countr_zero(word));
                        const uint64_t k = w * 64 + bit;
                        if (k < cols) {
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
        ParallelFor(0, rows, [&](size_t i) {
            int64_t sum = 0;
            for (uint64_t j = 0; j < cols; ++j) {
                sum += static_cast<int64_t>(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_int->set(i, static_cast<int32_t>(sum));
        });
    } else if (auto* res_i8_fb = dynamic_cast<DenseVector<int8_t>*>(&result)) {
        ParallelFor(0, rows, [&](size_t i) {
            int64_t sum = 0;
            for (uint64_t j = 0; j < cols; ++j) {
                sum += static_cast<int64_t>(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_i8_fb->set(i, static_cast<int8_t>(sum));
        });
    } else if (auto* res_i16_fb = dynamic_cast<DenseVector<int16_t>*>(&result)) {
        ParallelFor(0, rows, [&](size_t i) {
            int64_t sum = 0;
            for (uint64_t j = 0; j < cols; ++j) {
                sum += static_cast<int64_t>(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_i16_fb->set(i, static_cast<int16_t>(sum));
        });
    } else if (auto* res_i64_fb = dynamic_cast<DenseVector<int64_t>*>(&result)) {
        ParallelFor(0, rows, [&](size_t i) {
            int64_t sum = 0;
            for (uint64_t j = 0; j < cols; ++j) {
                sum += static_cast<int64_t>(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_i64_fb->set(i, sum);
        });
    } else if (auto* res_u8_fb = dynamic_cast<DenseVector<uint8_t>*>(&result)) {
        ParallelFor(0, rows, [&](size_t i) {
            uint64_t sum = 0;
            for (uint64_t j = 0; j < cols; ++j) {
                sum += static_cast<uint64_t>(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_u8_fb->set(i, static_cast<uint8_t>(sum));
        });
    } else if (auto* res_u16_fb = dynamic_cast<DenseVector<uint16_t>*>(&result)) {
        ParallelFor(0, rows, [&](size_t i) {
            uint64_t sum = 0;
            for (uint64_t j = 0; j < cols; ++j) {
                sum += static_cast<uint64_t>(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_u16_fb->set(i, static_cast<uint16_t>(sum));
        });
    } else if (auto* res_u32_fb = dynamic_cast<DenseVector<uint32_t>*>(&result)) {
        ParallelFor(0, rows, [&](size_t i) {
            uint64_t sum = 0;
            for (uint64_t j = 0; j < cols; ++j) {
                sum += static_cast<uint64_t>(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_u32_fb->set(i, static_cast<uint32_t>(sum));
        });
    } else if (auto* res_u64_fb = dynamic_cast<DenseVector<uint64_t>*>(&result)) {
        ParallelFor(0, rows, [&](size_t i) {
            uint64_t sum = 0;
            for (uint64_t j = 0; j < cols; ++j) {
                sum += static_cast<uint64_t>(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_u64_fb->set(i, sum);
        });
    } else if (auto* res_dbl = dynamic_cast<DenseVector<double>*>(&result)) {
        ParallelFor(0, rows, [&](size_t i) {
            double sum = 0.0;
            for (uint64_t j = 0; j < cols; ++j) {
                sum += m.get_element_as_double(i, j) * v.get_element_as_double(j);
            }
            res_dbl->set(i, sum);
        });
    } else if (auto* res_f32 = dynamic_cast<DenseVector<float>*>(&result)) {
        ParallelFor(0, rows, [&](size_t i) {
            float sum = 0.0f;
            for (uint64_t j = 0; j < cols; ++j) {
                sum += static_cast<float>(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_f32->set(i, sum);
        });
    } else if (auto* res_f16 = dynamic_cast<DenseVector<float16_t>*>(&result)) {
        ParallelFor(0, rows, [&](size_t i) {
            float sum = 0.0f;
            for (uint64_t j = 0; j < cols; ++j) {
                sum += static_cast<float>(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_f16->set(i, float16_t(sum));
        });
    } else {
        throw std::runtime_error("Unsupported result type for matrix_vector_multiply");
    }
}

void CpuSolver::vector_matrix_multiply(const VectorBase& v, const MatrixBase& m, VectorBase& result) {
    const uint64_t rows = m.rows();
    const uint64_t cols = m.cols();
    if (rows != v.size()) throw std::invalid_argument("Dimension mismatch");
    if (result.size() != cols) throw std::invalid_argument("Dimension mismatch");

    // Identity Optimization
    if (m.get_matrix_type() == MatrixType::IDENTITY) {
        if (rows != cols) throw std::invalid_argument("Dimension mismatch");
        const DataType res_dt = result.get_data_type();
        if (res_dt == DataType::COMPLEX_FLOAT16 || res_dt == DataType::COMPLEX_FLOAT32 || res_dt == DataType::COMPLEX_FLOAT64) {
            debug_trace::set_last("cpu.vecmat.identity.complex");
            const std::complex<double> scalar = m.get_scalar();
            if (auto* out16 = dynamic_cast<ComplexFloat16Vector*>(&result)) {
                auto* rdst = out16->real_data();
                auto* idst = out16->imag_data();
                ParallelFor(0, cols, [&](size_t i) {
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
                ParallelFor(0, cols, [&](size_t i) {
                    const std::complex<double> z = v.get_element_as_complex(i) * scalar;
                    dst[i] = std::complex<float>(static_cast<float>(z.real()), static_cast<float>(z.imag()));
                });
                out32->set_scalar(1.0);
                result.set_transposed(true);
                return;
            }
            if (auto* out64 = dynamic_cast<DenseVector<std::complex<double>>*>(&result)) {
                auto* dst = out64->data();
                ParallelFor(0, cols, [&](size_t i) { dst[i] = v.get_element_as_complex(i) * scalar; });
                out64->set_scalar(1.0);
                result.set_transposed(true);
                return;
            }
            throw std::runtime_error("Unsupported complex result type for vector_matrix_multiply");
        }

        double scalar = m.get_scalar().real();
        if (auto* res_int = dynamic_cast<DenseVector<int32_t>*>(&result)) {
            ParallelFor(0, cols, [&](size_t i) {
                res_int->set(i, (int32_t)(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_i8 = dynamic_cast<DenseVector<int8_t>*>(&result)) {
            ParallelFor(0, cols, [&](size_t i) {
                res_i8->set(i, static_cast<int8_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_i16 = dynamic_cast<DenseVector<int16_t>*>(&result)) {
            ParallelFor(0, cols, [&](size_t i) {
                res_i16->set(i, static_cast<int16_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_i64 = dynamic_cast<DenseVector<int64_t>*>(&result)) {
            ParallelFor(0, cols, [&](size_t i) {
                res_i64->set(i, static_cast<int64_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_u8 = dynamic_cast<DenseVector<uint8_t>*>(&result)) {
            ParallelFor(0, cols, [&](size_t i) {
                res_u8->set(i, static_cast<uint8_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_u16 = dynamic_cast<DenseVector<uint16_t>*>(&result)) {
            ParallelFor(0, cols, [&](size_t i) {
                res_u16->set(i, static_cast<uint16_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_u32 = dynamic_cast<DenseVector<uint32_t>*>(&result)) {
            ParallelFor(0, cols, [&](size_t i) {
                res_u32->set(i, static_cast<uint32_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_u64 = dynamic_cast<DenseVector<uint64_t>*>(&result)) {
            ParallelFor(0, cols, [&](size_t i) {
                res_u64->set(i, static_cast<uint64_t>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_f16 = dynamic_cast<DenseVector<float16_t>*>(&result)) {
            ParallelFor(0, cols, [&](size_t i) {
                res_f16->set(i, float16_t(static_cast<float>(v.get_element_as_double(i) * scalar)));
            });
        } else if (auto* res_f32 = dynamic_cast<DenseVector<float>*>(&result)) {
            ParallelFor(0, cols, [&](size_t i) {
                res_f32->set(i, static_cast<float>(v.get_element_as_double(i) * scalar));
            });
        } else if (auto* res_dbl = dynamic_cast<DenseVector<double>*>(&result)) {
            ParallelFor(0, cols, [&](size_t i) {
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
            ParallelFor(0, cols, [&](size_t j) {
                std::complex<double> sum = 0.0;
                for (uint64_t i = 0; i < rows; ++i) {
                    sum += v.get_element_as_complex(i) * m.get_element_as_complex(i, j);
                }
                rdst[j] = float16_t(sum.real());
                idst[j] = float16_t(sum.imag());
            });
            out16->set_scalar(1.0);
            result.set_transposed(true);
            return;
        }
        if (auto* out32 = dynamic_cast<DenseVector<std::complex<float>>*>(&result)) {
            auto* dst = out32->data();
            ParallelFor(0, cols, [&](size_t j) {
                std::complex<double> sum = 0.0;
                for (uint64_t i = 0; i < rows; ++i) {
                    sum += v.get_element_as_complex(i) * m.get_element_as_complex(i, j);
                }
                dst[j] = std::complex<float>(static_cast<float>(sum.real()), static_cast<float>(sum.imag()));
            });
            out32->set_scalar(1.0);
            result.set_transposed(true);
            return;
        }
        if (auto* out64 = dynamic_cast<DenseVector<std::complex<double>>*>(&result)) {
            auto* dst = out64->data();
            ParallelFor(0, cols, [&](size_t j) {
                std::complex<double> sum = 0.0;
                for (uint64_t i = 0; i < rows; ++i) {
                    sum += v.get_element_as_complex(i) * m.get_element_as_complex(i, j);
                }
                dst[j] = sum;
            });
            out64->set_scalar(1.0);
            result.set_transposed(true);
            return;
        }
        throw std::runtime_error("Unsupported complex result type for vector_matrix_multiply");
    }

    // Generic Fallback
    if (auto* res_int = dynamic_cast<DenseVector<int32_t>*>(&result)) {
        ParallelFor(0, cols, [&](size_t j) {
            int64_t sum = 0;
            for (uint64_t i = 0; i < rows; ++i) {
                sum += static_cast<int64_t>(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_int->set(j, static_cast<int32_t>(sum));
        });
    } else if (auto* res_i8_fb = dynamic_cast<DenseVector<int8_t>*>(&result)) {
        ParallelFor(0, cols, [&](size_t j) {
            int64_t sum = 0;
            for (uint64_t i = 0; i < rows; ++i) {
                sum += static_cast<int64_t>(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_i8_fb->set(j, static_cast<int8_t>(sum));
        });
    } else if (auto* res_i16_fb = dynamic_cast<DenseVector<int16_t>*>(&result)) {
        ParallelFor(0, cols, [&](size_t j) {
            int64_t sum = 0;
            for (uint64_t i = 0; i < rows; ++i) {
                sum += static_cast<int64_t>(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_i16_fb->set(j, static_cast<int16_t>(sum));
        });
    } else if (auto* res_i64_fb = dynamic_cast<DenseVector<int64_t>*>(&result)) {
        ParallelFor(0, cols, [&](size_t j) {
            int64_t sum = 0;
            for (uint64_t i = 0; i < rows; ++i) {
                sum += static_cast<int64_t>(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_i64_fb->set(j, sum);
        });
    } else if (auto* res_u8_fb = dynamic_cast<DenseVector<uint8_t>*>(&result)) {
        ParallelFor(0, cols, [&](size_t j) {
            uint64_t sum = 0;
            for (uint64_t i = 0; i < rows; ++i) {
                sum += static_cast<uint64_t>(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_u8_fb->set(j, static_cast<uint8_t>(sum));
        });
    } else if (auto* res_u16_fb = dynamic_cast<DenseVector<uint16_t>*>(&result)) {
        ParallelFor(0, cols, [&](size_t j) {
            uint64_t sum = 0;
            for (uint64_t i = 0; i < rows; ++i) {
                sum += static_cast<uint64_t>(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_u16_fb->set(j, static_cast<uint16_t>(sum));
        });
    } else if (auto* res_u32_fb = dynamic_cast<DenseVector<uint32_t>*>(&result)) {
        ParallelFor(0, cols, [&](size_t j) {
            uint64_t sum = 0;
            for (uint64_t i = 0; i < rows; ++i) {
                sum += static_cast<uint64_t>(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_u32_fb->set(j, static_cast<uint32_t>(sum));
        });
    } else if (auto* res_u64_fb = dynamic_cast<DenseVector<uint64_t>*>(&result)) {
        ParallelFor(0, cols, [&](size_t j) {
            uint64_t sum = 0;
            for (uint64_t i = 0; i < rows; ++i) {
                sum += static_cast<uint64_t>(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_u64_fb->set(j, sum);
        });
    } else if (auto* res_dbl = dynamic_cast<DenseVector<double>*>(&result)) {
        ParallelFor(0, cols, [&](size_t j) {
            double sum = 0.0;
            for (uint64_t i = 0; i < rows; ++i) {
                sum += v.get_element_as_double(i) * m.get_element_as_double(i, j);
            }
            res_dbl->set(j, sum);
        });
    } else if (auto* res_f32 = dynamic_cast<DenseVector<float>*>(&result)) {
        ParallelFor(0, cols, [&](size_t j) {
            float sum = 0.0f;
            for (uint64_t i = 0; i < rows; ++i) {
                sum += static_cast<float>(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_f32->set(j, sum);
        });
    } else if (auto* res_f16 = dynamic_cast<DenseVector<float16_t>*>(&result)) {
        ParallelFor(0, cols, [&](size_t j) {
            float sum = 0.0f;
            for (uint64_t i = 0; i < rows; ++i) {
                sum += static_cast<float>(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_f16->set(j, float16_t(sum));
        });
    } else {
        throw std::runtime_error("Unsupported result type for vector_matrix_multiply");
    }

    result.set_transposed(true);
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

    const bool needs_metadata_path = (a.get_scalar() != std::complex<double>(1.0, 0.0)) || a.is_conjugated();

    auto* a_cf16 = dynamic_cast<const ComplexFloat16Vector*>(&a);
    auto* res_cf16 = dynamic_cast<ComplexFloat16Vector*>(&result);
    if (a_cf16 && res_cf16) {
        float16_t* r_re = res_cf16->real_data();
        float16_t* r_im = res_cf16->imag_data();
        if (needs_metadata_path) {
            ParallelFor(0, n, [&](size_t i) {
                const std::complex<double> z = a.get_element_as_complex(i) * scalar;
                r_re[i] = float16_t(z.real());
                r_im[i] = float16_t(z.imag());
            });
        } else {
            const float16_t* a_re = a_cf16->real_data();
            const float16_t* a_im = a_cf16->imag_data();
            ParallelFor(0, n, [&](size_t i) {
                r_re[i] = float16_t(static_cast<double>(a_re[i]) * scalar);
                r_im[i] = float16_t(static_cast<double>(a_im[i]) * scalar);
            });
        }
        return;
    }

    auto* a_c32 = dynamic_cast<const DenseVector<std::complex<float>>*>(&a);
    auto* res_c32 = dynamic_cast<DenseVector<std::complex<float>>*>(&result);
    if (a_c32 && res_c32) {
        std::complex<float>* res_data = res_c32->data();
        const float s = static_cast<float>(scalar);
        if (needs_metadata_path) {
            ParallelFor(0, n, [&](size_t i) {
                const std::complex<double> z = a.get_element_as_complex(i) * scalar;
                res_data[i] = std::complex<float>(static_cast<float>(z.real()), static_cast<float>(z.imag()));
            });
        } else {
            const std::complex<float>* a_data = a_c32->data();
            ParallelFor(0, n, [&](size_t i) {
                res_data[i] = a_data[i] * s;
            });
        }
        return;
    }

    auto* a_c64 = dynamic_cast<const DenseVector<std::complex<double>>*>(&a);
    auto* res_c64 = dynamic_cast<DenseVector<std::complex<double>>*>(&result);
    if (a_c64 && res_c64) {
        std::complex<double>* res_data = res_c64->data();
        if (needs_metadata_path) {
            ParallelFor(0, n, [&](size_t i) {
                res_data[i] = a.get_element_as_complex(i) * scalar;
            });
        } else {
            const std::complex<double>* a_data = a_c64->data();
            ParallelFor(0, n, [&](size_t i) {
                res_data[i] = a_data[i] * scalar;
            });
        }
        return;
    }
    
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

    auto* a_f32 = dynamic_cast<const DenseVector<float>*>(&a);
    auto* res_f32 = dynamic_cast<DenseVector<float>*>(&result);
    if (a_f32 && res_f32) {
        const float* a_data = a_f32->data();
        float* res_data = res_f32->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = static_cast<float>(static_cast<double>(a_data[i]) * scalar);
        });
        return;
    }

    auto* a_f16 = dynamic_cast<const DenseVector<float16_t>*>(&a);
    auto* res_f16 = dynamic_cast<DenseVector<float16_t>*>(&result);
    if (a_f16 && res_f16) {
        const float16_t* a_data = a_f16->data();
        float16_t* res_data = res_f16->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = float16_t(static_cast<double>(a_data[i]) * scalar);
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

void CpuSolver::scalar_multiply_vector_complex(const VectorBase& a, std::complex<double> scalar, VectorBase& result) {
    const uint64_t n = a.size();

    const bool needs_metadata_path = (a.get_scalar() != std::complex<double>(1.0, 0.0)) || a.is_conjugated();

    auto* a_cf16 = dynamic_cast<const ComplexFloat16Vector*>(&a);
    auto* res_cf16 = dynamic_cast<ComplexFloat16Vector*>(&result);
    if (a_cf16 && res_cf16) {
        float16_t* r_re = res_cf16->real_data();
        float16_t* r_im = res_cf16->imag_data();
        if (needs_metadata_path) {
            ParallelFor(0, n, [&](size_t i) {
                const std::complex<double> p = a.get_element_as_complex(i) * scalar;
                r_re[i] = float16_t(p.real());
                r_im[i] = float16_t(p.imag());
            });
        } else {
            const float16_t* a_re = a_cf16->real_data();
            const float16_t* a_im = a_cf16->imag_data();
            ParallelFor(0, n, [&](size_t i) {
                const std::complex<double> z(static_cast<double>(a_re[i]), static_cast<double>(a_im[i]));
                const std::complex<double> p = z * scalar;
                r_re[i] = float16_t(p.real());
                r_im[i] = float16_t(p.imag());
            });
        }
        return;
    }

    auto* a_c32 = dynamic_cast<const DenseVector<std::complex<float>>*>(&a);
    auto* res_c32 = dynamic_cast<DenseVector<std::complex<float>>*>(&result);
    if (a_c32 && res_c32) {
        std::complex<float>* res_data = res_c32->data();
        const std::complex<float> s(static_cast<float>(scalar.real()), static_cast<float>(scalar.imag()));
        if (needs_metadata_path) {
            ParallelFor(0, n, [&](size_t i) {
                const std::complex<double> p = a.get_element_as_complex(i) * scalar;
                res_data[i] = std::complex<float>(static_cast<float>(p.real()), static_cast<float>(p.imag()));
            });
        } else {
            const std::complex<float>* a_data = a_c32->data();
            ParallelFor(0, n, [&](size_t i) {
                res_data[i] = a_data[i] * s;
            });
        }
        return;
    }

    auto* a_c64 = dynamic_cast<const DenseVector<std::complex<double>>*>(&a);
    auto* res_c64 = dynamic_cast<DenseVector<std::complex<double>>*>(&result);
    if (a_c64 && res_c64) {
        std::complex<double>* res_data = res_c64->data();
        if (needs_metadata_path) {
            ParallelFor(0, n, [&](size_t i) {
                res_data[i] = a.get_element_as_complex(i) * scalar;
            });
        } else {
            const std::complex<double>* a_data = a_c64->data();
            ParallelFor(0, n, [&](size_t i) {
                res_data[i] = a_data[i] * scalar;
            });
        }
        return;
    }

    throw std::invalid_argument("scalar_multiply_vector_complex requires a complex vector dtype");
}

void CpuSolver::scalar_add_vector(const VectorBase& a, double scalar, VectorBase& result) {
    uint64_t n = a.size();

    // Match VectorBase::add_scalar semantics:
    // - Applies the vector's stored scalar_ factor.
    // - Does NOT apply conjugation (add_scalar uses raw storage).
    const std::complex<double> s_self = a.get_scalar();
    const std::complex<double> add(scalar, 0.0);

    auto* a_cf16 = dynamic_cast<const ComplexFloat16Vector*>(&a);
    auto* res_cf16 = dynamic_cast<ComplexFloat16Vector*>(&result);
    if (a_cf16 && res_cf16) {
        const float16_t* a_re = a_cf16->real_data();
        const float16_t* a_im = a_cf16->imag_data();
        float16_t* r_re = res_cf16->real_data();
        float16_t* r_im = res_cf16->imag_data();
        ParallelFor(0, n, [&](size_t i) {
            const std::complex<double> z(static_cast<double>(a_re[i]), static_cast<double>(a_im[i]));
            const std::complex<double> out = z * s_self + add;
            r_re[i] = float16_t(out.real());
            r_im[i] = float16_t(out.imag());
        });
        return;
    }

    auto* a_c32 = dynamic_cast<const DenseVector<std::complex<float>>*>(&a);
    auto* res_c32 = dynamic_cast<DenseVector<std::complex<float>>*>(&result);
    if (a_c32 && res_c32) {
        const std::complex<float>* a_data = a_c32->data();
        std::complex<float>* res_data = res_c32->data();
        ParallelFor(0, n, [&](size_t i) {
            const std::complex<double> z(static_cast<double>(a_data[i].real()), static_cast<double>(a_data[i].imag()));
            const std::complex<double> out = z * s_self + add;
            res_data[i] = std::complex<float>(static_cast<float>(out.real()), static_cast<float>(out.imag()));
        });
        return;
    }

    auto* a_c64 = dynamic_cast<const DenseVector<std::complex<double>>*>(&a);
    auto* res_c64 = dynamic_cast<DenseVector<std::complex<double>>*>(&result);
    if (a_c64 && res_c64) {
        const std::complex<double>* a_data = a_c64->data();
        std::complex<double>* res_data = res_c64->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = a_data[i] * s_self + add;
        });
        return;
    }
    
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

    auto* a_f32 = dynamic_cast<const DenseVector<float>*>(&a);
    auto* res_f32 = dynamic_cast<DenseVector<float>*>(&result);
    if (a_f32 && res_f32) {
        const float* a_data = a_f32->data();
        float* res_data = res_f32->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = static_cast<float>(static_cast<double>(a_data[i]) + scalar);
        });
        return;
    }

    auto* a_f16 = dynamic_cast<const DenseVector<float16_t>*>(&a);
    auto* res_f16 = dynamic_cast<DenseVector<float16_t>*>(&result);
    if (a_f16 && res_f16) {
        const float16_t* a_data = a_f16->data();
        float16_t* res_data = res_f16->data();
        ParallelFor(0, n, [&](size_t i) {
            res_data[i] = float16_t(static_cast<double>(a_data[i]) + scalar);
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

void CpuSolver::cross_product(const VectorBase& a, const VectorBase& b, VectorBase& result) {
    if (a.size() != 3 || b.size() != 3 || result.size() != 3) {
        throw std::invalid_argument("cross_product only defined for 3D vectors");
    }

    auto* res_dbl = dynamic_cast<DenseVector<double>*>(&result);
    if (!res_dbl) {
        throw std::runtime_error("CpuSolver::cross_product requires DenseVector<double> result");
    }

    const auto* a_dbl = dynamic_cast<const DenseVector<double>*>(&a);
    const auto* b_dbl = dynamic_cast<const DenseVector<double>*>(&b);

    double ax, ay, az, bx, by, bz;

    if (a_dbl) {
        const double* d = a_dbl->data();
        ax = d[0];
        ay = d[1];
        az = d[2];
    } else {
        ax = a.get_element_as_double(0);
        ay = a.get_element_as_double(1);
        az = a.get_element_as_double(2);
    }

    if (b_dbl) {
        const double* d = b_dbl->data();
        bx = d[0];
        by = d[1];
        bz = d[2];
    } else {
        bx = b.get_element_as_double(0);
        by = b.get_element_as_double(1);
        bz = b.get_element_as_double(2);
    }

    res_dbl->set(0, ay * bz - az * by);
    res_dbl->set(1, az * bx - ax * bz);
    res_dbl->set(2, ax * by - ay * bx);
}

std::unique_ptr<TriangularMatrix<double>> CpuSolver::compute_k_matrix(
    const TriangularMatrix<bool>& C,
    double a,
    const std::string& output_path,
    int num_threads
) {
    (void)num_threads; // ParallelFor uses global ThreadPool settings

    const uint64_t n = C.rows();
    auto K = std::make_unique<TriangularMatrix<double>>(n, output_path);

    const char* c_raw_bytes = reinterpret_cast<const char*>(C.data());
    char* k_base_ptr = reinterpret_cast<char*>(K->data());

    ParallelFor(0, n, [&](size_t j) {
        std::vector<double> col_j(j + 1, 0.0);

        for (int64_t i = static_cast<int64_t>(j) - 1; i >= 0; --i) {
            double sum = 0.0;

            const uint64_t row_offset = C.get_row_offset(static_cast<uint64_t>(i));
            const uint64_t* row_ptr = reinterpret_cast<const uint64_t*>(c_raw_bytes + row_offset);

            if (j > static_cast<uint64_t>(i) + 1) {
                const uint64_t max_bit_index = j - static_cast<uint64_t>(i) - 2;
                const uint64_t num_words = (max_bit_index / 64) + 1;

                for (uint64_t w = 0; w < num_words; ++w) {
                    uint64_t word = row_ptr[w];
                    if (word == 0) continue;

                    if (w == num_words - 1) {
                        const uint64_t bits_in_last_word = (max_bit_index % 64) + 1;
                        if (bits_in_last_word < 64) {
                            const uint64_t mask = (1ULL << bits_in_last_word) - 1;
                            word &= mask;
                        }
                    }

                    while (word != 0) {
                        const int bit = std::countr_zero(word);
                        const uint64_t m = (static_cast<uint64_t>(i) + 1) + (w * 64 + static_cast<uint64_t>(bit));
                        if (m < j) {
                            sum += col_j[m];
                        }
                        word &= (word - 1);
                    }
                }
            }

            const uint64_t bit_offset_j = j - (static_cast<uint64_t>(i) + 1);
            const uint64_t word_idx_j = bit_offset_j / 64;
            const uint64_t bit_idx_j = bit_offset_j % 64;
            const uint64_t word_j = row_ptr[word_idx_j];
            const bool c_ij = ((word_j >> bit_idx_j) & 1ULL) != 0;

            const double val = ((c_ij ? 1.0 : 0.0) - sum) / a;
            col_j[static_cast<uint64_t>(i)] = val;

            const uint64_t k_row_offset = K->get_row_offset(static_cast<uint64_t>(i));
            const uint64_t k_col_idx = j - (static_cast<uint64_t>(i) + 1);
            double* k_row_ptr = reinterpret_cast<double*>(k_base_ptr + k_row_offset);
            k_row_ptr[k_col_idx] = val;
        }
    });

    return K;
}

double CpuSolver::frobenius_norm(const MatrixBase& m) {
    const uint64_t rows = m.rows();
    const uint64_t cols = m.cols();
    if (rows == 0 || cols == 0) return 0.0;

    const auto dt = m.get_data_type();
    const uint64_t total = rows * cols;
    if (dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64) {
        debug_trace::set_last("cpu.frobenius_norm.complex");
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (int64_t idx = 0; idx < static_cast<int64_t>(total); ++idx) {
            const uint64_t uidx = static_cast<uint64_t>(idx);
            const uint64_t i = uidx / cols;
            const uint64_t j = uidx % cols;
            const std::complex<double> z = m.get_element_as_complex(i, j);
            sum += std::norm(z);
        }
        return std::sqrt(sum);
    }

    debug_trace::set_last("cpu.frobenius_norm.real");
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int64_t idx = 0; idx < static_cast<int64_t>(total); ++idx) {
        const uint64_t uidx = static_cast<uint64_t>(idx);
        const uint64_t i = uidx / cols;
        const uint64_t j = uidx % cols;
        const double x = m.get_element_as_double(i, j);
        sum += x * x;
    }
    return std::sqrt(sum);
}

std::complex<double> CpuSolver::sum(const MatrixBase& m) {
    const uint64_t rows = m.rows();
    const uint64_t cols = m.cols();
    if (rows == 0 || cols == 0) {
        return {0.0, 0.0};
    }

    const auto dt = m.get_data_type();
    const bool complex =
        (dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64);

    if (complex) {
        debug_trace::set_last("cpu.sum.matrix.complex");
        double re = 0.0;
        double im = 0.0;
        #pragma omp parallel for reduction(+:re, im)
        for (int64_t i = 0; i < static_cast<int64_t>(rows); ++i) {
            for (uint64_t j = 0; j < cols; ++j) {
                const std::complex<double> z = m.get_element_as_complex(static_cast<uint64_t>(i), j);
                re += z.real();
                im += z.imag();
            }
        }
        return {re, im};
    }

    debug_trace::set_last("cpu.sum.matrix.real");
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int64_t i = 0; i < static_cast<int64_t>(rows); ++i) {
        for (uint64_t j = 0; j < cols; ++j) {
            sum += m.get_element_as_double(static_cast<uint64_t>(i), j);
        }
    }
    return {sum, 0.0};
}

double CpuSolver::trace(const MatrixBase& matrix) {
    const uint64_t rows = matrix.rows();
    const uint64_t cols = matrix.cols();
    const uint64_t n = std::min(rows, cols);

    const auto type = matrix.get_matrix_type();
    if (type == MatrixType::IDENTITY) {
        return matrix.get_scalar().real() * static_cast<double>(n);
    }

    const auto dt = matrix.get_data_type();
    const bool is_complex = (dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64);

    double tr = 0.0;
    if (is_complex) {
        for (uint64_t i = 0; i < n; ++i) {
            tr += matrix.get_element_as_complex(i, i).real();
        }
    } else {
        for (uint64_t i = 0; i < n; ++i) {
            tr += matrix.get_element_as_double(i, i);
        }
    }
    return tr;
}

double CpuSolver::determinant(const MatrixBase& matrix) {
    const uint64_t rows = matrix.rows();
    const uint64_t cols = matrix.cols();
    if (rows != cols) {
        throw std::runtime_error("Determinant requires square matrix");
    }

    const uint64_t n = rows;
    const auto type = matrix.get_matrix_type();
    const auto dt = matrix.get_data_type();
    const bool is_complex = (dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64);

    if (type == MatrixType::IDENTITY) {
        return std::pow(matrix.get_scalar(), n).real();
    }

    if (type == MatrixType::DIAGONAL) {
        if (is_complex) {
            std::complex<double> det = 1.0;
            for (uint64_t i = 0; i < n; ++i) {
                det *= matrix.get_element_as_complex(i, i);
            }
            return det.real();
        }

        double det = 1.0;
        for (uint64_t i = 0; i < n; ++i) {
            det *= matrix.get_element_as_double(i, i);
        }
        return det;
    }

    if (type == MatrixType::TRIANGULAR_FLOAT || type == MatrixType::CAUSAL) {
        bool has_diag = false;
        if (auto* m = dynamic_cast<const TriangularMatrix<double>*>(&matrix)) {
            has_diag = m->has_diagonal();
        } else if (auto* m = dynamic_cast<const TriangularMatrix<int32_t>*>(&matrix)) {
            has_diag = m->has_diagonal();
        }

        if (!has_diag) {
            return 0.0;
        }

        if (is_complex) {
            std::complex<double> det = 1.0;
            for (uint64_t i = 0; i < n; ++i) {
                det *= matrix.get_element_as_complex(i, i);
            }
            return det.real();
        }

        double det = 1.0;
        for (uint64_t i = 0; i < n; ++i) {
            det *= matrix.get_element_as_double(i, i);
        }
        return det;
    }

    // General case: LU determinant.
    // If there is any complex scalar/component, compute using complex LU and return the real part.
    const bool use_complex = is_complex || std::abs(matrix.get_scalar().imag()) > 1e-14;

    if (!use_complex) {
        auto data = to_memory_flat_real_square(matrix);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_eigen(
            data.data(), static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

        Eigen::PartialPivLU<Eigen::MatrixXd> lu(mat_eigen);
        return lu.determinant();
    }

    auto data = to_memory_flat_complex_square(matrix);
    Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_eigen(
        data.data(), static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    Eigen::PartialPivLU<Eigen::MatrixXcd> lu(mat_eigen);
    return lu.determinant().real();
}

void CpuSolver::qr(const MatrixBase& in, MatrixBase& Q, MatrixBase& R) {
    // Keep the same contract as DenseMatrix<double>::qr: square float64 matrices.
    if (in.rows() != in.cols()) {
        throw std::runtime_error("QR requires a square matrix");
    }

    auto* q_out = dynamic_cast<DenseMatrix<double>*>(&Q);
    auto* r_out = dynamic_cast<DenseMatrix<double>*>(&R);
    if (!q_out || !r_out) {
        throw std::runtime_error("CpuSolver::qr requires DenseMatrix<double> outputs");
    }

    const uint64_t n = in.rows();
    if (q_out->rows() != n || q_out->cols() != n || r_out->rows() != n || r_out->cols() != n) {
        throw std::runtime_error("CpuSolver::qr output matrix dimension mismatch");
    }

    double* q_data = q_out->data();
    double* r_data = r_out->data();

    // Initialize Q from input.
    ParallelFor(0, n, [&](size_t i) {
        for (size_t j = 0; j < n; ++j) {
            q_data[i * n + j] = in.get_element_as_double(static_cast<uint64_t>(i), static_cast<uint64_t>(j));
        }
    });

    std::fill(r_data, r_data + n * n, 0.0);

    for (size_t k = 0; k < n; ++k) {
        double norm_sq = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double val = q_data[i * n + k];
            norm_sq += val * val;
        }

        const double norm = std::sqrt(norm_sq);
        r_data[k * n + k] = norm;

        if (norm > 1e-12) {
            const double inv_norm = 1.0 / norm;
            ParallelFor(0, n, [&](size_t i) { q_data[i * n + k] *= inv_norm; });
        } else {
            ParallelFor(0, n, [&](size_t i) { q_data[i * n + k] = 0.0; });
        }

        ParallelFor(k + 1, n, [&](size_t j) {
            double dot = 0.0;
            for (size_t i = 0; i < n; ++i) {
                dot += q_data[i * n + k] * q_data[i * n + j];
            }
            r_data[k * n + j] = dot;

            for (size_t i = 0; i < n; ++i) {
                q_data[i * n + j] -= dot * q_data[i * n + k];
            }
        });
    }
}

} // namespace pycauset

