// CpuSolver: matmul kernels and gemm (split from CpuSolver.cpp, review 4.2).
#include "pycauset/core/ScalarUtils.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/math/ViewRead.hpp"
#include "pycauset/compute/cpu/CpuSolver.hpp"
#include <complex>
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
// NOTE: the bundled OpenBLAS is the standard LP64 (32-bit lapack_int) build
// (verified: DLL exports dgemm_/dgesv_/dgetrf_, NOT the ILP64 *_64_ symbols).
// Therefore `lapack_int` (int) matches the DLL ABI and LAPACK_ILP64 MUST stay
// undefined. Switching to an ILP64 OpenBLAS would require defining LAPACK_ILP64
// here AND rebuilding, otherwise every LAPACK call would read a 64-bit int
// where a 32-bit one was written (silent corruption).

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
#include "pycauset/math/VectorOps.hpp"
#include "pycauset/math/ViewValue.hpp"
#include "pycauset/core/Nvtx.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <bit>
#include <filesystem>
#include <vector>
#include <cstring>
#include <atomic>
#include <functional>
#include "pycauset/matrix/BitClosureMatrix.hpp"
#include "pycauset/core/DTypeDispatch.hpp"
#include "pycauset/math/ScalarOps.hpp"
#include <iostream>
#include <limits>
#include <random>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "CpuKernelsCommon.hpp"

namespace pycauset {

void set_num_threads(size_t n) {
    ThreadPool::set_num_threads(n);
    openblas_set_num_threads(static_cast<int>(ThreadPool::get_num_threads()));
}

void init_thread_budget() {
    const int n = static_cast<int>(ThreadPool::get_num_threads());
    // A limit from the user (OPENBLAS_NUM_THREADS / OMP_NUM_THREADS) is the BLAS budget too;
    // otherwise LAPACK kernels default to at most 8 threads.
    openblas_set_num_threads(ThreadPool::threads_limited() ? n : std::min(n, 8));
}

int gemm_thread_count() {
    const int n = static_cast<int>(ThreadPool::get_num_threads());
    // Capped at 20: asking OpenBLAS for more threads than it started with grows its pool at
    // run time, which deadlocked on a 3-vCPU macOS runner.
    return ThreadPool::threads_limited() ? n : std::min(n, 20);
}
using namespace cpu_detail;

namespace {
    // Runtime CPU detection for AVX-512 VPOPCNTDQ
    bool has_avx512_vpopcntdq() { return cpu_has_avx512_vpopcntdq(); }  // CpuKernelsCommon.hpp

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
    // Exact integer accumulation for the matmul kernels: 128 bits hold any 64 x 64-bit product.
    inline __int128 checked_add_i128(__int128 a, __int128 b) {
        __int128 out;
        if (__builtin_add_overflow(a, b, &out)) throw std::overflow_error("Integer matmul overflow: accumulator overflow");
        return out;
    }
    template <typename T>
    inline void check_fits(__int128 v) {
        if (v < static_cast<__int128>((std::numeric_limits<T>::min)()) || v > static_cast<__int128>((std::numeric_limits<T>::max)())) {
            throw std::overflow_error("Integer matmul overflow: result does not fit in output dtype");
        }
    }

    inline int64_t checked_add_int64(int64_t a, int64_t b) {
        if ((b > 0 && a > (std::numeric_limits<int64_t>::max)() - b) ||
            (b < 0 && a < (std::numeric_limits<int64_t>::min)() - b)) {
            throw std::overflow_error("Integer matmul overflow: accumulator overflow");
        }
        return a + b;
    }

    // Popcount-based dense bit x dense bit matmul producing int32. Both operands
    // must be non-transposed with uniform dense column indexing.
    inline void bit_popcount_matmul(const DenseMatrix<bool>& a, const DenseMatrix<bool>& b,
                                    DenseMatrix<int32_t>& c) {
        const uint64_t m = a.rows();
        const uint64_t k = a.cols();
        const uint64_t n = b.cols();
        const uint64_t* a_data = a.data();
        const uint64_t words_per_row = a.stride_bytes() / 8;
        const uint64_t rem_bits = k % 64u;
        const uint64_t last_mask = (rem_bits == 0u) ? ~0ull : ((1ull << rem_bits) - 1ull);

        auto b_transposed_mat = std::make_unique<DenseBitMatrix>(n, k, "");
        uint64_t* b_transposed_data = b_transposed_mat->data();

        for (uint64_t i = 0; i < k; ++i) {
            for (uint64_t j = 0; j < n; ++j) {
                if (b.get(i, j)) {
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
                if (use_avx512) {
#if defined(__x86_64__) || defined(_M_X64)
                    const uint64_t avx_limit = words_per_row & ~7ULL;
                    dot_product = dot_product_avx512(a_row, b_col, avx_limit);
                    w = avx_limit;
#endif
                }
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
                c.set(static_cast<uint64_t>(i), j, static_cast<int32_t>(dot_product));
            }
        });
    }

}  // namespace

namespace {
    // --- Helper: Direct Path Optimization ---
    // Tries to pin all matrices and run BLAS directly.
    // Returns true if successful, false if memory budget didn't allow it.
    template <typename T>
    bool attempt_direct_path(const DenseMatrix<T>* a_dense, const DenseMatrix<T>* b_dense, DenseMatrix<T>* c_dense) {
        NVTX_RANGE("CpuSolver::attempt_direct_path");
        // Only supported for Float/Double where we have BLAS
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
            // BLAS on the operands' storage, mapped or not (out of core the OS pages; a RAM
            // check here chose between this and the same dgemm further down, review 3.5 C-6).
            // No pinning (VirtualLock): it cost ~1-2 ms per call at n = 1024 for nothing on CPU.
            {
                const T* a_data = a_dense->data() + (a_dense->row_offset() * a_dense->base_cols() + a_dense->col_offset());
                const T* b_data = b_dense->data() + (b_dense->row_offset() * b_dense->base_cols() + b_dense->col_offset());
                T* c_data = c_dense->data() + (c_dense->row_offset() * c_dense->base_cols() + c_dense->col_offset());

                bool t_a = a_dense->is_transposed();
                bool t_b = b_dense->is_transposed();

                const int M = static_cast<int>(a_dense->rows());
                const int N = static_cast<int>(b_dense->cols());
                const int K = static_cast<int>(a_dense->cols());
                const int lda = static_cast<int>(a_dense->base_cols());
                const int ldb = static_cast<int>(b_dense->base_cols());
                const int ldc = static_cast<int>(c_dense->base_cols());

                // Matrix-vector (N == 1) and vector-matrix (M == 1): BLAS gemv. OpenBLAS 0.3.26
                // runs ?gemm with one column through the full GEMM packing (5x slower than
                // gemv; newer versions forward it themselves).
                if (!c_dense->is_transposed() && (N == 1 || M == 1)) {
                    auto gemv = [&](bool trans, int rows, int cols, const T* mat, int ld, const T* x, int incx,
                                    T* y, int incy) {
                        const auto t = trans ? CblasTrans : CblasNoTrans;
                        if constexpr (std::is_same_v<T, double>) cblas_dgemv(CblasRowMajor, t, rows, cols, 1.0, mat, ld, x, incx, 0.0, y, incy);
                        else cblas_sgemv(CblasRowMajor, t, rows, cols, 1.0f, mat, ld, x, incx, 0.0f, y, incy);
                    };
                    if (N == 1) {  // y (M) = op(A) x, x = B's column, y = C's column
                        const int incx = t_b ? 1 : ldb;
                        if (!t_a) gemv(false, M, K, a_data, lda, b_data, incx, c_data, ldc);
                        else gemv(true, K, M, a_data, lda, b_data, incx, c_data, ldc);
                    } else {  // M == 1: y (N) = B^T x, x = A's row, y = C's row
                        const int incx = t_a ? lda : 1;
                        if (!t_b) gemv(true, K, N, b_data, ldb, a_data, incx, c_data, 1);
                        else gemv(false, N, K, b_data, ldb, a_data, incx, c_data, 1);
                    }
                    c_dense->set_scalar(compose_factors(a_dense->get_scalar(), b_dense->get_scalar()));
                    return true;
                }

                // GEMM scales with threads, small LAPACK factorizations do not: the OpenBLAS
                // default stays low (init_thread_budget) and a GEMM gets gemm_thread_count(),
                // which is the user's budget when they set one (it was every hardware thread
                // whatever OPENBLAS_NUM_THREADS said, review 3.4 C-6). No set call when the
                // count already matches (no pool reconfiguration).
                const int prev_threads = openblas_get_num_threads();
                const int gemm_threads = gemm_thread_count();
                if (gemm_threads != prev_threads) openblas_set_num_threads(gemm_threads);
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
                if (gemm_threads != prev_threads) openblas_set_num_threads(prev_threads);

                c_dense->set_scalar(compose_factors(a_dense->get_scalar(), b_dense->get_scalar()));
                return true;
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

        const T* a_data = a_dense->data() + (a_dense->row_offset() * a_dense->base_cols() + a_dense->col_offset());
        const T* b_data = b_dense->data() + (b_dense->row_offset() * b_dense->base_cols() + b_dense->col_offset());
        T* c_data = c_dense->data() + (c_dense->row_offset() * c_dense->base_cols() + c_dense->col_offset());

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

        // --- Integer path: exact 128-bit accumulator + overflow check into T ---
        // Policy: never silently wrap, never silently widen output storage. A product of two
        // 64-bit values always fits in 128 bits; the sum is checked; the result must fit T.
        // (An int64 accumulator multiplied unchecked: [[2**31]] @ [[2**32]] wrapped to -2**63,
        // and uint64 values >= 2**63 were read as negative; review 3.6 C-1.)
        if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
            size_t block_size = 64;
            const __int128 lo = static_cast<__int128>((std::numeric_limits<T>::min)());
            const __int128 hi = static_cast<__int128>((std::numeric_limits<T>::max)());

            ParallelBlockMap(m, n, block_size, [&](size_t i_start, size_t i_end, size_t j_start, size_t j_end) {
                std::vector<__int128> sums(j_end - j_start);

                for (size_t i = i_start; i < i_end; ++i) {
                    for (size_t jj = 0; jj < sums.size(); ++jj) {
                        sums[jj] = 0;
                    }
                    for (size_t k_start = 0; k_start < k; k_start += block_size) {
                        const size_t k_end = std::min(k_start + block_size, static_cast<size_t>(k));
                        for (size_t kk = k_start; kk < k_end; ++kk) {
                            const __int128 val_a = static_cast<__int128>(a_data[idx_a(static_cast<uint64_t>(i), static_cast<uint64_t>(kk))]);
                            if (val_a == 0) continue;
                            for (size_t jj = 0; jj < sums.size(); ++jj) {
                                const uint64_t j = static_cast<uint64_t>(j_start + jj);
                                const __int128 val_b = static_cast<__int128>(b_data[idx_b(static_cast<uint64_t>(kk), j)]);
                                if (__builtin_add_overflow(sums[jj], val_a * val_b, &sums[jj])) {
                                    throw std::overflow_error("Integer matmul overflow: accumulator overflow");
                                }
                            }
                        }
                    }

                    for (size_t jj = 0; jj < sums.size(); ++jj) {
                        const __int128 s = sums[jj];
                        if (s < lo || s > hi) {
                            throw std::overflow_error("Integer matmul overflow: result does not fit in output dtype");
                        }
                        c_data[idx_c(static_cast<uint64_t>(i), static_cast<uint64_t>(j_start + jj))] = static_cast<T>(s);
                    }
                }
            });

            c_dense->set_scalar(compose_factors(a_dense->get_scalar(), b_dense->get_scalar()));
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
                c_dense->set_scalar(compose_factors(a_dense->get_scalar(), b_dense->get_scalar()));
                return;
            }
        }
        
        if constexpr (std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>>) {
            if (!t_c) {
                const T one(1), zero(0);
                const auto ta = t_a ? CblasTrans : CblasNoTrans;
                const auto tb = t_b ? CblasTrans : CblasNoTrans;
                if constexpr (std::is_same_v<T, std::complex<double>>) {
                    cblas_zgemm(CblasRowMajor, ta, tb, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                                &one, a_data, static_cast<int>(lda), b_data, static_cast<int>(ldb), &zero, c_data,
                                static_cast<int>(ldc));
                } else {
                    cblas_cgemm(CblasRowMajor, ta, tb, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                                &one, a_data, static_cast<int>(lda), b_data, static_cast<int>(ldb), &zero, c_data,
                                static_cast<int>(ldc));
                }
                c_dense->set_scalar(compose_factors(a_dense->get_scalar(), b_dense->get_scalar()));
                return;
            }
        }

        // Initialize result to 0
        for (uint64_t i = 0; i < m; ++i) {
            for (uint64_t j = 0; j < n; ++j) {
                c_data[idx_c(i, j)] = static_cast<T>(0);
            }
        }
        
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
        
        c_dense->set_scalar(compose_factors(a_dense->get_scalar(), b_dense->get_scalar()));
    }

    // Specializations for Double/Float are now handled by the template above
    // which includes the Direct Path optimization.

}  // namespace

namespace {
// Exact integer matmul through float64 BLAS (F-055). If every |a| <= A, |b| <= B and
// k * A * B < 2^53, every product and partial sum is an integer that double represents
// exactly, so dgemm returns the exact result in any summation order. The bound is MEASURED
// on the operands (an O(n^2) pass next to the O(n^3) product), never estimated. Returns
// false (nothing written) when not applicable; overflow of the output dtype raises as in the
// integer kernels.
template <typename R>
bool int_matmul_via_dgemm(const MatrixBase& a, const MatrixBase& b, DenseMatrix<R>& c) {
    const uint64_t m = a.rows(), k = a.cols(), n = b.cols();
    if (m == 0 || n == 0 || k == 0) return false;
    const size_t bytes = static_cast<size_t>((m * k + k * n + m * n) * sizeof(double));
    if (!pycauset::core::MemoryGovernor::instance().can_fit_in_ram(bytes)) return false;
    StridedReader<double> ra, rb;
    if (!make_strided_reader(a, m, k, ra) || !make_strided_reader(b, k, n, rb)) return false;
    std::vector<double> A(m * k), B(k * n), C(m * n);
    std::vector<double> amax(m, 0.0), bmax(k, 0.0);
    ParallelFor(0, m, [&](size_t i) {
        ra.row(i, 0, k, A.data() + i * k);
        double mx = 0.0;
        for (uint64_t q = 0; q < k; ++q) mx = std::max(mx, std::abs(A[i * k + q]));
        amax[i] = mx;
    });
    ParallelFor(0, k, [&](size_t i) {
        rb.row(i, 0, n, B.data() + i * n);
        double mx = 0.0;
        for (uint64_t q = 0; q < n; ++q) mx = std::max(mx, std::abs(B[i * n + q]));
        bmax[i] = mx;
    });
    double ma = 0.0, mb = 0.0;
    for (double v : amax) ma = std::max(ma, v);
    for (double v : bmax) mb = std::max(mb, v);
    // Integer-valued doubles: products of exact integers below 2^53 are exact; the bound
    // covers the largest partial sum.
    if (static_cast<long double>(k) * ma * mb >= 9007199254740992.0L) return false;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(m), static_cast<int>(n),
                static_cast<int>(k), 1.0, A.data(), static_cast<int>(k), B.data(), static_cast<int>(n), 0.0,
                C.data(), static_cast<int>(n));
    const double lo = static_cast<double>((std::numeric_limits<R>::lowest)());
    const double hi = static_cast<double>((std::numeric_limits<R>::max)());
    R* dst = c.data();
    std::atomic<bool> overflow{false};
    ParallelFor(0, m * n, [&](size_t idx) {
        const double v = C[idx];
        if (v < lo || v > hi) overflow.store(true, std::memory_order_relaxed);
        else dst[idx] = static_cast<R>(v);
    });
    if (overflow.load()) throw std::overflow_error("Integer matmul overflow: result does not fit in output dtype");
    c.set_scalar(1.0);
    return true;
}

template <typename R>
bool try_int_matmul_via_dgemm(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    auto* c = dynamic_cast<DenseMatrix<R>*>(&result);
    if (c == nullptr || c->is_transposed() || c->is_subview() || c->rows() != c->base_rows() ||
        c->cols() != c->base_cols()) return false;
    return int_matmul_via_dgemm<R>(a, b, *c);
}
}  // namespace

namespace {
// --- Matmul kernels (Phase N.1). Each states the inputs it accepts; dispatch_matmul
// (src/math/LinearAlgebra.cpp) normalises every other operand to one of these contracts.
// Common to all: no window (subview), no pending conjugation of complex values. Transposition
// and scalar factors are read by the kernels (the result scalar is the product of the
// operand scalars).

// Dense: DenseMatrix<T> x DenseMatrix<T> -> DenseMatrix<T>, T a float, complex or integer type.
template <typename T>
bool dense_kernel(const MatrixBase& a, const MatrixBase& b, MatrixBase& result, const char* tag) {
    auto* A = dynamic_cast<const DenseMatrix<T>*>(&a);
    auto* B = dynamic_cast<const DenseMatrix<T>*>(&b);
    auto* C = dynamic_cast<DenseMatrix<T>*>(&result);
    if (!A || !B || !C) return false;
    if constexpr (std::is_integral_v<T>) {
        if (try_int_matmul_via_dgemm<T>(a, b, result)) {
            debug_trace::set_last("cpu.matmul.int_via_dgemm");
            return true;
        }
    }
    debug_trace::set_last(tag);
    matmul_impl(A, B, C);
    return true;
}

// Triangular: upper TriangularMatrix<T> x upper TriangularMatrix<T> -> upper TriangularMatrix<T>,
// T a real non-bit, non-float16 type. Row i of the product sums rows k >= i of B; only the
// stored triangles are read. Integers accumulate in int64 and raise on overflow of T.
template <typename T>
bool triangular_kernel(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    auto* A = dynamic_cast<const TriangularMatrix<T>*>(&a);
    auto* B = dynamic_cast<const TriangularMatrix<T>*>(&b);
    auto* C = dynamic_cast<TriangularMatrix<T>*>(&result);
    if (!A || !B || !C) return false;
    if (A->is_transposed() || B->is_transposed() || C->is_transposed()) {
        throw std::invalid_argument("CpuSolver::matmul: triangular kernel takes upper (untransposed) operands");
    }
    debug_trace::set_last("cpu.matmul.triangular");
    using Acc = std::conditional_t<std::is_integral_v<T>, __int128, double>;  // exact (review 3.6 C-1)
    const uint64_t n = A->rows();
    const bool a_diag = A->has_diagonal(), b_diag = B->has_diagonal(), c_diag = C->has_diagonal();
    const char* a_base = reinterpret_cast<const char*>(A->data());
    const char* b_base = reinterpret_cast<const char*>(B->data());
    char* c_base = reinterpret_cast<char*>(C->data());
    ParallelFor(0, n, [&](size_t i) {
        std::vector<Acc> acc(n, Acc(0));
        const uint64_t k0 = a_diag ? i : i + 1;
        const T* a_row = reinterpret_cast<const T*>(a_base + A->get_row_offset(i));
        for (uint64_t k = k0; k < n; ++k) {
            const Acc va = static_cast<Acc>(a_row[k - k0]);
            if constexpr (std::is_integral_v<T>) {
                if (va == 0) continue;  // exact: integer zeros contribute nothing
            }
            const uint64_t j0 = b_diag ? k : k + 1;
            const T* b_row = reinterpret_cast<const T*>(b_base + B->get_row_offset(k));
            for (uint64_t j = j0; j < n; ++j) {
                if constexpr (std::is_integral_v<T>) {
                    acc[j] = checked_add_i128(acc[j], va * static_cast<Acc>(b_row[j - j0]));
                } else {
                    acc[j] += va * static_cast<Acc>(b_row[j - j0]);
                }
            }
        }
        const uint64_t c0 = c_diag ? i : i + 1;
        T* c_row = reinterpret_cast<T*>(c_base + C->get_row_offset(i));
        for (uint64_t j = c0; j < n; ++j) {
            if constexpr (std::is_integral_v<T>) check_fits<T>(acc[j]);
            c_row[j - c0] = static_cast<T>(acc[j]);
        }
    });
    C->set_scalar(compose_factors(A->get_scalar(), B->get_scalar()));
    return true;
}

// Bit x dense: DenseMatrix<bool> x DenseMatrix<T> -> DenseMatrix<T>. Keeps A bit-packed and
// adds the rows of B selected by its set bits (scale-first: A is never widened).
template <typename T>
bool bit_x_dense_kernel(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    auto* A = dynamic_cast<const DenseMatrix<bool>*>(&a);
    auto* B = dynamic_cast<const DenseMatrix<T>*>(&b);
    auto* C = dynamic_cast<DenseMatrix<T>*>(&result);
    if (!A || !B || !C) return false;
    if (C->is_transposed()) throw std::invalid_argument("CpuSolver::matmul: result must not be transposed");
    debug_trace::set_last("cpu.matmul.bit_x_dense");
    using Acc = std::conditional_t<std::is_integral_v<T>, __int128, T>;  // exact (review 3.6 C-1)
    const uint64_t m = A->rows(), k = A->cols(), n = B->cols();
    const bool a_packed = !A->is_transposed();
    const bool b_rows = !B->is_transposed();
    const uint64_t words = A->stride_bytes() / 8;
    const uint64_t* a_data = A->data();
    const T* b_data = B->data();
    const uint64_t ldb = B->base_cols();
    T* c_data = C->data();
    const uint64_t ldc = C->base_cols();
    ParallelFor(0, m, [&](size_t i) {
        std::vector<Acc> acc(n, Acc(0));
        auto add_row = [&](uint64_t kk) {
            for (uint64_t j = 0; j < n; ++j) {
                const Acc v = static_cast<Acc>(b_rows ? b_data[kk * ldb + j] : b_data[j * ldb + kk]);
                if constexpr (std::is_integral_v<T>) acc[j] = checked_add_i128(acc[j], v);
                else acc[j] += v;
            }
        };
        if (a_packed) {
            const uint64_t* row = a_data + i * words;
            for (uint64_t w = 0; w < words; ++w) {
                for (uint64_t word = row[w]; word; word &= word - 1) {
                    const uint64_t kk = w * 64 + static_cast<uint64_t>(std::countr_zero(word));
                    if (kk < k) add_row(kk);
                }
            }
        } else {
            for (uint64_t kk = 0; kk < k; ++kk) if (A->get(i, kk)) add_row(kk);
        }
        for (uint64_t j = 0; j < n; ++j) {
            if constexpr (std::is_integral_v<T>) check_fits<T>(acc[j]);
            c_data[i * ldc + j] = static_cast<T>(acc[j]);
        }
    });
    C->set_scalar(compose_factors(A->get_scalar(), B->get_scalar()));
    return true;
}

// Bit x bit: DenseMatrix<bool> x DenseMatrix<bool> -> DenseMatrix<int32_t> (path counts).
bool bit_x_bit_kernel(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    auto* A = dynamic_cast<const DenseMatrix<bool>*>(&a);
    auto* B = dynamic_cast<const DenseMatrix<bool>*>(&b);
    auto* C = dynamic_cast<DenseMatrix<int32_t>*>(&result);
    if (!A || !B || !C) return false;
    debug_trace::set_last("cpu.matmul.bitbit_popcount");
    // The popcount kernel reads rows of A and columns of B as packed words: a transposed
    // operand gets one typed O(n^2) plain copy (the per-element O(n^3) branch this replaced
    // took 104 ms vs NumPy 1.4 ms at n = 384, F-083).
    auto plain = [](const DenseMatrix<bool>& X) {
        auto out = std::make_unique<DenseMatrix<bool>>(X.rows(), X.cols(), std::string());
        for (uint64_t i = 0; i < X.rows(); ++i)
            for (uint64_t j = 0; j < X.cols(); ++j)
                if (X.get(i, j)) out->set(i, j, true);
        return out;
    };
    std::unique_ptr<DenseMatrix<bool>> ha = A->is_transposed() ? plain(*A) : nullptr;
    std::unique_ptr<DenseMatrix<bool>> hb = B->is_transposed() ? plain(*B) : nullptr;
    bit_popcount_matmul(ha ? *ha : *A, hb ? *hb : *B, *C);
    C->set_scalar(compose_factors(A->get_scalar(), B->get_scalar()));
    return true;
}
}  // namespace

void CpuSolver::matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    if (a.cols() != b.rows() || result.rows() != a.rows() || result.cols() != b.cols()) {
        throw std::invalid_argument("Dimension mismatch");
    }
    require_blas_dims({a.rows(), a.cols(), b.cols()}, "matmul");
    auto viewed = [](const MatrixBase& m) {
        const DataType d = m.get_data_type();
        const bool cplx = is_complex_dtype(d);
        return m.is_subview() || (m.is_conjugated() && cplx);
    };
    if (viewed(a) || viewed(b) || result.is_subview()) {
        throw std::invalid_argument("CpuSolver::matmul: windows and conjugated operands are written out by "
                                    "dispatch_matmul, not passed to the kernels");
    }

    if (dense_kernel<double>(a, b, result, "cpu.matmul.f64") ||
        dense_kernel<float>(a, b, result, "cpu.matmul.f32") ||
        dense_kernel<std::complex<double>>(a, b, result, "cpu.matmul.c64") ||
        dense_kernel<std::complex<float>>(a, b, result, "cpu.matmul.c32") ||
        dense_kernel<int8_t>(a, b, result, "cpu.matmul.i8") ||
        dense_kernel<int16_t>(a, b, result, "cpu.matmul.i16") ||
        dense_kernel<int32_t>(a, b, result, "cpu.matmul.i32") ||
        dense_kernel<int64_t>(a, b, result, "cpu.matmul.i64") ||
        dense_kernel<uint8_t>(a, b, result, "cpu.matmul.u8") ||
        dense_kernel<uint16_t>(a, b, result, "cpu.matmul.u16") ||
        dense_kernel<uint32_t>(a, b, result, "cpu.matmul.u32") ||
        dense_kernel<uint64_t>(a, b, result, "cpu.matmul.u64")) {
        return;
    }
    if (triangular_kernel<double>(a, b, result) || triangular_kernel<float>(a, b, result) ||
        triangular_kernel<int8_t>(a, b, result) || triangular_kernel<int16_t>(a, b, result) ||
        triangular_kernel<int32_t>(a, b, result) || triangular_kernel<int64_t>(a, b, result) ||
        triangular_kernel<uint8_t>(a, b, result) || triangular_kernel<uint16_t>(a, b, result) ||
        triangular_kernel<uint32_t>(a, b, result) || triangular_kernel<uint64_t>(a, b, result)) {
        return;
    }
    if (bit_x_bit_kernel(a, b, result) || bit_x_dense_kernel<double>(a, b, result) ||
        bit_x_dense_kernel<float>(a, b, result) || bit_x_dense_kernel<int32_t>(a, b, result) ||
        bit_x_dense_kernel<int16_t>(a, b, result)) {
        return;
    }
    throw std::invalid_argument(std::string("CpuSolver::matmul: no kernel for ") + dtype_name(a.get_data_type()) +
                                " x " + dtype_name(b.get_data_type()) + " -> " +
                                dtype_name(result.get_data_type()) +
                                " (dispatch_matmul normalises operands to a kernel contract)");
}


} // namespace pycauset
