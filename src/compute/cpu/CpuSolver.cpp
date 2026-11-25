#include <Spectra/SymEigsSolver.h>
#include <Spectra/GenEigsSolver.h>
#include "pycauset/core/ScalarUtils.hpp"
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
#include "pycauset/core/Nvtx.hpp"
#include <Eigen/Dense>
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
        
#ifdef _WIN32
        int cpuInfo[4];
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
    // --- SIMD Infrastructure for Phase 4 ---
    
    bool has_avx2() {
#if defined(__x86_64__) || defined(_M_X64)
        static bool checked = false;
        static bool available = false;
        if (checked) return available;
        
#ifdef _WIN32
        int cpuInfo[4];
        __cpuid(cpuInfo, 0);
        if (cpuInfo[0] >= 7) {
            __cpuidex(cpuInfo, 7, 0);
            available = (cpuInfo[1] & (1 << 5)) != 0; // EBX bit 5 is AVX2
        }
#else
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid_max(0, &eax) >= 7) {
            __cpuid_count(7, 0, eax, ebx, ecx, edx);
            available = (ebx & (1 << 5)) != 0;
        }
#endif
        checked = true;
        return available;
#else
        return false;
#endif
    }

    template<typename T>
    using SimdKernel = void(*)(T* dst, const T* a, const T* b, size_t n);

    // Scalar fallbacks (Auto-vectorizable)
    template<typename T> void scalar_add(T* dst, const T* a, const T* b, size_t n) {
        for(size_t i=0; i<n; ++i) dst[i] = static_cast<T>(scalar::WrapPlus{}(a[i], b[i]));
    }
    template<typename T> void scalar_sub(T* dst, const T* a, const T* b, size_t n) {
        for(size_t i=0; i<n; ++i) dst[i] = static_cast<T>(scalar::WrapMinus{}(a[i], b[i]));
    }
    template<typename T> void scalar_mul(T* dst, const T* a, const T* b, size_t n) {
        for(size_t i=0; i<n; ++i) dst[i] = static_cast<T>(scalar::WrapMultiplies{}(a[i], b[i]));
    }
    template<typename T> void scalar_div(T* dst, const T* a, const T* b, size_t n) { 
        for(size_t i=0; i<n; ++i) dst[i] = a[i] / b[i]; 
    }

    // AVX2 Kernels, per-function ISA target for GCC/Clang so these compile
    // without a global -march (runtime-dispatched via has_avx2() below). MSVC
    // emits intrinsics without /arch, so the attribute is a no-op there.
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64))
    #define PYCAUSET_AVX2_TARGET __attribute__((target("avx2")))
#else
    #define PYCAUSET_AVX2_TARGET
#endif
#if defined(__x86_64__) || defined(_M_X64)
    // Float
    PYCAUSET_AVX2_TARGET void avx2_add_f32(float* dst, const float* a, const float* b, size_t n) {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
        }
        for (; i < n; ++i) dst[i] = a[i] + b[i];
    }
    PYCAUSET_AVX2_TARGET void avx2_sub_f32(float* dst, const float* a, const float* b, size_t n) {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(dst + i, _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
        }
        for (; i < n; ++i) dst[i] = a[i] - b[i];
    }
    PYCAUSET_AVX2_TARGET void avx2_mul_f32(float* dst, const float* a, const float* b, size_t n) {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
        }
        for (; i < n; ++i) dst[i] = a[i] * b[i];
    }
    PYCAUSET_AVX2_TARGET void avx2_div_f32(float* dst, const float* a, const float* b, size_t n) {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(dst + i, _mm256_div_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
        }
        for (; i < n; ++i) dst[i] = a[i] / b[i];
    }

    // Double
    PYCAUSET_AVX2_TARGET void avx2_add_f64(double* dst, const double* a, const double* b, size_t n) {
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            _mm256_storeu_pd(dst + i, _mm256_add_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
        }
        for (; i < n; ++i) dst[i] = a[i] + b[i];
    }
    PYCAUSET_AVX2_TARGET void avx2_sub_f64(double* dst, const double* a, const double* b, size_t n) {
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            _mm256_storeu_pd(dst + i, _mm256_sub_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
        }
        for (; i < n; ++i) dst[i] = a[i] - b[i];
    }
    PYCAUSET_AVX2_TARGET void avx2_mul_f64(double* dst, const double* a, const double* b, size_t n) {
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            _mm256_storeu_pd(dst + i, _mm256_mul_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
        }
        for (; i < n; ++i) dst[i] = a[i] * b[i];
    }
    PYCAUSET_AVX2_TARGET void avx2_div_f64(double* dst, const double* a, const double* b, size_t n) {
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            _mm256_storeu_pd(dst + i, _mm256_div_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
        }
        for (; i < n; ++i) dst[i] = a[i] / b[i];
    }
#endif
#undef PYCAUSET_AVX2_TARGET

    // Discriminators
    template<typename T> SimdKernel<T> get_add_kernel() { return scalar_add<T>; }
    template<typename T> SimdKernel<T> get_sub_kernel() { return scalar_sub<T>; }
    template<typename T> SimdKernel<T> get_mul_kernel() { return scalar_mul<T>; }
    template<typename T> SimdKernel<T> get_div_kernel() { return scalar_div<T>; }

    // Explicit specializations for runtime dispatch
#if defined(__x86_64__) || defined(_M_X64)
    template<> SimdKernel<float> get_add_kernel<float>() { return has_avx2() ? avx2_add_f32 : scalar_add<float>; }
    template<> SimdKernel<float> get_sub_kernel<float>() { return has_avx2() ? avx2_sub_f32 : scalar_sub<float>; }
    template<> SimdKernel<float> get_mul_kernel<float>() { return has_avx2() ? avx2_mul_f32 : scalar_mul<float>; }
    template<> SimdKernel<float> get_div_kernel<float>() { return has_avx2() ? avx2_div_f32 : scalar_div<float>; }
    
    template<> SimdKernel<double> get_add_kernel<double>() { return has_avx2() ? avx2_add_f64 : scalar_add<double>; }
    template<> SimdKernel<double> get_sub_kernel<double>() { return has_avx2() ? avx2_sub_f64 : scalar_sub<double>; }
    template<> SimdKernel<double> get_mul_kernel<double>() { return has_avx2() ? avx2_mul_f64 : scalar_mul<double>; }
    template<> SimdKernel<double> get_div_kernel<double>() { return has_avx2() ? avx2_div_f64 : scalar_div<double>; }
#endif

    template<typename T>
    void parallel_simd_binary_op(const MatrixBase& a, const MatrixBase& b, MatrixBase& result, SimdKernel<T> kernel) {
        if (a.is_transposed() || b.is_transposed() || result.is_transposed()) {
            // Fallback for non-contiguous memory layouts
            throw std::runtime_error("SIMD path requires contiguous memory (transpose not supported yet)");
        }
        // Assume dense checks passed by caller
        const T* a_ptr = reinterpret_cast<const DenseMatrix<T>&>(a).data();
        const T* b_ptr = reinterpret_cast<const DenseMatrix<T>&>(b).data();
        T* res_ptr = reinterpret_cast<DenseMatrix<T>&>(result).data();
        size_t total = result.rows() * result.cols();
        
        size_t block_size = 32768 / sizeof(T); // L1/L2 cache friendly chunk
        size_t num_blocks = (total + block_size - 1) / block_size;
        
        ParallelFor(0, num_blocks, [=](size_t block_idx) {
            size_t start = block_idx * block_size;
            size_t end = std::min(start + block_size, total);
            kernel(res_ptr + start, a_ptr + start, b_ptr + start, end - start);
        });
    }

    template<typename T>
    bool try_fast_simd(const MatrixBase& a, const MatrixBase& b, MatrixBase& result, SimdKernel<T> kernel) {
        if (auto* rd = dynamic_cast<DenseMatrix<T>*>(&result)) {
            if (auto* ad = dynamic_cast<const DenseMatrix<T>*>(&a)) {
                if (auto* bd = dynamic_cast<const DenseMatrix<T>*>(&b)) {
                    // Raw flat pointers only for full spans (not transposed, not a window).
                    auto full_span = [](const DenseMatrix<T>* m) { return !m->is_transposed() && !m->is_subview(); };
                    bool is_contiguous = full_span(rd) && full_span(ad) && full_span(bd);
                                         
                    if (is_contiguous &&
                        ad->rows() == bd->rows() && ad->cols() == bd->cols() &&
                        rd->rows() == ad->rows() && rd->cols() == ad->cols() &&
                        rd->get_scalar() == 1.0 && ad->get_scalar() == 1.0 && bd->get_scalar() == 1.0) {
                            parallel_simd_binary_op<T>(a, b, result, kernel);
                            return true;
                    }
                }
            }
        }
        return false;
    }
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

    // Correctness-first integer matmul for mixed-kind inputs. Accumulates in
    // double, which is exact for integer results up to 2^53, so every integer
    // width except int64/uint64 is always exact; int64/uint64 results are exact
    // up to 2^53 (documented boundary).
    template <typename T>
    inline void generic_integer_matmul_into(const MatrixBase& a, const MatrixBase& b,
                                            DenseMatrix<T>& c, uint64_t m, uint64_t k, uint64_t n) {
        const double lo = static_cast<double>((std::numeric_limits<T>::lowest)());
        const double hi = static_cast<double>((std::numeric_limits<T>::max)());
        ParallelFor(0, m, [&](size_t i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t kk = 0; kk < k; ++kk) {
                    sum += a.get_element_as_double(static_cast<uint64_t>(i), static_cast<uint64_t>(kk)) *
                           b.get_element_as_double(static_cast<uint64_t>(kk), static_cast<uint64_t>(j));
                }
                if (sum > hi || sum < lo) {
                    throw std::overflow_error("Integer matmul overflow: result does not fit in output dtype");
                }
                c.set(static_cast<uint64_t>(i), static_cast<uint64_t>(j), static_cast<T>(sum));
            }
        });
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
        b_transposed_mat->set_temporary(true);
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

    // Materialize a strictly-upper triangular bit matrix to a dense bit matrix.
    inline std::unique_ptr<DenseMatrix<bool>> materialize_bit_to_dense(const TriangularMatrix<bool>& tri) {
        const uint64_t n = tri.rows();
        auto dense = std::make_unique<DenseMatrix<bool>>(n, n, "");
        dense->set_temporary(true);
        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t j = i + 1; j < n; ++j) {
                if (tri.get(i, j)) {
                    dense->set(i, j, true);
                }
            }
        }
        dense->set_scalar(tri.get_scalar());
        return dense;
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
        pycauset::float16_t* rdst = out.real_data();
        pycauset::float16_t* idst = out.imag_data();

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
                rdst[idx] = pycauset::float16_t(vc.real());
                idst[idx] = pycauset::float16_t(vc.imag());
            }
        });

        out.set_scalar(1.0);
    }

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

            // Check: does the operation fit in RAM? (Anti-Nanny Logic). When it
            // does, call BLAS directly. We intentionally do NOT pin the operands
            // (VirtualLock): pinning is a GPU-DMA optimization (R1_GPU) and is pure
            // overhead for CPU dgemm, it added ~1-2ms per call at n=1024 and was
            // the dominant residual in the matmul parity gap.
            if (governor.should_use_direct_path(total_bytes)) {
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

                if constexpr (std::is_same_v<T, double>) {
                    // GEMM scales well with threads, while small LAPACK
                    // factorizations (invert/determinant) do not, so we keep the
                    // global OpenBLAS default low and bump it here only for the
                    // matmul GEMM, then restore. The bump is capped to the number
                    // of hardware threads: asking for 20 threads on a small vCPU
                    // runner (macOS CI exposes 3) made OpenBLAS grow its pthread
                    // pool at runtime, which deadlocked. We also skip the set call
                    // entirely when the target already equals the current count, so
                    // no pool reconfiguration happens on machines where the global
                    // default already matches the hardware thread count.
                    const int prev_threads = openblas_get_num_threads();
                    unsigned hw_threads = std::thread::hardware_concurrency();
                    if (hw_threads == 0) hw_threads = 1;
                    const int gemm_threads = static_cast<int>(hw_threads > 20u ? 20u : hw_threads);
                    if (gemm_threads != prev_threads) {
                        openblas_set_num_threads(gemm_threads);
                    }
                    cblas_dgemm(
                        CblasRowMajor,
                        t_a ? CblasTrans : CblasNoTrans,
                        t_b ? CblasTrans : CblasNoTrans,
                        M, N, K,
                        1.0, a_data, lda,
                        b_data, ldb,
                        0.0, c_data, ldc
                    );
                    if (gemm_threads != prev_threads) {
                        openblas_set_num_threads(prev_threads);
                    }
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

                c_dense->set_scalar(a_dense->get_scalar() * b_dense->get_scalar());
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
        
        // Compute pointers with offsets
        const T* src_base = in_dense->data() + (in_dense->row_offset() * in_dense->base_cols() + in_dense->col_offset());
        T* dst_base = out_dense->data() + (out_dense->row_offset() * out_dense->base_cols() + out_dense->col_offset());
        
        const uint64_t src_stride = in_dense->base_cols();
        const uint64_t dst_stride = out_dense->base_cols();

        const bool is_transposed = in_dense->is_transposed();
        
        // 1. Copy input to output
        if (is_transposed) {
             ParallelFor(0, n, [&](size_t i) {
                for (size_t j = 0; j < n; ++j) {
                    dst_base[i * dst_stride + j] = src_base[j * src_stride + i];
                }
            });
        } else {
             ParallelFor(0, n, [&](size_t i) {
                for (size_t j = 0; j < n; ++j) {
                    dst_base[i * dst_stride + j] = src_base[i * src_stride + j];
                }
             });
        }
        
        // 2. LAPACK Factorization (dgetrf)
        std::vector<lapack_int> ipiv(n);
        lapack_int info = 0;
        
        if constexpr (std::is_same_v<T, double>) {
            info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, (lapack_int)n, (lapack_int)n, dst_base, (lapack_int)dst_stride, ipiv.data());
        } else {
            info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, (lapack_int)n, (lapack_int)n, dst_base, (lapack_int)dst_stride, ipiv.data());
        }
        
        if (info > 0) throw std::runtime_error("Matrix is singular");
        if (info < 0) throw std::runtime_error("Illegal argument in LAPACK dgetrf");
        
        // 3. LAPACK Inverse (dgetri)
        if constexpr (std::is_same_v<T, double>) {
            info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, (lapack_int)n, dst_base, (lapack_int)dst_stride, ipiv.data());
        } else {
            info = LAPACKE_sgetri(LAPACK_ROW_MAJOR, (lapack_int)n, dst_base, (lapack_int)dst_stride, ipiv.data());
        }
        
        if (info > 0) throw std::runtime_error("Matrix is singular");
        if (info < 0) throw std::runtime_error("Illegal argument in LAPACK dgetri");
        
        // Handle scalar
        std::complex<double> s = in_dense->get_scalar();
        if (s != 1.0) {
            out_dense->set_scalar(1.0 / s);
        }
    }

    // --- Helper: Recursive Blocked Inverse (Out-of-Core Friendly) ---
    template <typename T>
    void inverse_recursive(const DenseMatrix<T>* in, DenseMatrix<T>* out) {
        const uint64_t n = in->rows();
        if (n <= 256) {
             inverse_direct(in, out);
             return;
        }

        const uint64_t n1 = n / 2;
        const uint64_t n2 = n - n1;

        auto A11 = in->submatrix(0, 0, n1, n1);
        auto A12 = in->submatrix(0, n1, n1, n2); 
        auto A21 = in->submatrix(n1, 0, n2, n1);
        auto A22 = in->submatrix(n1, n1, n2, n2);

        // U = A11^-1
        auto U = std::make_unique<DenseMatrix<T>>(n1, n1);
        inverse_recursive(A11.get(), U.get());
        
        // T1 = U * A12
        auto T1 = std::make_unique<DenseMatrix<T>>(n1, n2);
        matmul_impl(U.get(), A12.get(), T1.get());
        
        // T2 = A21 * T1
        auto T2 = std::make_unique<DenseMatrix<T>>(n2, n2);
        matmul_impl(A21.get(), T1.get(), T2.get());
        
        // S = A22 - T2
        auto S = std::make_unique<DenseMatrix<T>>(n2, n2);
        {
            const T* a22_ptr = A22->data() + (A22->row_offset()*A22->base_cols() + A22->col_offset());
            const T* t2_ptr = T2->data();
            T* s_ptr = S->data();
            const uint64_t stride_a22 = A22->base_cols();
            const uint64_t stride_s = n2;
            const uint64_t stride_t2 = n2; // T2 is new/dense
            
            ParallelFor(0, n2, [&](size_t i) {
                for (size_t j = 0; j < n2; ++j) {
                    s_ptr[i*stride_s + j] = a22_ptr[i*stride_a22 + j] - t2_ptr[i*stride_t2 + j];
                }
            });
        }
        
        // V = S^-1 -> Out22
        auto V = out->submatrix(n1, n1, n2, n2);
        inverse_recursive(S.get(), V.get());
        
        // Clean S, T2
        S.reset(); T2.reset();
        
        // Out12 = - T1 * V
        auto Out12 = out->submatrix(0, n1, n1, n2);
        matmul_impl(T1.get(), V.get(), Out12.get());
        {
             T* ptr = Out12->data() + (Out12->row_offset()*Out12->base_cols() + Out12->col_offset());
             uint64_t stride = Out12->base_cols();
             ParallelFor(0, n1, [&](size_t i) { for(size_t j=0; j<n2; ++j) ptr[i*stride+j] *= -1.0; });
        }
        
        // Out21 = - V * A21 * U
        auto T3 = std::make_unique<DenseMatrix<T>>(n2, n1);
        matmul_impl(V.get(), A21.get(), T3.get());
        auto Out21 = out->submatrix(n1, 0, n2, n1);
        matmul_impl(T3.get(), U.get(), Out21.get());
        {
             T* ptr = Out21->data() + (Out21->row_offset()*Out21->base_cols() + Out21->col_offset());
             uint64_t stride = Out21->base_cols();
             ParallelFor(0, n2, [&](size_t i) { for(size_t j=0; j<n1; ++j) ptr[i*stride+j] *= -1.0; });
        }
        
        // Out11 = U - T1 * Out21
        auto T4 = std::make_unique<DenseMatrix<T>>(n1, n1);
        matmul_impl(T1.get(), Out21.get(), T4.get());
        auto Out11 = out->submatrix(0, 0, n1, n1);
        {
            const T* u_ptr = U->data();
            const T* t4_ptr = T4->data();
            T* o_ptr = Out11->data() + (Out11->row_offset()*Out11->base_cols() + Out11->col_offset());
            uint64_t stride = Out11->base_cols();
            
            ParallelFor(0, n1, [&](size_t i) {
                for (size_t j = 0; j < n1; ++j) {
                    o_ptr[i*stride + j] = u_ptr[i*n1 + j] - t4_ptr[i*n1 + j];
                }
            });
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
            // Prioritize LAPACK
            if (pycauset::core::MemoryGovernor::instance().should_use_direct_path(total_bytes)) {
                inverse_direct(in_dense, out_dense);
                return;
            }
            
            // Fallback: Recursive Blocked Inverse
            // This safely handles large matrices by decomposing them into smaller blocks
            // that fit in memory, spilling intermediates to disk if necessary.
            inverse_recursive(in_dense, out_dense);
            return;
        }

        throw std::runtime_error("CpuSolver::inverse not implemented for this dtype (only f32/f64)");
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
} // namespace

namespace {
    // Reads element (i, j) of a dense operand as the result type T, honouring the view
    // (transpose, offsets, broadcasting, scale factor, conjugation). Built once per operation.
    template <typename T>
    struct StridedReader {
        using RowFn = void (*)(const StridedReader&, uint64_t, uint64_t, uint64_t, T*);
        const void* data = nullptr;
        int64_t base = 0, si = 0, sj = 0;
        std::complex<double> scalar = 1.0;
        int64_t kf = 1;  // integer factor of an integer view (applied with wrapping in U)
        bool scaled = false, conj = false;
        RowFn row_fn = nullptr;
        // Fills out[0 .. j1-j0) with elements (i, j0) .. (i, j1-1).
        void row(uint64_t i, uint64_t j0, uint64_t j1, T* out) const { row_fn(*this, i, j0, j1, out); }
    };

    template <typename T, typename U>
    void strided_row(const StridedReader<T>& r, uint64_t i, uint64_t j0, uint64_t j1, T* out) {
        const U* src = static_cast<const U*>(r.data) + r.base + static_cast<int64_t>(i) * r.si;
        const int64_t sj = r.sj;
        const uint64_t n = j1 - j0;
        src += static_cast<int64_t>(j0) * sj;
        if constexpr (std::is_integral_v<U> && !std::is_same_v<U, bool>) {
            // Integer operand: the view's value is wrap(x * k) in U, then converted to T.
            if (r.scaled) {
                const int64_t k = r.kf;
                for (uint64_t q = 0; q < n; ++q) {
                    const U v = pycauset::scalar::wrap_scale(src[static_cast<int64_t>(q) * sj], k);
                    if constexpr (is_std_complex_v<T>) out[q] = static_cast<T>(std::complex<double>(static_cast<double>(v), 0.0));
                    else if constexpr (std::is_integral_v<T>) out[q] = static_cast<T>(v);
                    else out[q] = static_cast<T>(static_cast<double>(v));
                }
                return;
            }
        }
        if constexpr (is_std_complex_v<T>) {
            const bool conj = r.conj, scaled = r.scaled;
            const std::complex<double> s = r.scalar;
            for (uint64_t q = 0; q < n; ++q) {
                std::complex<double> z = pycauset::scalar::to_complex_double(src[static_cast<int64_t>(q) * sj]);
                if (conj) z = std::conj(z);
                if (scaled) z *= s;
                out[q] = static_cast<T>(z);
            }
        } else if constexpr (std::is_integral_v<T>) {
            for (uint64_t q = 0; q < n; ++q) out[q] = static_cast<T>(src[static_cast<int64_t>(q) * sj]);
        } else {
            if (r.scaled) {
                const double s = r.scalar.real();
                for (uint64_t q = 0; q < n; ++q) out[q] = static_cast<T>(static_cast<double>(src[static_cast<int64_t>(q) * sj]) * s);
            } else if constexpr (std::is_same_v<T, U>) {
                if (sj == 1) {
                    std::memcpy(out, src, n * sizeof(T));
                } else {
                    for (uint64_t q = 0; q < n; ++q) out[q] = src[static_cast<int64_t>(q) * sj];
                }
            } else {
                for (uint64_t q = 0; q < n; ++q) out[q] = static_cast<T>(static_cast<double>(src[static_cast<int64_t>(q) * sj]));
            }
        }
    }

    // Which (result T, operand U) pairs the strided path handles: any operand into a complex
    // result; real operands into a real float result; integer operands into an integer result.
    template <typename T, typename U>
    constexpr bool strided_supported_v =
        is_std_complex_v<T> || (!is_std_complex_v<U> && (!std::is_integral_v<T> || std::is_integral_v<U>));

    template <typename T, typename U>
    bool try_strided(const MatrixBase& m, uint64_t rows, uint64_t cols, StridedReader<T>& r) {
        if constexpr (!strided_supported_v<T, U>) {
            return false;
        } else {
            const auto* d = dynamic_cast<const DenseMatrix<U>*>(&m);
            if (d == nullptr) return false;
            if constexpr (std::is_integral_v<U> && !std::is_same_v<U, bool>) {
                // Integer operands carry only exact integer factors (integer views, F-042).
                if (!pycauset::scalar::is_exact_int_factor(m.get_scalar())) return false;
                r.kf = static_cast<int64_t>(m.get_scalar().real());
            }
            if constexpr (!is_std_complex_v<T>) {
                if (m.get_scalar().imag() != 0.0) return false;
            }
            const int64_t bc = static_cast<int64_t>(d->base_cols());
            const bool tr = d->is_transposed();
            int64_t si = tr ? 1 : bc, sj = tr ? bc : 1;
            if (d->rows() == 1 && rows != 1) si = 0;  // broadcasting
            if (d->cols() == 1 && cols != 1) sj = 0;
            r.data = d->data();
            r.base = static_cast<int64_t>(d->row_offset()) * bc + static_cast<int64_t>(d->col_offset());
            r.si = si;
            r.sj = sj;
            r.scalar = m.get_scalar();
            r.scaled = r.scalar != std::complex<double>(1.0, 0.0);
            r.conj = m.is_conjugated();
            r.row_fn = &strided_row<T, U>;
            return true;
        }
    }

    template <typename T>
    bool make_strided_reader(const MatrixBase& m, uint64_t rows, uint64_t cols, StridedReader<T>& r) {
        if (m.get_structure() != Structure::DENSE) return false;
        switch (m.get_data_type()) {
            case DataType::INT8: return try_strided<T, int8_t>(m, rows, cols, r);
            case DataType::INT16: return try_strided<T, int16_t>(m, rows, cols, r);
            case DataType::INT32: return try_strided<T, int32_t>(m, rows, cols, r);
            case DataType::INT64: return try_strided<T, int64_t>(m, rows, cols, r);
            case DataType::UINT8: return try_strided<T, uint8_t>(m, rows, cols, r);
            case DataType::UINT16: return try_strided<T, uint16_t>(m, rows, cols, r);
            case DataType::UINT32: return try_strided<T, uint32_t>(m, rows, cols, r);
            case DataType::UINT64: return try_strided<T, uint64_t>(m, rows, cols, r);
            case DataType::FLOAT32: return try_strided<T, float>(m, rows, cols, r);
            case DataType::FLOAT64: return try_strided<T, double>(m, rows, cols, r);
            case DataType::COMPLEX_FLOAT32: return try_strided<T, std::complex<float>>(m, rows, cols, r);
            case DataType::COMPLEX_FLOAT64: return try_strided<T, std::complex<double>>(m, rows, cols, r);
            default: return false;  // bit, float16, complex16: per-element path
        }
    }

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
    auto* c_f16 = dynamic_cast<DenseMatrix<pycauset::float16_t>*>(&result);
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
        pycauset::float16_t* c_data = c_f16->data();
        ParallelFor(0, m, [&](size_t i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t kk = 0; kk < k; ++kk) {
                    sum += static_cast<float>(a.get_element_as_double(static_cast<uint64_t>(i), static_cast<uint64_t>(kk))) *
                           static_cast<float>(b.get_element_as_double(static_cast<uint64_t>(kk), static_cast<uint64_t>(j)));
                }
                const pycauset::float16_t out(sum);
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
        pycauset::float16_t* rdst = c_cf16->real_data();
        pycauset::float16_t* idst = c_cf16->imag_data();

        ParallelFor(0, m, [&](size_t i) {
            for (size_t j = 0; j < n; ++j) {
                std::complex<double> sum = 0.0;
                for (size_t kk = 0; kk < k; ++kk) {
                    sum += a.get_element_as_complex(static_cast<uint64_t>(i), static_cast<uint64_t>(kk)) *
                           b.get_element_as_complex(static_cast<uint64_t>(kk), static_cast<uint64_t>(j));
                }
                const uint64_t idx = t_c ? (static_cast<uint64_t>(j) * storage_cols + static_cast<uint64_t>(i))
                                        : (static_cast<uint64_t>(i) * storage_cols + static_cast<uint64_t>(j));
                rdst[idx] = pycauset::float16_t(sum.real());
                idst[idx] = pycauset::float16_t(sum.imag());
            }
        });

        c_cf16->set_scalar(1.0);
        return;
    }

    // 2b. Integer x integer -> integer: exact through float64 BLAS when the measured
    // bounds allow it (F-055); otherwise the integer kernels below.
    {
        const DataType da = a.get_data_type(), db = b.get_data_type(), dr = result.get_data_type();
        auto is_int = [](DataType d) { return pycauset::is_integer_or_bit(d) && d != DataType::BIT; };
        if (is_int(da) && is_int(db) && is_int(dr) && a.get_structure() == Structure::DENSE &&
            b.get_structure() == Structure::DENSE) {
            bool done = false;
            switch (dr) {
                case DataType::INT8: done = try_int_matmul_via_dgemm<int8_t>(a, b, result); break;
                case DataType::INT16: done = try_int_matmul_via_dgemm<int16_t>(a, b, result); break;
                case DataType::INT32: done = try_int_matmul_via_dgemm<int32_t>(a, b, result); break;
                case DataType::INT64: done = try_int_matmul_via_dgemm<int64_t>(a, b, result); break;
                case DataType::UINT8: done = try_int_matmul_via_dgemm<uint8_t>(a, b, result); break;
                case DataType::UINT16: done = try_int_matmul_via_dgemm<uint16_t>(a, b, result); break;
                case DataType::UINT32: done = try_int_matmul_via_dgemm<uint32_t>(a, b, result); break;
                case DataType::UINT64: done = try_int_matmul_via_dgemm<uint64_t>(a, b, result); break;
                default: break;
            }
            if (done) {
                debug_trace::set_last("cpu.matmul.int_via_dgemm");
                return;
            }
        }
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

    // 5. Bit x Bit -> int32 (popcount), for any mix of dense and triangular bit
    // matrices. Triangular operands are materialized to dense first so the
    // popcount kernel can assume uniform dense column indexing.
    auto* c_int = dynamic_cast<DenseMatrix<int32_t>*>(&result);
    auto* a_tbm = dynamic_cast<const TriangularMatrix<bool>*>(&a);
    auto* b_tbm = dynamic_cast<const TriangularMatrix<bool>*>(&b);
    if (c_int && (a_bit || a_tbm) && (b_bit || b_tbm)) {
        debug_trace::set_last("cpu.matmul.bitbit_popcount");
        const uint64_t m = a.rows();
        const uint64_t k = a.cols();
        const uint64_t n = b.cols();
        if (b.rows() != k || result.rows() != m || result.cols() != n) {
            throw std::invalid_argument("Dimension mismatch");
        }

        std::unique_ptr<DenseMatrix<bool>> a_holder;
        std::unique_ptr<DenseMatrix<bool>> b_holder;
        const DenseMatrix<bool>* a_dense = a_bit;
        const DenseMatrix<bool>* b_dense = b_bit;
        if (a_tbm) { a_holder = materialize_bit_to_dense(*a_tbm); a_dense = a_holder.get(); }
        if (b_tbm) { b_holder = materialize_bit_to_dense(*b_tbm); b_dense = b_holder.get(); }

        if (!a_dense->is_transposed() && !b_dense->is_transposed()) {
            bit_popcount_matmul(*a_dense, *b_dense, *c_int);
        } else {
            // Correct generic fallback (supports transpose flags).
            ParallelFor(0, m, [&](size_t i) {
                for (uint64_t j = 0; j < n; ++j) {
                    int64_t dot_product = 0;
                    for (uint64_t kk = 0; kk < k; ++kk) {
                        if (a_dense->get(static_cast<uint64_t>(i), kk) && b_dense->get(kk, j)) {
                            dot_product = checked_add_int64(dot_product, 1);
                        }
                    }
                    if (dot_product > static_cast<int64_t>((std::numeric_limits<int32_t>::max)())) {
                        throw std::overflow_error("Integer matmul overflow: bit×bit result does not fit in int32 output");
                    }
                    c_int->set(static_cast<uint64_t>(i), j, static_cast<int32_t>(dot_product));
                }
            });
        }
        c_int->set_scalar(a.get_scalar() * b.get_scalar());
        return;
    }

    // 5c. Triangular int32 x Triangular {bit, int32} -> Triangular int32.
    auto* a_ti32 = dynamic_cast<const TriangularMatrix<int32_t>*>(&a);
    auto* b_ti32 = dynamic_cast<const TriangularMatrix<int32_t>*>(&b);
    auto* a_tbit = dynamic_cast<const TriangularMatrix<bool>*>(&a);
    auto* b_tbit = dynamic_cast<const TriangularMatrix<bool>*>(&b);
    auto* c_ti32 = dynamic_cast<TriangularMatrix<int32_t>*>(&result);
    if (c_ti32 && (a_ti32 || a_tbit) && (b_ti32 || b_tbit)) {
        debug_trace::set_last("cpu.matmul.tri_i32");
        const uint64_t n = a.rows();
        if (b.rows() != n || result.rows() != n || result.cols() != n) {
            throw std::invalid_argument("Dimension mismatch");
        }

        // Upper x upper: (i, j) sums k in [i, j]. Covers operands with and without a diagonal
        // (the strict-only loop dropped the diagonal terms, F-066).
        const bool c_diag = c_ti32->has_diagonal();
        ParallelFor(0, n, [&](size_t i) {
            for (uint64_t j = c_diag ? i : i + 1; j < n; ++j) {
                double acc = 0.0;
                for (uint64_t k = i; k <= j; ++k) {
                    const double av = a.get_element_as_double(static_cast<uint64_t>(i), k);
                    const double bv = b.get_element_as_double(k, j);
                    acc += av * bv;
                }
                if (acc > static_cast<double>((std::numeric_limits<int32_t>::max)())) {
                    throw std::overflow_error("Integer matmul overflow: triangular result does not fit in int32 output");
                }
                if (acc != 0.0) {
                    c_ti32->set(static_cast<uint64_t>(i), j, static_cast<int32_t>(acc));
                }
            }
        });
        // get_element_as_double applies the operand scalars, so the raw result
        // already holds the scaled product and the result scalar is the identity.
        c_ti32->set_scalar(1.0);
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

    // Generic integer result fallback for mixed-kind inputs (int x bit, bit x int,
    // and mixed-width int). Handles every integer width via a templated helper.
    {
        DenseMatrix<int8_t>*   c_i8  = dynamic_cast<DenseMatrix<int8_t>*>(&result);
        DenseMatrix<int16_t>*  c_i16 = dynamic_cast<DenseMatrix<int16_t>*>(&result);
        DenseMatrix<int32_t>*  c_i32 = dynamic_cast<DenseMatrix<int32_t>*>(&result);
        DenseMatrix<int64_t>*  c_i64 = dynamic_cast<DenseMatrix<int64_t>*>(&result);
        DenseMatrix<uint8_t>*  c_u8  = dynamic_cast<DenseMatrix<uint8_t>*>(&result);
        DenseMatrix<uint16_t>* c_u16 = dynamic_cast<DenseMatrix<uint16_t>*>(&result);
        DenseMatrix<uint32_t>* c_u32 = dynamic_cast<DenseMatrix<uint32_t>*>(&result);
        DenseMatrix<uint64_t>* c_u64 = dynamic_cast<DenseMatrix<uint64_t>*>(&result);

        if (c_i8 || c_i16 || c_i32 || c_i64 || c_u8 || c_u16 || c_u32 || c_u64) {
            debug_trace::set_last("cpu.matmul.generic_fallback_int");
            if (b.rows() != k || result.rows() != m || result.cols() != n) {
                throw std::invalid_argument("Dimension mismatch");
            }
            if (c_i8)  { generic_integer_matmul_into(a, b, *c_i8,  m, k, n); return; }
            if (c_i16) { generic_integer_matmul_into(a, b, *c_i16, m, k, n); return; }
            if (c_i32) { generic_integer_matmul_into(a, b, *c_i32, m, k, n); return; }
            if (c_i64) { generic_integer_matmul_into(a, b, *c_i64, m, k, n); return; }
            if (c_u8)  { generic_integer_matmul_into(a, b, *c_u8,  m, k, n); return; }
            if (c_u16) { generic_integer_matmul_into(a, b, *c_u16, m, k, n); return; }
            if (c_u32) { generic_integer_matmul_into(a, b, *c_u32, m, k, n); return; }
            if (c_u64) { generic_integer_matmul_into(a, b, *c_u64, m, k, n); return; }
        }
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

void CpuSolver::gemm(const MatrixBase& a, const MatrixBase& b, MatrixBase& c, double alpha, double beta) {
    // 1. Double Precision
    auto* a_dense = dynamic_cast<const DenseMatrix<double>*>(&a);
    auto* b_dense = dynamic_cast<const DenseMatrix<double>*>(&b);
    auto* c_dense = dynamic_cast<DenseMatrix<double>*>(&c);

    if (a_dense && b_dense && c_dense) {
        // Basic shape validation (worker usually guarantees this, but safety first)
        if (a_dense->rows() != c_dense->rows() || b_dense->cols() != c_dense->cols() || a_dense->cols() != b_dense->rows()) {
            throw std::invalid_argument("CpuSolver::gemm dimension mismatch");
        }

        const double* a_data = a_dense->data() + (a_dense->row_offset() * a_dense->base_cols() + a_dense->col_offset());
        const double* b_data = b_dense->data() + (b_dense->row_offset() * b_dense->base_cols() + b_dense->col_offset());
        double* c_data = c_dense->data() + (c_dense->row_offset() * c_dense->base_cols() + c_dense->col_offset());

        bool t_a = a_dense->is_transposed();
        bool t_b = b_dense->is_transposed();

        int M = static_cast<int>(a_dense->rows());
        int N = static_cast<int>(b_dense->cols());
        int K = static_cast<int>(a_dense->cols());

        // For submatrices/views, base_cols() is stride
        int lda = static_cast<int>(a_dense->base_cols());
        int ldb = static_cast<int>(b_dense->base_cols());
        int ldc = static_cast<int>(c_dense->base_cols());

        cblas_dgemm(
            CblasRowMajor,
            t_a ? CblasTrans : CblasNoTrans,
            t_b ? CblasTrans : CblasNoTrans,
            M, N, K,
            alpha, a_data, lda,
            b_data, ldb,
            beta, c_data, ldc
        );
        return;
    }

    // 2. Single Precision
    auto* a_f32 = dynamic_cast<const DenseMatrix<float>*>(&a);
    auto* b_f32 = dynamic_cast<const DenseMatrix<float>*>(&b);
    auto* c_f32 = dynamic_cast<DenseMatrix<float>*>(&c);

    if (a_f32 && b_f32 && c_f32) {
        if (a_f32->rows() != c_f32->rows() || b_f32->cols() != c_f32->cols() || a_f32->cols() != b_f32->rows()) {
            throw std::invalid_argument("CpuSolver::gemm dimension mismatch");
        }

        const float* a_data = a_f32->data() + (a_f32->row_offset() * a_f32->base_cols() + a_f32->col_offset());
        const float* b_data = b_f32->data() + (b_f32->row_offset() * b_f32->base_cols() + b_f32->col_offset());
        float* c_data = c_f32->data() + (c_f32->row_offset() * c_f32->base_cols() + c_f32->col_offset());

        bool t_a = a_f32->is_transposed();
        bool t_b = b_f32->is_transposed();

        int M = static_cast<int>(a_f32->rows());
        int N = static_cast<int>(b_f32->cols());
        int K = static_cast<int>(a_f32->cols());

        int lda = static_cast<int>(a_f32->base_cols());
        int ldb = static_cast<int>(b_f32->base_cols());
        int ldc = static_cast<int>(c_f32->base_cols());

        cblas_sgemm(
            CblasRowMajor,
            t_a ? CblasTrans : CblasNoTrans,
            t_b ? CblasTrans : CblasNoTrans,
            M, N, K,
            static_cast<float>(alpha), a_data, lda,
            b_data, ldb,
            static_cast<float>(beta), c_data, ldc
        );
        return;
    }

    throw std::runtime_error("CpuSolver::gemm fallback not implemented for these dtypes");
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

            for (uint64_t i = 0; i < n; ++i) {
                if (in_data[i] == 0.0) throw std::runtime_error("inverse: matrix is singular (zero on the diagonal)");
            }
            ParallelFor(0, n, [&](size_t i) { out_data[i] = 1.0 / in_data[i]; });  // no tolerance (F-063)

            out_diag->set_scalar(1.0 / in_diag->get_scalar());
            return;
        }
    }

    throw std::runtime_error("CpuSolver::inverse only supports DenseMatrix<double>, DenseMatrix<float>, or DiagonalMatrix<double>");
}

void CpuSolver::cholesky(const MatrixBase& matrix_in, MatrixBase& matrix_out) {
    if (matrix_in.rows() != matrix_in.cols()) {
        throw std::invalid_argument("CpuSolver::cholesky requires a square matrix");
    }

    const uint64_t n = matrix_in.rows();

    if (auto* in_dense = dynamic_cast<const DenseMatrix<double>*>(&matrix_in)) {
        auto* out_dense = dynamic_cast<DenseMatrix<double>*>(&matrix_out);
        if (!out_dense) throw std::runtime_error("CpuSolver::cholesky output must be DenseMatrix<double>");

        // Copy input to output
        std::memcpy(out_dense->data(), in_dense->data(), n * n * sizeof(double));

        // Use LAPACKE to compute Cholesky in-place on matrix_out
        lapack_int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', (lapack_int)n, out_dense->data(), (lapack_int)n);

        if (info != 0) {
            throw std::runtime_error("CpuSolver::cholesky failed (matrix not SPD, info=" + std::to_string(info) + ")");
        }

        // Zero out upper triangle (excluding diagonal)
        double* data = out_dense->data();
        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t j = i + 1; j < n; ++j) {
                data[i * n + j] = 0.0;
            }
        }
        return;
    }

    if (auto* in_dense = dynamic_cast<const DenseMatrix<float>*>(&matrix_in)) {
        auto* out_dense = dynamic_cast<DenseMatrix<float>*>(&matrix_out);
        if (!out_dense) throw std::runtime_error("CpuSolver::cholesky output must be DenseMatrix<float>");

        std::memcpy(out_dense->data(), in_dense->data(), n * n * sizeof(float));

        lapack_int info = LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'L', (lapack_int)n, out_dense->data(), (lapack_int)n);

        if (info != 0) {
            throw std::runtime_error("CpuSolver::cholesky failed (matrix not SPD, info=" + std::to_string(info) + ")");
        }

        float* data = out_dense->data();
        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t j = i + 1; j < n; ++j) {
                data[i * n + j] = 0.0f;
            }
        }
        return;
    }

    throw std::runtime_error("CpuSolver::cholesky only supports DenseMatrix<double> or DenseMatrix<float>");
}

namespace {

// y = A x through CpuSolver::matrix_vector_multiply, so every structure, dtype and
// storage (RAM or disk-backed) that matvec supports works here. Spectra drives it.
class MatVecOperator {
public:
    using Scalar = double;
    MatVecOperator(CpuSolver& solver, const MatrixBase& a)
        : solver_(solver), a_(a), n_(a.rows()), x_(n_, ""), y_(n_, "") {}
    Eigen::Index rows() const { return static_cast<Eigen::Index>(n_); }
    Eigen::Index cols() const { return static_cast<Eigen::Index>(n_); }
    void perform_op(const double* x_in, double* y_out) const {
        std::memcpy(x_.data(), x_in, n_ * sizeof(double));
        solver_.matrix_vector_multiply(a_, x_, y_);
        std::memcpy(y_out, y_.data(), n_ * sizeof(double));
    }

private:
    CpuSolver& solver_;
    const MatrixBase& a_;
    uint64_t n_;
    mutable DenseVector<double> x_;
    mutable DenseVector<double> y_;
};

}  // namespace

// Top-k eigenvalues by magnitude via ARPACK-style restarted Krylov methods (Spectra):
// symmetric input -> implicitly restarted Lanczos (real), otherwise implicitly restarted
// Arnoldi (complex). Convergence is checked against `tol`; non-convergence raises instead
// of returning approximations (F-030). Small problems where Krylov parameters can't be
// satisfied fall back to a dense eigensolver. `out` is DenseVector<double> for the
// symmetric path and DenseVector<complex<double>> otherwise.
void CpuSolver::eigvals_arnoldi(const MatrixBase& a, VectorBase& out, int k, int m, double tol) {
    if (k <= 0 || m <= 0) {
        throw std::invalid_argument("CpuSolver::eigvals_arnoldi requires positive k and m");
    }
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("CpuSolver::eigvals_arnoldi requires a square matrix");
    }
    const int64_t n = static_cast<int64_t>(a.rows());
    if (k > n) {
        throw std::invalid_argument("CpuSolver::eigvals_arnoldi: k exceeds the matrix size");
    }
    if (out.size() < static_cast<uint64_t>(k)) {
        throw std::invalid_argument("CpuSolver::eigvals_arnoldi output vector is too small");
    }
    auto* out_real = dynamic_cast<DenseVector<double>*>(&out);
    auto* out_cplx = dynamic_cast<DenseVector<std::complex<double>>*>(&out);
    if (!out_real && !out_cplx) {
        throw std::runtime_error("CpuSolver::eigvals_arnoldi output must be a float64 or complex128 vector");
    }
    const bool symmetric = out_real != nullptr;
    constexpr int kMaxRestarts = 1000;

    // Sort by |lambda| descending; ties broken by real then imaginary part (deterministic).
    auto write_sorted = [&](std::vector<std::complex<double>> vals) {
        std::stable_sort(vals.begin(), vals.end(), [](const auto& x, const auto& y) {
            if (std::abs(x) != std::abs(y)) return std::abs(x) > std::abs(y);
            if (x.real() != y.real()) return x.real() > y.real();
            return x.imag() > y.imag();
        });
        for (int i = 0; i < k; ++i) {
            if (symmetric) out_real->set(i, vals[i].real());
            else out_cplx->set(i, vals[i]);
        }
    };

    // Krylov constraints: Lanczos needs k < ncv <= n, Arnoldi needs k + 2 <= ncv <= n.
    const int64_t min_ncv = symmetric ? k + 1 : k + 2;
    const int64_t ncv = std::min<int64_t>(n, std::max<int64_t>(m, std::max<int64_t>(min_ncv, 2 * k + 1)));
    if (ncv < min_ncv || n <= 2) {
        // Too small for a Krylov method: dense eigensolver on the materialized matrix.
        Eigen::MatrixXd dense(n, n);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j) dense(i, j) = a.get_element_as_double(i, j);
        std::vector<std::complex<double>> vals;
        if (symmetric) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(dense, Eigen::EigenvaluesOnly);
            for (int64_t i = 0; i < n; ++i) vals.emplace_back(es.eigenvalues()(i), 0.0);
        } else {
            Eigen::EigenSolver<Eigen::MatrixXd> es(dense, false);
            for (int64_t i = 0; i < n; ++i) vals.push_back(es.eigenvalues()(i));
        }
        write_sorted(std::move(vals));
        return;
    }

    MatVecOperator op(*this, a);
    std::vector<std::complex<double>> vals;
    if (symmetric) {
        Spectra::SymEigsSolver<MatVecOperator> eigs(op, k, ncv);
        eigs.init();
        eigs.compute(Spectra::SortRule::LargestMagn, kMaxRestarts, tol, Spectra::SortRule::LargestMagn);
        if (eigs.info() != Spectra::CompInfo::Successful) {
            throw std::runtime_error("eigvals_arnoldi did not converge to tol=" + std::to_string(tol) +
                                     " (Lanczos, k=" + std::to_string(k) + ", m=" + std::to_string(ncv) +
                                     "); increase m or tol");
        }
        const Eigen::VectorXd ev = eigs.eigenvalues();
        for (Eigen::Index i = 0; i < ev.size(); ++i) vals.emplace_back(ev(i), 0.0);
    } else {
        Spectra::GenEigsSolver<MatVecOperator> eigs(op, k, ncv);
        eigs.init();
        eigs.compute(Spectra::SortRule::LargestMagn, kMaxRestarts, tol, Spectra::SortRule::LargestMagn);
        if (eigs.info() != Spectra::CompInfo::Successful) {
            throw std::runtime_error("eigvals_arnoldi did not converge to tol=" + std::to_string(tol) +
                                     " (Arnoldi, k=" + std::to_string(k) + ", m=" + std::to_string(ncv) +
                                     "); increase m or tol");
        }
        const Eigen::VectorXcd ev = eigs.eigenvalues();
        for (Eigen::Index i = 0; i < ev.size(); ++i) vals.push_back(ev(i));
    }
    write_sorted(std::move(vals));
}

void CpuSolver::eigvals_skew(const MatrixBase& a, VectorBase& out, int k) {
    if (k <= 0) {
        throw std::invalid_argument("CpuSolver::eigvals_skew requires positive k");
    }
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("CpuSolver::eigvals_skew requires a square matrix");
    }

    auto* out_c = dynamic_cast<DenseVector<std::complex<double>>*>(&out);
    if (!out_c) {
        throw std::runtime_error("CpuSolver::eigvals_skew output must be DenseVector<complex<double>>");
    }

    const uint64_t n = a.rows();
    const int kk = std::min(k, static_cast<int>(n));

    // A real skew-symmetric A is normal with eigenvalues +-i*lambda. H = iA is Hermitian, so
    // the Hermitian solver gives real mu with A's eigenvalues -i*mu: exact +- pairs, real
    // arithmetic accuracy, ~2.4x faster than the general dgeev used before (L7.1).
    // to_memory_flat_real_square folds in the matrix scalar.
    auto data = to_memory_flat_real_square(a);
    std::vector<std::complex<double>> h(n * n);
    for (uint64_t idx = 0; idx < n * n; ++idx) h[idx] = std::complex<double>(0.0, data[idx]);
    data.clear();
    data.shrink_to_fit();
    std::vector<double> mu(n);
    // Column-major view of the row-major buffer is conj(H): same (real) eigenvalues, no copy.
    lapack_int info = LAPACKE_zheevd(LAPACK_COL_MAJOR, 'N', 'U', static_cast<lapack_int>(n),
        reinterpret_cast<lapack_complex_double*>(h.data()), static_cast<lapack_int>(n), mu.data());
    if (info > 0) throw std::runtime_error("CpuSolver::eigvals_skew failed to converge");
    if (info < 0) throw std::runtime_error("CpuSolver::eigvals_skew: illegal argument");

    std::vector<std::complex<double>> evals(n);
    for (uint64_t i = 0; i < n; ++i) evals[i] = std::complex<double>(0.0, -mu[i]);
    // Descending magnitude; within a +-pair (magnitudes equal up to rounding: relative 1e-12 of
    // the largest), +i first. A deterministic function of the values.
    double scale = 0.0;
    for (const auto& z : evals) scale = std::max(scale, std::abs(z.imag()));
    const double tol = 1e-12 * scale;
    std::sort(evals.begin(), evals.end(), [](const std::complex<double>& x, const std::complex<double>& y) {
        return std::abs(x.imag()) > std::abs(y.imag());
    });
    for (uint64_t i = 0; i + 1 < n; ++i) {
        if (std::abs(std::abs(evals[i].imag()) - std::abs(evals[i + 1].imag())) <= tol &&
            evals[i].imag() < evals[i + 1].imag()) {
            std::swap(evals[i], evals[i + 1]);
            ++i;
        }
    }

    for (int i = 0; i < kk; ++i) {
        out_c->set(i, evals[i]);
    }
}

void CpuSolver::eig_skew(const MatrixBase& a, VectorBase& eigenvalues, MatrixBase& eigenvectors, int k) {
    if (k <= 0) {
        throw std::invalid_argument("CpuSolver::eig_skew requires positive k");
    }
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("CpuSolver::eig_skew requires a square matrix");
    }

    auto* out_w = dynamic_cast<DenseVector<std::complex<double>>*>(&eigenvalues);
    auto* out_v = dynamic_cast<DenseMatrix<std::complex<double>>*>(&eigenvectors);
    if (!out_w || !out_v) {
        throw std::runtime_error("CpuSolver::eig_skew output must be complex<double> vector + matrix");
    }

    const uint64_t n = a.rows();
    const int kk = std::min(k, static_cast<int>(n));
    if (eigenvectors.rows() != n || eigenvectors.cols() != static_cast<uint64_t>(kk)) {
        throw std::invalid_argument("CpuSolver::eig_skew eigenvectors matrix bad shape");
    }

    // Same scratch-and-factorize pattern as eigvals_skew, but request the right
    // eigenvectors so the Sorkin-Johnston W = V_+ diag(σ) V_+^H can be built.
    auto data = to_memory_flat_real_square(a);

    std::vector<double> wr(n);
    std::vector<double> wi(n);
    std::vector<double> vr(n * n);
    lapack_int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', static_cast<lapack_int>(n),
        data.data(), static_cast<lapack_int>(n), wr.data(), wi.data(),
        nullptr, static_cast<lapack_int>(n), vr.data(), static_cast<lapack_int>(n));
    if (info > 0) throw std::runtime_error("CpuSolver::eig_skew failed to converge");
    if (info < 0) throw std::runtime_error("CpuSolver::eig_skew: illegal argument");

    // Build the full complex eigensystem (eigenvalues + eigenvectors), unpacking
    // conjugate pairs exactly like the general eig path. Eigenvalues are set in a
    // separate loop first: each conjugate pair contributes BOTH eigenvalues, and the
    // eigenvector loop below skips the second column of each pair (++j), so writing
    // full_evals inside that loop would leave the "-i*lambda" eigenvalue unset.
    std::vector<std::complex<double>> full_evals(n);
    for (size_t j = 0; j < n; ++j) {
        full_evals[j] = { wr[j], wi[j] };
    }
    std::vector<std::complex<double>> full_evecs(n * n);
    for (size_t j = 0; j < n; ++j) {
        if (wi[j] == 0.0) {
            for (size_t i = 0; i < n; ++i) full_evecs[i * n + j] = { vr[i * n + j], 0.0 };
        } else {
            // Conjugate pair (j, j+1): v[j] = VR[:,j] + i VR[:,j+1],
            //                          v[j+1] = VR[:,j] - i VR[:,j+1].
            for (size_t i = 0; i < n; ++i) {
                full_evecs[i * n + j] = { vr[i * n + j], vr[i * n + (j + 1)] };
                full_evecs[i * n + (j + 1)] = { vr[i * n + j], -vr[i * n + (j + 1)] };
            }
            ++j;
        }
    }

    // Order by descending eigenvalue magnitude and return the top k.
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t x, size_t y) {
        return std::norm(full_evals[x]) > std::norm(full_evals[y]);
    });

    for (int t = 0; t < kk; ++t) {
        const size_t src = idx[t];
        out_w->set(t, full_evals[src]);
        for (size_t i = 0; i < n; ++i) {
            out_v->set(i, t, full_evecs[i * n + src]);
        }
    }
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
    // Exact element reader for integer results over any structure (resolved once per operand):
    // reads in the operand's own type and applies an integer view factor with wrapping.
    // Returns an empty function when the operand isn't integer/bit-typed.
    template <typename T>
    std::function<T(uint64_t, uint64_t)> exact_int_reader(const MatrixBase& m) {
        const bool scaled = m.get_scalar() != std::complex<double>(1.0, 0.0);
        if (scaled && !pycauset::scalar::is_exact_int_factor(m.get_scalar())) return {};
        const int64_t k = static_cast<int64_t>(m.get_scalar().real());
        try {
            return pycauset::visit_matrix(m, [&](const auto& typed) -> std::function<T(uint64_t, uint64_t)> {
                using M = std::decay_t<decltype(typed)>;
                using E = std::decay_t<decltype(typed.get(uint64_t{0}, uint64_t{0}))>;
                if constexpr (std::is_integral_v<E>) {
                    const M* p = &typed;
                    return [p, k, scaled](uint64_t i, uint64_t j) {
                        E v = p->get(i, j);
                        if (scaled) v = pycauset::scalar::wrap_scale(v, k);
                        return static_cast<T>(v);
                    };
                } else {
                    return {};
                }
            });
        } catch (const std::exception&) {
            return {};
        }
    }

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
            
            // Fast path: all dense, no transpose, factor 1, full spans (not windows). a_dense /
            // b_dense may be null (mixed-type operands): checked first.
            if (a_dense && b_dense && a_full && b_full &&
                res_dense->get_scalar() == 1.0 && !res_dense->is_transposed() && !res_dense->is_subview() &&
                !res_dense->is_conjugated() &&
                a_dense->get_scalar() == 1.0 && !a_dense->is_transposed() && !a_dense->is_subview() &&
                !a_dense->is_conjugated() &&
                b_dense->get_scalar() == 1.0 && !b_dense->is_transposed() && !b_dense->is_subview() &&
                !b_dense->is_conjugated()) {
                
                const T* a_data = a_dense->data();
                const T* b_data = b_dense->data();
                T* res_data = res_dense->data();
                
                ParallelFor(0, rows * cols, [&](size_t i) {
                    res_data[i] = op(a_data[i], b_data[i]);
                });
                return;
            }
            
            // View-aware dense path (F-043, F-049): each operand is read straight from storage,
            // with transpose / offsets / broadcasting as index arithmetic and the scale factor and
            // conjugation applied in the inner loop. Exact for integers (no detour through
            // double), as fast as the plain path, tiled so transposed reads stay in cache.
            // Operands that aren't dense numeric storage take the per-element path below.
            StridedReader<T> ra, rb;
            const bool res_plain = res_dense->rows() == res_dense->base_rows() &&
                                   res_dense->cols() == res_dense->base_cols() &&
                                   res_dense->get_scalar() == 1.0 && !res_dense->is_transposed() &&
                                   !res_dense->is_subview() && !res_dense->is_conjugated();
            if (res_plain && make_strided_reader(a, rows, cols, ra) && make_strided_reader(b, rows, cols, rb)) {
                T* res_data = res_dense->data();
                constexpr uint64_t TILE = 64;
                const uint64_t row_tiles = (rows + TILE - 1) / TILE;
                ParallelFor(0, row_tiles, [&](size_t rt) {
                    const uint64_t i0 = rt * TILE, i1 = std::min<uint64_t>(rows, i0 + TILE);
                    for (uint64_t j0 = 0; j0 < cols; j0 += TILE) {
                        const uint64_t j1 = std::min<uint64_t>(cols, j0 + TILE);
                        T bufa[TILE];
                        T bufb[TILE];
                        const uint64_t w = j1 - j0;
                        for (uint64_t i = i0; i < i1; ++i) {
                            ra.row(i, j0, j1, bufa);
                            rb.row(i, j0, j1, bufb);
                            T* out = res_data + i * cols + j0;
                            for (uint64_t k = 0; k < w; ++k) {
                                out[k] = op(bufa[k], bufb[k]);
                            }
                        }
                    }
                });
                return;
            }

            // Per-element fallback (bit / float16 / complex16 / structured operands). Integer
            // results read integer operands exactly (int64 beyond 2^53, wrapping views).
            if constexpr (std::is_integral_v<T>) {
                auto ea = exact_int_reader<T>(a);
                auto eb = exact_int_reader<T>(b);
                if (ea && eb) {
                    ParallelFor(0, rows, [&](size_t i) {
                        for (size_t j = 0; j < cols; ++j) {
                            const T va = ea(broadcast_index(i, a.rows()), broadcast_index(j, a.cols()));
                            const T vb = eb(broadcast_index(i, b.rows()), broadcast_index(j, b.cols()));
                            res_dense->set(i, j, op(va, vb));
                        }
                    });
                    return;
                }
            }
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
            // Structured results: integer element types read exactly (see exact_int_reader).
            std::function<T(uint64_t, uint64_t)> ea, eb;
            if constexpr (std::is_integral_v<T>) {
                ea = exact_int_reader<T>(a);
                eb = exact_int_reader<T>(b);
            }
            auto read_a = [&](uint64_t i, uint64_t j) -> T {
                return ea ? ea(i, j) : static_cast<T>(a.get_element_as_double(i, j));
            };
            auto read_b = [&](uint64_t i, uint64_t j) -> T {
                return eb ? eb(i, j) : static_cast<T>(b.get_element_as_double(i, j));
            };
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
                        T val = op(read_a(ia, ja),
                                   read_b(ib, jb));
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
                if (result.get_structure() == Structure::IDENTITY) {
                     if (rows > 0 && cols > 0) {
                         const uint64_t ia = broadcast_index(0, a.rows());
                         const uint64_t ja = broadcast_index(0, a.cols());
                         const uint64_t ib = broadcast_index(0, b.rows());
                         const uint64_t jb = broadcast_index(0, b.cols());
                         T val = op(read_a(ia, ja),
                                    read_b(ib, jb));
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
                    T val = op(read_a(ia, ja),
                               read_b(ib, jb));
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
                        T val = op(read_a(ia, ja),
                                   read_b(ib, jb));
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
            res->get_scalar() == 1.0 && !res->is_transposed() && !res->is_subview() &&
            res->rows() == res->base_rows() && res->cols() == res->base_cols() &&
            a_dense->get_scalar() == 1.0 && !a_dense->is_transposed() && !a_dense->is_subview() &&
            !a_dense->is_conjugated() &&
            a_dense->rows() == a_dense->base_rows() && a_dense->cols() == a_dense->base_cols()) {
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
            binary_op_complex16_impl(a, b, *out, scalar::WrapPlus{});
            return;
        }
        throw std::runtime_error("CpuSolver::add complex_float16 result type mismatch");
    }
    if (dtype == DataType::COMPLEX_FLOAT32) {
        debug_trace::set_last("cpu.add.c32");
        binary_op_impl<std::complex<float>>(a, b, result, scalar::WrapPlus{});
        return;
    }
    if (dtype == DataType::COMPLEX_FLOAT64) {
        debug_trace::set_last("cpu.add.c64");
        binary_op_impl<std::complex<double>>(a, b, result, scalar::WrapPlus{});
        return;
    }

    if (dtype == DataType::FLOAT64) {
        debug_trace::set_last("cpu.add.f64");
        if (try_fast_simd<double>(a, b, result, get_add_kernel<double>())) return;
        binary_op_impl<double>(a, b, result, scalar::WrapPlus{});
    } else if (dtype == DataType::FLOAT16) {
        debug_trace::set_last("cpu.add.f16");
        if (try_fast_simd<pycauset::float16_t>(a, b, result, get_add_kernel<pycauset::float16_t>())) return;
        binary_op_impl<pycauset::float16_t>(a, b, result, scalar::WrapPlus{});
    } else if (dtype == DataType::FLOAT32) {
        debug_trace::set_last("cpu.add.f32");
        if (try_fast_simd<float>(a, b, result, get_add_kernel<float>())) return;
        binary_op_impl<float>(a, b, result, scalar::WrapPlus{});
    } else if (dtype == DataType::INT8) {
        debug_trace::set_last("cpu.add.i8");
        if (try_fast_simd<int8_t>(a, b, result, get_add_kernel<int8_t>())) return;
        binary_op_impl<int8_t>(a, b, result, scalar::WrapPlus{});
    } else if (dtype == DataType::INT16) {
        debug_trace::set_last("cpu.add.i16");
        if (try_fast_simd<int16_t>(a, b, result, get_add_kernel<int16_t>())) return;
        binary_op_impl<int16_t>(a, b, result, scalar::WrapPlus{});
    } else if (dtype == DataType::INT32) {
        debug_trace::set_last("cpu.add.i32");
        if (try_fast_simd<int32_t>(a, b, result, get_add_kernel<int32_t>())) return;
        binary_op_impl<int32_t>(a, b, result, scalar::WrapPlus{});
    } else if (dtype == DataType::INT64) {
        debug_trace::set_last("cpu.add.i64");
        if (try_fast_simd<int64_t>(a, b, result, get_add_kernel<int64_t>())) return;
        binary_op_impl<int64_t>(a, b, result, scalar::WrapPlus{});
    } else if (dtype == DataType::UINT8) {
        debug_trace::set_last("cpu.add.u8");
        if (try_fast_simd<uint8_t>(a, b, result, get_add_kernel<uint8_t>())) return;
        binary_op_impl<uint8_t>(a, b, result, scalar::WrapPlus{});
    } else if (dtype == DataType::UINT16) {
        debug_trace::set_last("cpu.add.u16");
        binary_op_impl<uint16_t>(a, b, result, scalar::WrapPlus{});
    } else if (dtype == DataType::UINT32) {
        debug_trace::set_last("cpu.add.u32");
        binary_op_impl<uint32_t>(a, b, result, scalar::WrapPlus{});
    } else if (dtype == DataType::UINT64) {
        debug_trace::set_last("cpu.add.u64");
        binary_op_impl<uint64_t>(a, b, result, scalar::WrapPlus{});
    } else {
        throw std::runtime_error("CpuSolver::add result data type not supported");
    }
}

void CpuSolver::subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    DataType dtype = result.get_data_type();
    if (dtype == DataType::COMPLEX_FLOAT16) {
        debug_trace::set_last("cpu.subtract.cf16");
        if (auto* out = dynamic_cast<ComplexFloat16Matrix*>(&result)) {
            binary_op_complex16_impl(a, b, *out, scalar::WrapMinus{});
            return;
        }
        throw std::runtime_error("CpuSolver::subtract complex_float16 result type mismatch");
    }
    if (dtype == DataType::COMPLEX_FLOAT32) {
        debug_trace::set_last("cpu.subtract.c32");
        binary_op_impl<std::complex<float>>(a, b, result, scalar::WrapMinus{});
        return;
    }
    if (dtype == DataType::COMPLEX_FLOAT64) {
        debug_trace::set_last("cpu.subtract.c64");
        binary_op_impl<std::complex<double>>(a, b, result, scalar::WrapMinus{});
        return;
    }

    if (dtype == DataType::FLOAT64) {
        debug_trace::set_last("cpu.subtract.f64");
        if (try_fast_simd<double>(a, b, result, get_sub_kernel<double>())) return;
        binary_op_impl<double>(a, b, result, scalar::WrapMinus{});
    } else if (dtype == DataType::FLOAT16) {
        debug_trace::set_last("cpu.subtract.f16");
        binary_op_impl<pycauset::float16_t>(a, b, result, scalar::WrapMinus{});
    } else if (dtype == DataType::FLOAT32) {
        debug_trace::set_last("cpu.subtract.f32");
        if (try_fast_simd<float>(a, b, result, get_sub_kernel<float>())) return;
        binary_op_impl<float>(a, b, result, scalar::WrapMinus{});
    } else if (dtype == DataType::INT8) {
        debug_trace::set_last("cpu.subtract.i8");
        binary_op_impl<int8_t>(a, b, result, scalar::WrapMinus{});
    } else if (dtype == DataType::INT16) {
        debug_trace::set_last("cpu.subtract.i16");
        binary_op_impl<int16_t>(a, b, result, scalar::WrapMinus{});
    } else if (dtype == DataType::INT32) {
        debug_trace::set_last("cpu.subtract.i32");
        binary_op_impl<int32_t>(a, b, result, scalar::WrapMinus{});
    } else if (dtype == DataType::INT64) {
        debug_trace::set_last("cpu.subtract.i64");
        binary_op_impl<int64_t>(a, b, result, scalar::WrapMinus{});
    } else if (dtype == DataType::UINT8) {
        debug_trace::set_last("cpu.subtract.u8");
        binary_op_impl<uint8_t>(a, b, result, scalar::WrapMinus{});
    } else if (dtype == DataType::UINT16) {
        debug_trace::set_last("cpu.subtract.u16");
        binary_op_impl<uint16_t>(a, b, result, scalar::WrapMinus{});
    } else if (dtype == DataType::UINT32) {
        debug_trace::set_last("cpu.subtract.u32");
        binary_op_impl<uint32_t>(a, b, result, scalar::WrapMinus{});
    } else if (dtype == DataType::UINT64) {
        debug_trace::set_last("cpu.subtract.u64");
        binary_op_impl<uint64_t>(a, b, result, scalar::WrapMinus{});
    } else {
        throw std::runtime_error("CpuSolver::subtract result data type not supported");
    }
}

void CpuSolver::elementwise_multiply(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    DataType dtype = result.get_data_type();
    if (dtype == DataType::COMPLEX_FLOAT16) {
        debug_trace::set_last("cpu.mul.cf16");
        if (auto* out = dynamic_cast<ComplexFloat16Matrix*>(&result)) {
            binary_op_complex16_impl(a, b, *out, scalar::WrapMultiplies{});
            return;
        }
        throw std::runtime_error("CpuSolver::elementwise_multiply complex_float16 result type mismatch");
    }
    if (dtype == DataType::COMPLEX_FLOAT32) {
        debug_trace::set_last("cpu.mul.c32");
        binary_op_impl<std::complex<float>>(a, b, result, scalar::WrapMultiplies{});
        return;
    }
    if (dtype == DataType::COMPLEX_FLOAT64) {
        debug_trace::set_last("cpu.mul.c64");
        binary_op_impl<std::complex<double>>(a, b, result, scalar::WrapMultiplies{});
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
        if (try_fast_simd<double>(a, b, result, get_mul_kernel<double>())) return;
        binary_op_impl<double>(a, b, result, scalar::WrapMultiplies{});
    } else if (dtype == DataType::FLOAT16) {
        debug_trace::set_last("cpu.mul.f16");
        binary_op_impl<pycauset::float16_t>(a, b, result, scalar::WrapMultiplies{});
    } else if (dtype == DataType::FLOAT32) {
        debug_trace::set_last("cpu.mul.f32");
        if (try_fast_simd<float>(a, b, result, get_mul_kernel<float>())) return;
        binary_op_impl<float>(a, b, result, scalar::WrapMultiplies{});
    } else if (dtype == DataType::INT8) {
        debug_trace::set_last("cpu.mul.i8");
        binary_op_impl<int8_t>(a, b, result, scalar::WrapMultiplies{});
    } else if (dtype == DataType::INT16) {
        debug_trace::set_last("cpu.mul.i16");
        binary_op_impl<int16_t>(a, b, result, scalar::WrapMultiplies{});
    } else if (dtype == DataType::INT32) {
        debug_trace::set_last("cpu.mul.i32");
        binary_op_impl<int32_t>(a, b, result, scalar::WrapMultiplies{});
    } else if (dtype == DataType::INT64) {
        debug_trace::set_last("cpu.mul.i64");
        binary_op_impl<int64_t>(a, b, result, scalar::WrapMultiplies{});
    } else if (dtype == DataType::UINT8) {
        debug_trace::set_last("cpu.mul.u8");
        binary_op_impl<uint8_t>(a, b, result, scalar::WrapMultiplies{});
    } else if (dtype == DataType::UINT16) {
        debug_trace::set_last("cpu.mul.u16");
        binary_op_impl<uint16_t>(a, b, result, scalar::WrapMultiplies{});
    } else if (dtype == DataType::UINT32) {
        debug_trace::set_last("cpu.mul.u32");
        binary_op_impl<uint32_t>(a, b, result, scalar::WrapMultiplies{});
    } else if (dtype == DataType::UINT64) {
        debug_trace::set_last("cpu.mul.u64");
        binary_op_impl<uint64_t>(a, b, result, scalar::WrapMultiplies{});
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
        if (try_fast_simd<double>(a, b, result, get_div_kernel<double>())) return;
        binary_op_impl<double>(a, b, result, [](auto x, auto y) { return x / y; });
    } else if (dtype == DataType::FLOAT16) {
        debug_trace::set_last("cpu.div.f16");
        binary_op_impl<pycauset::float16_t>(a, b, result, [](auto x, auto y) { return x / y; });
    } else if (dtype == DataType::FLOAT32) {
        debug_trace::set_last("cpu.div.f32");
        if (try_fast_simd<float>(a, b, result, get_div_kernel<float>())) return;
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
    } else if (auto* r = dynamic_cast<DenseMatrix<pycauset::float16_t>*>(&result)) {
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
        // BLAS ddot: the former OpenMP reduction cost ~8 ms for 1M elements (thread-team
        // start-up next to OpenBLAS's own pool); ddot ~0.1 ms. View factors multiply in.
        debug_trace::set_last("cpu.dot.f64");
        const double s = cblas_ddot(static_cast<int>(n), a_dbl->data(), 1, b_dbl->data(), 1);
        return s * (a.get_scalar() * b.get_scalar()).real();
    }
    if (auto* a_f = dynamic_cast<const DenseVector<float>*>(&a)) {
        if (auto* b_f = dynamic_cast<const DenseVector<float>*>(&b)) {
            debug_trace::set_last("cpu.dot.f32");
            const double s = cblas_dsdot(static_cast<int>(n), a_f->data(), 1, b_f->data(), 1);
            return s * (a.get_scalar() * b.get_scalar()).real();
        }
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

    // Real dense matrix x vector through BLAS gemv (F-012: the generic per-element path was 24x
    // slower than NumPy). Views map to gemv arguments: transpose -> CblasTrans, offsets ->
    // pointer + leading dimension, factor -> alpha.
    {
        auto gemv = [&](auto tag) -> bool {
            using T = typename decltype(tag)::type;
            auto* md = dynamic_cast<const DenseMatrix<T>*>(&m);
            auto* vd = dynamic_cast<const DenseVector<T>*>(&v);
            auto* out = dynamic_cast<DenseVector<T>*>(&result);
            if (!md || !vd || !out) return false;
            if (m.get_scalar().imag() != 0.0 || v.get_scalar() != std::complex<double>(1.0, 0.0)) return false;
            const bool tr = md->is_transposed();
            const int ld = static_cast<int>(md->base_cols());
            const T* a = md->data() + md->row_offset() * md->base_cols() + md->col_offset();
            // Storage block seen by BLAS: (rows x cols) if not transposed, (cols x rows) if transposed.
            const int sr = static_cast<int>(tr ? cols : rows), sc = static_cast<int>(tr ? rows : cols);
            const T alpha = static_cast<T>(m.get_scalar().real());
            if constexpr (std::is_same_v<T, double>) {
                cblas_dgemv(CblasRowMajor, tr ? CblasTrans : CblasNoTrans, sr, sc, alpha, a, ld, vd->data(), 1, 0.0,
                            out->data(), 1);
            } else {
                cblas_sgemv(CblasRowMajor, tr ? CblasTrans : CblasNoTrans, sr, sc, alpha, a, ld, vd->data(), 1, 0.0f,
                            out->data(), 1);
            }
            out->set_scalar(1.0);
            return true;
        };
        if (gemv(pycauset::type_tag<double>{}) || gemv(pycauset::type_tag<float>{})) {
            debug_trace::set_last("cpu.matvec.blas");
            return;
        }
    }

    // Closure view C +- C^T over causal bits (LAZY.md L2b): two triangular passes over the
    // bits, nothing materialized. Used when the int32 entries can't wrap under the factor.
    if (auto* cl = dynamic_cast<const BitClosureMatrix*>(&m); cl && !cl->is_materialized()) {
        auto* out = dynamic_cast<DenseVector<double>*>(&result);
        const double k = m.get_scalar().real();
        if (out && m.get_scalar().imag() == 0.0 && std::abs(k) < 1073741824.0 &&
            v.get_data_type() != DataType::COMPLEX_FLOAT16 && v.get_data_type() != DataType::COMPLEX_FLOAT32 &&
            v.get_data_type() != DataType::COMPLEX_FLOAT64) {
            debug_trace::set_last("cpu.matvec.bit_closure");
            const auto& c = cl->bits();
            const double sg = static_cast<double>(cl->sign());
            const uint64_t n = rows;
            std::vector<double> x(n);
            for (uint64_t j = 0; j < n; ++j) x[j] = v.get_element_as_double(j);
            // Row i holds columns i+1..n-1 as bits, starting at byte offset row_offset(i).
            const auto* base = static_cast<const char*>(c.shared_mapper()->get_data());
            auto row_words = [&](uint64_t i) {
                return reinterpret_cast<const uint64_t*>(base + c.get_row_offset(i));
            };
            std::vector<double> up(n, 0.0), down(n, 0.0);
            // (C x)_i: gather along row i, visiting set bits only.
            ParallelFor(0, n, [&](size_t i) {
                if (i + 1 >= n) return;
                const uint64_t* w = row_words(i);
                const uint64_t nbits = n - 1 - i;
                double acc = 0.0;
                for (uint64_t q = 0; q * 64 < nbits; ++q) {
                    uint64_t bits = w[q];
                    const uint64_t remaining = nbits - q * 64;
                    if (remaining < 64) bits &= (uint64_t{1} << remaining) - 1;
                    while (bits) {
                        const uint64_t t = static_cast<uint64_t>(__builtin_ctzll(bits));
                        acc += x[i + 1 + q * 64 + t];
                        bits &= bits - 1;
                    }
                }
                up[i] = acc;
            });
            // (C^T x)_j = sum_{i<j} C_ij x_i: scatter, parallel over disjoint column blocks.
            constexpr uint64_t BLOCK = 4096;
            ParallelFor(0, (n + BLOCK - 1) / BLOCK, [&](size_t blk) {
                const uint64_t c0 = blk * BLOCK, c1 = std::min<uint64_t>(n, c0 + BLOCK);
                for (uint64_t i = 0; i + 1 < c1; ++i) {
                    const uint64_t j_lo = std::max<uint64_t>(c0, i + 1);
                    if (j_lo >= c1) continue;
                    const uint64_t* w = row_words(i);
                    const double xi = x[i];
                    const uint64_t b_lo = j_lo - (i + 1), b_hi = c1 - (i + 1);  // bit range [b_lo, b_hi)
                    for (uint64_t q = b_lo / 64; q * 64 < b_hi; ++q) {
                        uint64_t bits = w[q];
                        const uint64_t q0 = q * 64;
                        if (b_lo > q0) bits &= ~uint64_t{0} << (b_lo - q0);
                        if (b_hi < q0 + 64) bits &= (uint64_t{1} << (b_hi - q0)) - 1;
                        while (bits) {
                            const uint64_t t = static_cast<uint64_t>(__builtin_ctzll(bits));
                            down[i + 1 + q0 + t] += xi;
                            bits &= bits - 1;
                        }
                    }
                }
            });
            double* y = out->data();
            for (uint64_t i = 0; i < n; ++i) y[i] = k * (up[i] + sg * down[i]);
            out->set_scalar(1.0);
            return;
        }
    }

    // Identity Optimization
    if (m.get_structure() == Structure::IDENTITY) {
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
                    rdst[i] = pycauset::float16_t(z.real());
                    idst[i] = pycauset::float16_t(z.imag());
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
        } else if (auto* res_f16 = dynamic_cast<DenseVector<pycauset::float16_t>*>(&result)) {
            ParallelFor(0, rows, [&](size_t i) {
                res_f16->set(i, pycauset::float16_t(static_cast<float>(v.get_element_as_double(i) * scalar)));
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
                rdst[i] = pycauset::float16_t(sum.real());
                idst[i] = pycauset::float16_t(sum.imag());
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

    // TriangularBitMatrix x DenseVector<int32_t> (scale-first).
    auto* m_tbm = dynamic_cast<const TriangularMatrix<bool>*>(&m);
    if (m_tbm && v_i32 && res_i32 && !m_tbm->is_transposed()) {
        debug_trace::set_last("cpu.matvec.tri_bit_x_i32");
        const uint64_t n = rows;
        const uint64_t* m_data = m_tbm->data();
        const int32_t* v_data = v_i32->data();
        int32_t* res_data = res_i32->data();
        ParallelFor(0, n, [&](size_t i) {
            int64_t sum = 0;
            const uint64_t row_len = (n - 1) - i;
            const uint64_t words = (row_len + 63) / 64;
            const uint64_t word_offset = m_tbm->get_row_offset(i) / 8;
            const uint64_t* row_ptr = m_data + word_offset;
            for (uint64_t w = 0; w < words; ++w) {
                uint64_t word = row_ptr[w];
                while (word) {
                    const uint64_t bit = static_cast<uint64_t>(std::countr_zero(word));
                    const uint64_t j = (i + 1) + w * 64 + bit;
                    if (j < n) {
                        sum = checked_add_int64(sum, static_cast<int64_t>(v_data[j]));
                    }
                    word &= (word - 1);
                }
            }
            if (sum > static_cast<int64_t>((std::numeric_limits<int32_t>::max)()) ||
                sum < static_cast<int64_t>((std::numeric_limits<int32_t>::min)())) {
                throw std::overflow_error("Integer matvec overflow: triangular bit x int32 result does not fit in int32 output");
            }
            res_data[i] = static_cast<int32_t>(sum);
        });
        res_i32->set_scalar(m_tbm->get_scalar() * v_i32->get_scalar());
        return;
    }

    // TriangularBitMatrix x DenseVector<double> (scale-first).
    if (m_tbm && v_f64 && res_f64 && !m_tbm->is_transposed()) {
        debug_trace::set_last("cpu.matvec.tri_bit_x_f64");
        const uint64_t n = rows;
        const uint64_t* m_data = m_tbm->data();
        const double* v_data = v_f64->data();
        double* res_data = res_f64->data();
        ParallelFor(0, n, [&](size_t i) {
            double sum = 0.0;
            const uint64_t row_len = (n - 1) - i;
            const uint64_t words = (row_len + 63) / 64;
            const uint64_t word_offset = m_tbm->get_row_offset(i) / 8;
            const uint64_t* row_ptr = m_data + word_offset;
            for (uint64_t w = 0; w < words; ++w) {
                uint64_t word = row_ptr[w];
                while (word) {
                    const uint64_t bit = static_cast<uint64_t>(std::countr_zero(word));
                    const uint64_t j = (i + 1) + w * 64 + bit;
                    if (j < n) {
                        sum += v_data[j];
                    }
                    word &= (word - 1);
                }
            }
            res_data[i] = sum;
        });
        res_f64->set_scalar(m_tbm->get_scalar() * v_f64->get_scalar());
        return;
    }

    // TriangularBitMatrix x BitVector -> int32 (count set bits).
    if (m_tbm && v_bit && res_int && !m_tbm->is_transposed()) {
        debug_trace::set_last("cpu.matvec.tri_bit_x_bit");
        const uint64_t n = rows;
        const uint64_t* m_data = m_tbm->data();
        int32_t* res_data = res_int->data();
        ParallelFor(0, n, [&](size_t i) {
            int32_t sum = 0;
            const uint64_t row_len = (n - 1) - i;
            const uint64_t words = (row_len + 63) / 64;
            const uint64_t word_offset = m_tbm->get_row_offset(i) / 8;
            const uint64_t* row_ptr = m_data + word_offset;
            for (uint64_t w = 0; w < words; ++w) {
                uint64_t word = row_ptr[w];
                while (word) {
                    const uint64_t bit = static_cast<uint64_t>(std::countr_zero(word));
                    const uint64_t j = (i + 1) + w * 64 + bit;
                    if (j < n && v_bit->get(j)) {
                        sum++;
                    }
                    word &= (word - 1);
                }
            }
            res_data[i] = sum;
        });
        res_int->set_scalar(m_tbm->get_scalar() * v_bit->get_scalar());
        return;
    }

    // Exact integer path (F-051): integer matrix x integer vector -> integer vector, read in the
    // operands' own types and accumulated with wrapping 64-bit arithmetic (NumPy semantics).
    // The generic fallback below goes through double, which loses int64 precision.
    if (pycauset::is_integer_or_bit(result.get_data_type()) && result.get_data_type() != DataType::BIT &&
        m.get_data_type() != DataType::BIT && v.get_data_type() != DataType::BIT &&
        pycauset::is_integer_or_bit(m.get_data_type()) && pycauset::is_integer_or_bit(v.get_data_type()) &&
        v.get_scalar() == std::complex<double>(1.0, 0.0)) {
        StridedReader<int64_t> rm;
        if (make_strided_reader(m, rows, cols, rm)) {
            std::vector<int64_t> x(cols);
            pycauset::visit_vector(v, [&](const auto& tv) {
                using V = std::decay_t<decltype(tv)>;
                if constexpr (!std::is_same_v<V, UnitVector>) {
                    using E = std::decay_t<decltype(tv.get(uint64_t{0}))>;
                    if constexpr (std::is_integral_v<E>) {
                        for (uint64_t j = 0; j < cols; ++j) x[j] = static_cast<int64_t>(tv.get(j));
                    }
                }
            });
            debug_trace::set_last("cpu.matvec.int_exact");
            pycauset::visit_vector(result, [&](auto& out) {
                using V = std::decay_t<decltype(out)>;
                if constexpr (!std::is_same_v<V, UnitVector>) {
                    using R = std::decay_t<decltype(out.get(uint64_t{0}))>;
                    if constexpr (std::is_integral_v<R> && !std::is_same_v<R, bool>) {
                        ParallelFor(0, rows, [&](size_t i) {
                            std::vector<int64_t> row(cols);
                            rm.row(i, 0, cols, row.data());
                            uint64_t acc = 0;
                            for (uint64_t j = 0; j < cols; ++j) {
                                acc += static_cast<uint64_t>(row[j]) * static_cast<uint64_t>(x[j]);
                            }
                            out.set(i, static_cast<R>(acc));
                        });
                    }
                }
            });
            result.set_scalar(1.0);
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
    } else if (auto* res_f16 = dynamic_cast<DenseVector<pycauset::float16_t>*>(&result)) {
        ParallelFor(0, rows, [&](size_t i) {
            float sum = 0.0f;
            for (uint64_t j = 0; j < cols; ++j) {
                sum += static_cast<float>(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res_f16->set(i, pycauset::float16_t(sum));
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

    // Exact integer path (F-051): see matrix_vector_multiply.
    if (pycauset::is_integer_or_bit(result.get_data_type()) && result.get_data_type() != DataType::BIT &&
        m.get_data_type() != DataType::BIT && v.get_data_type() != DataType::BIT &&
        pycauset::is_integer_or_bit(m.get_data_type()) && pycauset::is_integer_or_bit(v.get_data_type()) &&
        v.get_scalar() == std::complex<double>(1.0, 0.0)) {
        StridedReader<int64_t> rm;
        if (make_strided_reader(m, rows, cols, rm)) {
            std::vector<int64_t> x(rows);
            pycauset::visit_vector(v, [&](const auto& tv) {
                using V = std::decay_t<decltype(tv)>;
                if constexpr (!std::is_same_v<V, UnitVector>) {
                    using E = std::decay_t<decltype(tv.get(uint64_t{0}))>;
                    if constexpr (std::is_integral_v<E>) {
                        for (uint64_t r = 0; r < rows; ++r) x[r] = static_cast<int64_t>(tv.get(r));
                    }
                }
            });
            debug_trace::set_last("cpu.vecmat.int_exact");
            pycauset::visit_vector(result, [&](auto& out) {
                using V = std::decay_t<decltype(out)>;
                if constexpr (!std::is_same_v<V, UnitVector>) {
                    using R = std::decay_t<decltype(out.get(uint64_t{0}))>;
                    if constexpr (std::is_integral_v<R> && !std::is_same_v<R, bool>) {
                        constexpr uint64_t TILE = 256;
                        ParallelFor(0, (cols + TILE - 1) / TILE, [&](size_t t) {
                            const uint64_t j0 = t * TILE, j1 = std::min<uint64_t>(cols, j0 + TILE);
                            uint64_t acc[TILE] = {};
                            int64_t buf[TILE];
                            for (uint64_t r = 0; r < rows; ++r) {
                                rm.row(r, j0, j1, buf);
                                const uint64_t xr = static_cast<uint64_t>(x[r]);
                                for (uint64_t k = 0; k < j1 - j0; ++k) acc[k] += xr * static_cast<uint64_t>(buf[k]);
                            }
                            for (uint64_t k = 0; k < j1 - j0; ++k) out.set(j0 + k, static_cast<R>(acc[k]));
                        });
                    }
                }
            });
            result.set_scalar(1.0);
            return;
        }
    }

    // Identity Optimization
    if (m.get_structure() == Structure::IDENTITY) {
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
                    rdst[i] = pycauset::float16_t(z.real());
                    idst[i] = pycauset::float16_t(z.imag());
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
        } else if (auto* res_f16 = dynamic_cast<DenseVector<pycauset::float16_t>*>(&result)) {
            ParallelFor(0, cols, [&](size_t i) {
                res_f16->set(i, pycauset::float16_t(static_cast<float>(v.get_element_as_double(i) * scalar)));
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
                rdst[j] = pycauset::float16_t(sum.real());
                idst[j] = pycauset::float16_t(sum.imag());
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
    } else if (auto* res_f16 = dynamic_cast<DenseVector<pycauset::float16_t>*>(&result)) {
        ParallelFor(0, cols, [&](size_t j) {
            float sum = 0.0f;
            for (uint64_t i = 0; i < rows; ++i) {
                sum += static_cast<float>(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            res_f16->set(j, pycauset::float16_t(sum));
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
                rdst[idx] = pycauset::float16_t(vc.real());
                idst[idx] = pycauset::float16_t(vc.imag());
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
    } else if (auto* m = dynamic_cast<DenseMatrix<pycauset::float16_t>*>(&result)) {
        ParallelFor(0, n, [&](size_t i) {
            const float val_a = static_cast<float>(a.get_element_as_double(i));
            for (uint64_t j = 0; j < n; ++j) {
                const float v = val_a * static_cast<float>(b.get_element_as_double(j));
                m->set(i, j, pycauset::float16_t(v));
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



LuPivots CpuSolver::lu_pivots(const MatrixBase& matrix) {
    // The entry points (math/Eigen.cpp) pass plain storage; the structured shortcuts read
    // through the accessors, which apply any view anyway. Complex inputs keep their imaginary
    // parts (F-059).
    const uint64_t n = matrix.rows();
    if (n != matrix.cols()) {
        throw std::runtime_error("Determinant requires square matrix");
    }
    const auto type = matrix.get_structure();
    const auto dt = matrix.get_data_type();
    const bool use_complex = dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 ||
                             dt == DataType::COMPLEX_FLOAT64 || matrix.get_scalar().imag() != 0.0;
    LuPivots out;
    out.pivots.resize(n);

    if (type == Structure::IDENTITY) {
        std::fill(out.pivots.begin(), out.pivots.end(), matrix.get_scalar());
        return out;
    }
    const bool strict_triangular =
        type == Structure::TRIANGULAR && !static_cast<const TriangularMatrixBase&>(matrix).has_diagonal();
    if (type == Structure::DIAGONAL || type == Structure::TRIANGULAR) {
        // Every element type (F-059); a strict triangle has a zero diagonal.
        for (uint64_t i = 0; i < n; ++i) {
            out.pivots[i] = strict_triangular ? std::complex<double>(0.0) : matrix.get_element_as_complex(i, i);
        }
        return out;
    }

    // General case: LU (getrf). The row-major buffer is passed as column-major, i.e. A^T is
    // factorized: det(A^T) = det(A), and LAPACKE does not need its transposed copy.
    const auto ni = static_cast<lapack_int>(n);
    std::vector<lapack_int> ipiv(n);
    auto record = [&](auto&& diag_at) {
        for (uint64_t i = 0; i < n; ++i) {
            out.pivots[i] = diag_at(i);
            if (ipiv[i] != static_cast<lapack_int>(i + 1)) out.odd_permutation = !out.odd_permutation;
        }
    };
    if (use_complex) {
        auto data = to_memory_flat_complex_square(matrix);  // folds factor and conjugation
        const lapack_int info = LAPACKE_zgetrf(LAPACK_COL_MAJOR, ni, ni,
            reinterpret_cast<lapack_complex_double*>(data.data()), ni, ipiv.data());
        if (info < 0) throw std::runtime_error("LAPACK zgetrf: illegal argument");
        record([&](uint64_t i) { return data[i * n + i]; });  // info > 0: an exact zero pivot
        return out;
    }
    auto data = to_memory_flat_real_square(matrix);  // folds the factor
    const lapack_int info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, ni, ni, data.data(), ni, ipiv.data());
    if (info < 0) throw std::runtime_error("LAPACK dgetrf: illegal argument");
    record([&](uint64_t i) { return std::complex<double>(data[i * n + i], 0.0); });
    return out;
}



// --- Eigen Solvers ---

namespace {
    // Helper to extract real parts into a vector
    template <typename T>
    void extract_real_vector(uint64_t n, const std::vector<T>& src, VectorBase& dst) {
        if (auto* d_dst = dynamic_cast<DenseVector<double>*>(&dst)) {
            double* ptr = d_dst->data();
            for(size_t i=0; i<n; ++i) ptr[i] = static_cast<double>(std::real(src[i]));
        } else if (auto* f_dst = dynamic_cast<DenseVector<float>*>(&dst)) {
            float* ptr = f_dst->data();
            for(size_t i=0; i<n; ++i) ptr[i] = static_cast<float>(std::real(src[i]));
        } else {
            throw std::runtime_error("Unsupported vector type for eigenvalues");
        }
    }

    template <typename T>
    void extract_vector(uint64_t n, const T* src, VectorBase& dst) {
        if (auto* d_dst = dynamic_cast<DenseVector<double>*>(&dst)) {
            double* ptr = d_dst->data();
            for(size_t i=0; i<n; ++i) ptr[i] = static_cast<double>(src[i]);
        } else if (auto* f_dst = dynamic_cast<DenseVector<float>*>(&dst)) {
            float* ptr = f_dst->data();
            for(size_t i=0; i<n; ++i) ptr[i] = static_cast<float>(src[i]);
        } else {
            throw std::runtime_error("Unsupported vector type for eigenvalues");
        }
    }

    template <typename T>
    void extract_complex_vector(uint64_t n, const T* wr, const T* wi, VectorBase& dst) {
        if (auto* d_dst = dynamic_cast<DenseVector<std::complex<double>>*>(&dst)) {
            std::complex<double>* ptr = d_dst->data();
            for(size_t i=0; i<n; ++i) {
                ptr[i] = { static_cast<double>(wr[i]), static_cast<double>(wi[i]) };
            }
        } else if (auto* f_dst = dynamic_cast<DenseVector<std::complex<float>>*>(&dst)) {
            std::complex<float>* ptr = f_dst->data();
            for(size_t i=0; i<n; ++i) {
               ptr[i] = { static_cast<float>(wr[i]), static_cast<float>(wi[i]) };
            }
        } else {
             throw std::runtime_error("eig/eigvals requires complex vector output");
        }
    }
}

void CpuSolver::eigh(const MatrixBase& in, VectorBase& eigenvalues, MatrixBase& eigenvectors, char uplo) {
    if (in.rows() != in.cols()) throw std::invalid_argument("eigh requires square matrix");
    
    // We assume input is Symmetric/Hermitian.
    // LAPACK dsyev destroys input matrix to produce eigenvectors.
    // So we copy IN -> Eigenvectors matrix first.
    
    // 1. Copy in -> eigenvectors
    // Use the existing 'fill' or 'set' or a dedicated copy.
    // Since 'eigenvectors' is MatrixBase, easiest is to use our copy mechanisms or simple elementwise copy if types vary,
    // but here we expect Dense Output.
    
    const uint64_t n = in.rows();
    if (eigenvectors.rows() != n || eigenvectors.cols() != n) throw std::invalid_argument("Eigenvectors matrix bad shape");
    if (eigenvalues.size() != n) throw std::invalid_argument("Eigenvalues vector bad size");

    // Type dispatch
    if (auto* out_d = dynamic_cast<DenseMatrix<double>*>(&eigenvectors)) {
        if (auto* in_d = dynamic_cast<const DenseMatrix<double>*>(&in)) {
             // 1. Copy
             // Respecting offsets!
             const double* src = in_d->data() + (in_d->row_offset() * in_d->base_cols() + in_d->col_offset());
             double* dst = out_d->data() + (out_d->row_offset() * out_d->base_cols() + out_d->col_offset());
             
             uint64_t src_stride = in_d->base_cols();
             uint64_t dst_stride = out_d->base_cols();
             
             ParallelFor(0, n, [&](size_t i) {
                 for(size_t j=0; j<n; ++j) {
                     dst[i * dst_stride + j] = src[i * src_stride + j];
                 }
             });
             
             // 2. Run dsyev
             // JOBZ = 'V' (Compute eigenvectors)
             // UPLO = 'U' (Upper triangle assumed)
             lapack_int info;
             
             // Output eigenvalues 'w' needs a buffer
             std::vector<double> w(n);
             
             info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', uplo, (lapack_int)n, dst, (lapack_int)dst_stride, w.data());
             
             if (info > 0) throw std::runtime_error("eigh failed to converge");
             if (info < 0) throw std::runtime_error("eigh: illegal argument");
             
             // 3. Copy w -> eigenvalues
             extract_vector(n, w.data(), eigenvalues);
             return;
        }
    }
    
     if (auto* out_f = dynamic_cast<DenseMatrix<float>*>(&eigenvectors)) {
        if (auto* in_f = dynamic_cast<const DenseMatrix<float>*>(&in)) {
             const float* src = in_f->data() + (in_f->row_offset() * in_f->base_cols() + in_f->col_offset());
             float* dst = out_f->data() + (out_f->row_offset() * out_f->base_cols() + out_f->col_offset());
             
             uint64_t src_stride = in_f->base_cols();
             uint64_t dst_stride = out_f->base_cols();
             
             ParallelFor(0, n, [&](size_t i) {
                 for(size_t j=0; j<n; ++j) {
                     dst[i * dst_stride + j] = src[i * src_stride + j];
                 }
             });
             
             std::vector<float> w(n);
             lapack_int info = LAPACKE_ssyev(LAPACK_ROW_MAJOR, 'V', uplo, (lapack_int)n, dst, (lapack_int)dst_stride, w.data());
             
             if (info > 0) throw std::runtime_error("eigh failed to converge");
             if (info < 0) throw std::runtime_error("eigh: illegal argument");
             
             extract_vector(n, w.data(), eigenvalues);
             return;
        }
    }
    
    throw std::runtime_error("eigh not implemented for these types");
}

void CpuSolver::eigvalsh(const MatrixBase& in, VectorBase& eigenvalues, char uplo) {
    if (in.rows() != in.cols()) throw std::invalid_argument("eigvalsh requires square matrix");
    const uint64_t n = in.rows();
    if (eigenvalues.size() != n) throw std::invalid_argument("Eigenvalues vector bad size");
    
    // JOBZ = 'N' (Eigenvalues only)
    // We still need a copy of A because dsyev destroys it.
    
    if (auto* in_d = dynamic_cast<const DenseMatrix<double>*>(&in)) {
         // Create Temp DenseMatrix. Default constructor makes it anonymous/temp.
         DenseMatrix<double> temp(n, n);
         
         const double* src = in_d->data() + (in_d->row_offset() * in_d->base_cols() + in_d->col_offset());
         double* dst = temp.data(); // No offsets for new temp
         uint64_t src_stride = in_d->base_cols();
         uint64_t dst_stride = n;
         
         ParallelFor(0, n, [&](size_t i) {
             for(size_t j=0; j<n; ++j) {
                  dst[i * dst_stride + j] = src[i * src_stride + j];
             }
         });
         
         std::vector<double> w(n);
         lapack_int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', uplo, (lapack_int)n, dst, (lapack_int)dst_stride, w.data());
         
         if (info > 0) throw std::runtime_error("eigvalsh failed to converge");
         if (info < 0) throw std::runtime_error("eigvalsh: illegal argument");
         
         extract_vector(n, w.data(), eigenvalues);
         return;
    }
    
    if (auto* in_f = dynamic_cast<const DenseMatrix<float>*>(&in)) {
         DenseMatrix<float> temp(n, n);
         
         const float* src = in_f->data() + (in_f->row_offset() * in_f->base_cols() + in_f->col_offset());
         float* dst = temp.data(); 
         uint64_t src_stride = in_f->base_cols();
         uint64_t dst_stride = n;
         
         ParallelFor(0, n, [&](size_t i) {
             for(size_t j=0; j<n; ++j) {
                  dst[i * dst_stride + j] = src[i * src_stride + j];
             }
         });
         
         std::vector<float> w(n);
         lapack_int info = LAPACKE_ssyev(LAPACK_ROW_MAJOR, 'N', uplo, (lapack_int)n, dst, (lapack_int)dst_stride, w.data());
         
         if (info > 0) throw std::runtime_error("eigvalsh failed to converge");
         if (info < 0) throw std::runtime_error("eigvalsh: illegal argument");
         
         extract_vector(n, w.data(), eigenvalues);
         return;
    }

    throw std::runtime_error("eigvalsh not implemented for these types");
}

void CpuSolver::eig(const MatrixBase& in, VectorBase& eigenvalues, MatrixBase& eigenvectors) {
     if (in.rows() != in.cols()) throw std::invalid_argument("eig requires square matrix");
     
     const uint64_t n = in.rows();
     if (eigenvectors.rows() != n || eigenvectors.cols() != n) throw std::invalid_argument("Eigenvectors matrix bad shape");
     if (eigenvalues.size() != n) throw std::invalid_argument("Eigenvalues vector bad size");
     
     // 1. Prepare Inputs (Double)
     if (auto* in_d = dynamic_cast<const DenseMatrix<double>*>(&in)) {
         DenseMatrix<double> temp(n, n);
         // Copy with offsets
         const double* src = in_d->data() + (in_d->row_offset() * in_d->base_cols() + in_d->col_offset());
         double* dst = temp.data(); 
         uint64_t src_stride = in_d->base_cols();
         uint64_t dst_stride = n;
         ParallelFor(0, n, [&](size_t i) {
             for(size_t j=0; j<n; ++j) {
                  dst[i * dst_stride + j] = src[i * src_stride + j];
             }
         });
         
         std::vector<double> wr(n);
         std::vector<double> wi(n);
         
         // VR needs to be strict n*n double buffer for LAPACK, but result is complex.
         // We essentially need a temporary double buffer for VR.
         std::vector<double> vr(n * n);
         
         lapack_int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', (lapack_int)n, dst, (lapack_int)dst_stride, 
                                         wr.data(), wi.data(), 
                                         nullptr, (lapack_int)n, 
                                         vr.data(), (lapack_int)n);
                                         
         if (info > 0) throw std::runtime_error("eig failed to compute eigenvalues");
         if (info < 0) throw std::runtime_error("eig: illegal argument");
         
         // 2. Extract Eigenvalues
         extract_complex_vector(n, wr.data(), wi.data(), eigenvalues);
         
         // 3. Extract Eigenvectors
         if (auto* out_cd = dynamic_cast<DenseMatrix<std::complex<double>>*>(&eigenvectors)) {
             std::complex<double>* out_ptr = out_cd->data(); // Assume contiguous for now, or use set().
             // Using direct pointer for speed, assuming contiguous output (safe for fresh result matrix)
             
             // Check contiguity or use loops
             bool contiguous = (out_cd->row_offset() == 0 && out_cd->col_offset() == 0 && out_cd->base_cols() == n);
             
             for (size_t j = 0; j < n; ++j) {
                 if (wi[j] == 0.0) {
                     // Real eigenvalue -> Real eigenvector at column j
                     for (size_t i = 0; i < n; ++i) {
                         std::complex<double> val = { vr[i * n + j], 0.0 };
                         if (contiguous) out_ptr[i * n + j] = val;
                         else out_cd->set(i, j, val); 
                     }
                 } else {
                     // Complex pair w[j], w[j+1]
                     // v[j] = VR[:, j] + i*VR[:, j+1]
                     // v[j+1] = VR[:, j] - i*VR[:, j+1]
                     
                     for (size_t i = 0; i < n; ++i) {
                         double re = vr[i * n + j];
                         double im = vr[i * n + (j + 1)];
                         
                         std::complex<double> v1 = {re, im};
                         std::complex<double> v2 = {re, -im};
                         
                         if (contiguous) {
                             out_ptr[i * n + j] = v1;
                             out_ptr[i * n + (j+1)] = v2;
                         } else {
                             out_cd->set(i, j, v1);
                             out_cd->set(i, j+1, v2);
                         }
                     }
                     
                     j++; // Skip next
                 }
             }
             return;
         }
         throw std::runtime_error("eig requires a complex_float64 output for float64 input");
     }
     
     // Float Implementation
     if (auto* in_f = dynamic_cast<const DenseMatrix<float>*>(&in)) {
         DenseMatrix<float> temp(n, n);
         const float* src = in_f->data() + (in_f->row_offset() * in_f->base_cols() + in_f->col_offset());
         float* dst = temp.data(); 
         uint64_t src_stride = in_f->base_cols();
         uint64_t dst_stride = n;
         ParallelFor(0, n, [&](size_t i) {
             for(size_t j=0; j<n; ++j) dst[i * dst_stride + j] = src[i * src_stride + j];
         });
         
         std::vector<float> wr(n), wi(n), vr(n * n);
         
         lapack_int info = LAPACKE_sgeev(LAPACK_ROW_MAJOR, 'N', 'V', (lapack_int)n, dst, (lapack_int)dst_stride, 
                                         wr.data(), wi.data(), 
                                         nullptr, (lapack_int)n, 
                                         vr.data(), (lapack_int)n);
                                         
         if (info > 0) throw std::runtime_error("eig failed to compute eigenvalues");

         extract_complex_vector(n, wr.data(), wi.data(), eigenvalues);
         
         if (auto* out_cf = dynamic_cast<DenseMatrix<std::complex<float>>*>(&eigenvectors)) {
             std::complex<float>* out_ptr = out_cf->data();
             bool contiguous = (out_cf->row_offset() == 0 && out_cf->col_offset() == 0 && out_cf->base_cols() == n);
             
             for (size_t j = 0; j < n; ++j) {
                 if (wi[j] == 0.0f) {
                     for (size_t i = 0; i < n; ++i) {
                         std::complex<float> val = { vr[i * n + j], 0.0f };
                         if (contiguous) out_ptr[i * n + j] = val;
                         else out_cf->set(i, j, val);
                     }
                 } else {
                     for (size_t i = 0; i < n; ++i) {
                         float re = vr[i * n + j];
                         float im = vr[i * n + (j + 1)];
                         std::complex<float> v1 = {re, im};
                         std::complex<float> v2 = {re, -im};
                         if (contiguous) {
                             out_ptr[i * n + j] = v1;
                             out_ptr[i * n + (j+1)] = v2;
                         } else {
                             out_cf->set(i, j, v1);
                             out_cf->set(i, j+1, v2);
                         }
                     }
                     j++;
                 }
             }
             return;
         }
         throw std::runtime_error("eig requires a complex_float32 output for float32 input");
     }

    throw std::runtime_error("eig not implemented for these types");
}

void CpuSolver::eigvals(const MatrixBase& in, VectorBase& eigenvalues) {
     if (in.rows() != in.cols()) throw std::invalid_argument("eigvals requires square matrix");
     
     const uint64_t n = in.rows();
     if (eigenvalues.size() != n) throw std::invalid_argument("Eigenvalues vector bad size");
     
     if (auto* in_d = dynamic_cast<const DenseMatrix<double>*>(&in)) {
         DenseMatrix<double> temp(n, n);
         const double* src = in_d->data() + (in_d->row_offset() * in_d->base_cols() + in_d->col_offset());
         double* dst = temp.data(); 
         uint64_t src_stride = in_d->base_cols();
         uint64_t dst_stride = n;
         ParallelFor(0, n, [&](size_t i) {
             for(size_t j=0; j<n; ++j) dst[i * dst_stride + j] = src[i * src_stride + j];
         });
         
         std::vector<double> wr(n), wi(n);
         
         lapack_int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'N', (lapack_int)n, dst, (lapack_int)dst_stride, 
                                         wr.data(), wi.data(), 
                                         nullptr, (lapack_int)n, 
                                         nullptr, (lapack_int)n);
         
         if (info > 0) throw std::runtime_error("eigvals failed");
         extract_complex_vector(n, wr.data(), wi.data(), eigenvalues);
         return;
     }
     
     if (auto* in_f = dynamic_cast<const DenseMatrix<float>*>(&in)) {
         DenseMatrix<float> temp(n, n);
         const float* src = in_f->data() + (in_f->row_offset() * in_f->base_cols() + in_f->col_offset());
         float* dst = temp.data(); 
         uint64_t src_stride = in_f->base_cols();
         uint64_t dst_stride = n;
         ParallelFor(0, n, [&](size_t i) {
             for(size_t j=0; j<n; ++j) dst[i * dst_stride + j] = src[i * src_stride + j];
         });
         
         std::vector<float> wr(n), wi(n);
         lapack_int info = LAPACKE_sgeev(LAPACK_ROW_MAJOR, 'N', 'N', (lapack_int)n, dst, (lapack_int)dst_stride, 
                                         wr.data(), wi.data(), 
                                         nullptr, (lapack_int)n, 
                                         nullptr, (lapack_int)n);
         if (info > 0) throw std::runtime_error("eigvals failed");
         extract_complex_vector(n, wr.data(), wi.data(), eigenvalues);
         return;
     }

     throw std::runtime_error("eigvals not implemented for these types");
}


void CpuSolver::qr(const MatrixBase& in, MatrixBase& Q, MatrixBase& R) {
    // QR Decomposition using LAPACKE_xgeqrf and xorgqr/xungqr
    // A = Q * R.
    // Dimensions: A is (M, N).
    // Q is (M, K) where K = min(M, N). (Reduced QR)
    // R is (K, N).
    
    uint64_t m = in.rows();
    uint64_t n = in.cols();
    uint64_t k = std::min(m, n);
    
    if (Q.rows() != m || Q.cols() != k) throw std::invalid_argument("QR: Q shape incorrect");
    if (R.rows() != k || R.cols() != n) throw std::invalid_argument("QR: R shape incorrect");

    if (auto* in_d = dynamic_cast<const DenseMatrix<double>*>(&in)) {
         // Copy A to a temp buffer (needed for geqrf)
         // Since Q will eventually hold the orthogonal matrix, we can use Q's buffer if size allows (M x M), 
         // but here Q is (M, K). geqrf works on (M, N). 
         // So we allocate a copy of A.
         std::vector<double> a_data = to_memory_flat_real_square(in); // Wait, this helper enforces square.
         // Let's just do a manual copy for general rectangular.
         
         std::vector<double> a_buf(m * n);
         // We must handle strides properly
         ParallelFor(0, m, [&](size_t i) {
             for(size_t j=0; j<n; ++j) {
                 a_buf[i * n + j] = in_d->get(i, j);
             }
         });
         
         std::vector<double> tau(k);
         
         // 1. GEQRF
         lapack_int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, (lapack_int)m, (lapack_int)n, a_buf.data(), (lapack_int)n, tau.data());
         if (info != 0) throw std::runtime_error("geqrf failed");
         
         // 2. Extract R
         // R is upper triangular part of A (which is now in a_buf).
         // R is K x N.
         auto* r_d = dynamic_cast<DenseMatrix<double>*>(&R);
         if (!r_d) throw std::runtime_error("R must be DenseMatrix<double>");
         
         // Zero out R first?
         // Actually we just copy the upper trapezoid
         double* r_ptr = r_d->data();
         ParallelFor(0, k, [&](size_t i) {
             for(size_t j=0; j<n; ++j) {
                 if (j >= i) r_ptr[i * n + j] = a_buf[i * n + j];
                 else r_ptr[i * n + j] = 0.0;
             }
         });
         
         // 3. Generate Q
         // Q is generated by replacing the columns of A with the reflectors and calling orgqr.
         // But geqrf overwrote A.
         // Wait, geqrf overwrites A with elementary reflectors in the lower part-ish.
         // We can use orgqr on a_buf to generate Q.
         // BUT orgqr replaces A with Q.
         // A was MxN. We want Q to be MxK (reduced).
         // info = LAPACKE_dorgqr(matrix_layout, m, n, k, a, lda, tau)
         // If we want Q to be MxK, we pass n=K.
         // But the input A buffer must be large enough. A is MxN.
         // If N > K, we are shrinking.
         // If M > N, K=N.
         
         // Let's create a buffer for Q generation that is MxK, initialized from columns 0..K-1 of a_buf??
         // No, dorgqr expects the details in the 'a' arguments corresponding to reflectors.
         // reflectors are stored in lower part of A.
         
         // If we want "Thin" QR (Q is MxK), we call dorgqr with n=K.
         // But we must preserve the reflectors which are in the first K columns of A.
         
         // So we can reuse a_buf if N >= K (always true).
         // However, dorgqr overwrites a_buf.
         info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, (lapack_int)m, (lapack_int)k, (lapack_int)k, a_buf.data(), (lapack_int)n, tau.data());
         if (info != 0) throw std::runtime_error("dorgqr failed");
         
         // Now a_buf (first K columns) contains Q.
         // Copy to Q matrix.
         auto* q_d = dynamic_cast<DenseMatrix<double>*>(&Q);
         if (!q_d) throw std::runtime_error("Q must be DenseMatrix<double>");
         double* q_ptr = q_d->data();
         
         ParallelFor(0, m, [&](size_t i) {
             for(size_t j=0; j<k; ++j) {
                 q_ptr[i * k + j] = a_buf[i * n + j];
             }
         });
         
         return;
    }
    
    // Fallback for float
    if (auto* in_f = dynamic_cast<const DenseMatrix<float>*>(&in)) {
         std::vector<float> a_buf(m * n);
         for(size_t i=0; i<m; ++i) {
             for(size_t j=0; j<n; ++j) {
                 a_buf[i * n + j] = in_f->get(i, j);
             }
         }
         
         std::vector<float> tau(k);
         lapack_int info = LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, (lapack_int)m, (lapack_int)n, a_buf.data(), (lapack_int)n, tau.data());
         if (info != 0) throw std::runtime_error("sgeqrf failed");
         
         auto* r_f = dynamic_cast<DenseMatrix<float>*>(&R);
         float* r_ptr = r_f->data();
         for(size_t i=0; i<k; ++i) {
             for(size_t j=0; j<n; ++j) {
                 if (j >= i) r_ptr[i * n + j] = a_buf[i * n + j];
                 else r_ptr[i * n + j] = 0.0f;
             }
         }
         
         info = LAPACKE_sorgqr(LAPACK_ROW_MAJOR, (lapack_int)m, (lapack_int)k, (lapack_int)k, a_buf.data(), (lapack_int)n, tau.data());
         if (info != 0) throw std::runtime_error("sorgqr failed");
         
         auto* q_f = dynamic_cast<DenseMatrix<float>*>(&Q);
         float* q_ptr = q_f->data();
         for(size_t i=0; i<m; ++i) {
             for(size_t j=0; j<k; ++j) {
                 q_ptr[i * k + j] = a_buf[i * n + j];
             }
         }
         return;
    }
    
    throw std::runtime_error("QR only implemented for float/double");
}

void CpuSolver::lu(const MatrixBase& in, MatrixBase& P, MatrixBase& L, MatrixBase& U) {
    // LU Decomposition: A = P * L * U
    // A: (M, N)
    // P: (M, M) - Permutation matrix
    // L: (M, K) - Unit Lower Triangular (K = min(M,N))
    // U: (K, N) - Upper Triangular
    
    uint64_t m = in.rows();
    uint64_t n = in.cols();
    uint64_t k = std::min(m, n);
    
    if (auto* in_d = dynamic_cast<const DenseMatrix<double>*>(&in)) {
        // Copy to working buffer (row-major); dgetrf factorizes in place.
        std::vector<double> a_buf(m * n);
        for(size_t i=0; i<m; ++i) for(size_t j=0; j<n; ++j) a_buf[i*n + j] = in_d->get(i, j);

        std::vector<lapack_int> ipiv(k);
        lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, (lapack_int)m, (lapack_int)n,
                                         a_buf.data(), (lapack_int)n, ipiv.data());
        if (info < 0) throw std::runtime_error("dgetrf: illegal argument");
        // info > 0 => U(info,info) exactly zero (singular); the LU factorisation is
        // still valid and returned (matches the float path's sgetrf behaviour).

        // U = upper triangular part (K x N)
        auto* u_out = dynamic_cast<DenseMatrix<double>*>(&U);
        double* u_ptr = u_out->data();
        std::fill(u_ptr, u_ptr + k * n, 0.0);
        for(size_t i=0; i<k; ++i) {
            for (size_t j=i; j<n; ++j) u_ptr[i*n + j] = a_buf[i*n + j];
        }

        // L = unit lower triangular (M x K)
        auto* l_out = dynamic_cast<DenseMatrix<double>*>(&L);
        double* l_ptr = l_out->data();
        std::fill(l_ptr, l_ptr + m * k, 0.0);
        for(size_t i=0; i<m; ++i) {
            for(size_t j=0; j<k && j<=i; ++j) {
                if (i == j) l_ptr[i*k + j] = 1.0;
                else l_ptr[i*k + j] = a_buf[i*n + j];
            }
        }

        // P = permutation matrix from ipiv (row-swap semantics)
        auto* p_out = dynamic_cast<DenseMatrix<double>*>(&P);
        double* p_ptr = p_out->data();
        std::fill(p_ptr, p_ptr + m * m, 0.0);
        std::vector<int> p_indices(m);
        for(size_t i=0; i<m; ++i) p_indices[i] = (int)i;
        for(size_t i=0; i<k; ++i) {
            std::swap(p_indices[i], p_indices[ipiv[i] - 1]);
        }
        for(size_t i=0; i<m; ++i) {
            p_ptr[p_indices[i] * m + i] = 1.0;
        }
        return;
    }
    
    // Float fallback omitted for brevity but follows same pattern...
    if (auto* in_f = dynamic_cast<const DenseMatrix<float>*>(&in)) {
         std::vector<float> a_buf(m * n);
         for(size_t i=0; i<m; ++i) 
             for(size_t j=0; j<n; ++j) 
                 a_buf[i * n + j] = in_f->get(i, j);

        std::vector<int> ipiv(std::min(m, n));
        lapack_int info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, (lapack_int)m, (lapack_int)n, a_buf.data(), (lapack_int)n, (lapack_int*)ipiv.data());
        if (info < 0) throw std::runtime_error("sgetrf: illegal argument");
        
        auto* u_f = dynamic_cast<DenseMatrix<float>*>(&U);
        float* u_ptr = u_f->data();
        std::fill(u_ptr, u_ptr + k * n, 0.0f);
        for(size_t i=0; i<k; ++i) {
            for (size_t j=i; j<n; ++j) u_ptr[i * n + j] = a_buf[i * n + j];
        }
        
        auto* l_f = dynamic_cast<DenseMatrix<float>*>(&L);
        float* l_ptr = l_f->data();
        std::fill(l_ptr, l_ptr + m * k, 0.0f);
        for(size_t i=0; i<m; ++i) {
            for(size_t j=0; j<k && j<=i; ++j) {
                if (i == j) l_ptr[i * k + j] = 1.0f;
                else l_ptr[i * k + j] = a_buf[i * n + j];
            }
        }
        
        auto* p_f = dynamic_cast<DenseMatrix<float>*>(&P);
        float* p_ptr = p_f->data();
        std::fill(p_ptr, p_ptr + m * m, 0.0f);
        std::vector<int> p_indices(m);
        for(int i=0; i<(int)m; ++i) p_indices[i] = i;
        
        for(size_t i=0; i<std::min(m, n); ++i) {
            std::swap(p_indices[i], p_indices[ipiv[i] - 1]);
        }
        
        for(size_t i=0; i<m; ++i) {
            p_ptr[p_indices[i] * m + i] = 1.0f;
        }
        return;
    }
    
    throw std::runtime_error("LU only implemented for float/double");
}

void CpuSolver::svd(const MatrixBase& in, MatrixBase& U, VectorBase& S, MatrixBase& VT) {
    // SVD: A = U * S * VT
    // A: (M, N)
    // U: (M, K) - K=min(M,N)
    // S: (K,)
    // VT: (K, N)
    // Using GESVD with JOBU='S' (return first min(M,N) columns of U) and JOBVT='S' (return first min(M,N) rows of VT).
    
    uint64_t m = in.rows();
    uint64_t n = in.cols();
    // LAPACKE_dgesvd
    
    if (auto* in_d = dynamic_cast<const DenseMatrix<double>*>(&in)) {
         std::vector<double> a_buf(m * n);
         if (!in_d->is_transposed() && !in_d->is_subview()) {
             std::memcpy(a_buf.data(), in_d->data(), m * n * sizeof(double));
         } else {
             for(size_t i=0; i<m; ++i)
                 for(size_t j=0; j<n; ++j)
                     a_buf[i * n + j] = in_d->get(i, j);
         }
                 
         // Output buffers
         auto* u_d = dynamic_cast<DenseMatrix<double>*>(&U);
         auto* vt_d = dynamic_cast<DenseMatrix<double>*>(&VT);
         
         // S is a vector
         // But VectorBase access is generic. Let's cast to DenseVector<double>.
         // Or if S is different type? We assumed it matches 'in' type's float variant.
         // Let's assume DenseVector<double> or similar.
         // We'll calculate into a std::vector then copy.
         
         uint64_t k = std::min(m, n);
         std::vector<double> s_vec(k);
         
         // dgesdd (divide-and-conquer) is far faster than dgesvd (QR-based) and is
         // what NumPy uses by default.
         lapack_int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S',
             (lapack_int)m, (lapack_int)n, a_buf.data(), (lapack_int)n,
             s_vec.data(),
             u_d->data(), (lapack_int)k,
             vt_d->data(), (lapack_int)n);
         
         if (info > 0) throw std::runtime_error("dgesdd did not converge");
         if (info < 0) throw std::runtime_error("dgesdd: illegal argument");
         
         // Copy S to VectorBase
         extract_vector(k, s_vec.data(), S);
         return;
    }
    
    if (auto* in_f = dynamic_cast<const DenseMatrix<float>*>(&in)) {
         std::vector<float> a_buf(m * n);
         if (!in_f->is_transposed() && !in_f->is_subview()) {
             std::memcpy(a_buf.data(), in_f->data(), m * n * sizeof(float));
         } else {
             for(size_t i=0; i<m; ++i) for(size_t j=0; j<n; ++j) a_buf[i * n + j] = in_f->get(i, j);
         }
         
         auto* u_f = dynamic_cast<DenseMatrix<float>*>(&U);
         auto* vt_f = dynamic_cast<DenseMatrix<float>*>(&VT);
         uint64_t k = std::min(m, n);
         std::vector<float> s_vec(k);
         
         lapack_int info = LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'S',
             (lapack_int)m, (lapack_int)n, a_buf.data(), (lapack_int)n,
             s_vec.data(),
             u_f->data(), (lapack_int)k,
             vt_f->data(), (lapack_int)n);
         
         if (info > 0) throw std::runtime_error("sgesdd did not converge");
         if (info < 0) throw std::runtime_error("sgesdd: illegal argument");
         
         extract_vector(k, s_vec.data(), S);
         return;
    }

    throw std::runtime_error("SVD only implemented for float/double");
}

namespace {
// A X = B for dense row-major storage of one dtype (LAPACK ?gesv: LU with partial pivoting,
// the routine NumPy's np.linalg.solve uses).
template <typename T>
lapack_int gesv(lapack_int n, lapack_int nrhs, T* a, lapack_int* ipiv, T* b) {
    if constexpr (std::is_same_v<T, double>) return LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, a, n, ipiv, b, nrhs);
    else if constexpr (std::is_same_v<T, float>) return LAPACKE_sgesv(LAPACK_ROW_MAJOR, n, nrhs, a, n, ipiv, b, nrhs);
    else if constexpr (std::is_same_v<T, std::complex<double>>) return LAPACKE_zgesv(LAPACK_ROW_MAJOR, n, nrhs, a, n, ipiv, b, nrhs);
    else return LAPACKE_cgesv(LAPACK_ROW_MAJOR, n, nrhs, a, n, ipiv, b, nrhs);
}

template <typename T>
bool solve_dense(const MatrixBase& a, const MatrixBase& b, MatrixBase& x) {
    auto* A = dynamic_cast<const DenseMatrix<T>*>(&a);
    auto* B = dynamic_cast<const DenseMatrix<T>*>(&b);
    auto* X = dynamic_cast<DenseMatrix<T>*>(&x);
    if (!A || !B || !X) return false;
    debug_trace::set_last("cpu.solve.gesv");
    const uint64_t n = a.rows(), nrhs = b.cols();
    std::vector<T> wa(A->data(), A->data() + n * n);  // gesv overwrites A with its LU factors
    std::vector<lapack_int> ipiv(n);
    T* xp = X->data();
    std::copy(B->data(), B->data() + n * nrhs, xp);
    const lapack_int info = gesv<T>((lapack_int)n, (lapack_int)nrhs, wa.data(), ipiv.data(), xp);
    if (info > 0) throw std::runtime_error("solve failed: matrix singular (zero pivot at " + std::to_string(info) + ")");
    if (info < 0) throw std::runtime_error("solve: illegal argument to gesv");
    return true;
}
}  // namespace

// Inputs are dense, plain and of the result dtype (LinearAlgebra::solve writes them out).
void CpuSolver::solve(const MatrixBase& a, const MatrixBase& b, MatrixBase& x) {
    if (a.rows() != a.cols()) throw std::invalid_argument("solve: A must be square");
    if (a.rows() != b.rows()) throw std::invalid_argument("solve: A and B must have same row count");
    if (solve_dense<double>(a, b, x) || solve_dense<float>(a, b, x) ||
        solve_dense<std::complex<double>>(a, b, x) || solve_dense<std::complex<float>>(a, b, x)) {
        return;
    }
    throw std::runtime_error("solve: operands must be dense float32/float64/complex64/complex128");
}

} // namespace pycauset

