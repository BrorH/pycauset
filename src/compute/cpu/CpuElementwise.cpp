// CpuSolver: element-wise add / subtract / multiply / divide (split from CpuSolver.cpp, review 4.2).
#include <Spectra/SymEigsSolver.h>
#include <Spectra/GenEigsSolver.h>
#include "pycauset/core/ScalarUtils.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/math/RawRows.hpp"
#include "pycauset/math/ViewRead.hpp"
#include "pycauset/core/DTypeDispatch.hpp"
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

#include "CpuKernelsCommon.hpp"

namespace pycauset {
using namespace cpu_detail;

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
    inline uint64_t broadcast_index(uint64_t out_index, uint64_t in_dim) {
        return (in_dim == 1) ? 0 : out_index;
    }

    // Logical rows of an operand, as T, through the typed row reader (ViewRead.hpp): int64 for
    // integer results (lossless for every integer type, wrapping views exact), double for real,
    // complex128 for complex results. Rows [i0, i1) of the RESULT, broadcast from the operand
    // (one row or one column). Returns false when the reader cannot read the operand (lazy
    // closures, windows of structured classes).
    template <typename T>
    using wide_of_t = std::conditional_t<is_std_complex_v<T>, std::complex<double>,
                                         std::conditional_t<std::is_integral_v<T>, int64_t, double>>;

    template <typename T>
    bool load_rows(const MatrixBase& m, uint64_t i0, uint64_t i1, uint64_t cols, std::vector<T>& dst) {
        using W = wide_of_t<T>;
        const uint64_t mr = m.rows(), mc = m.cols();
        const uint64_t r0 = mr == 1 ? 0 : i0, r1 = mr == 1 ? 1 : i1;
        // Stored in T already, no view: the bulk reader straight into dst (no wide copy and no
        // per-element view rule). Tiles of rows in parallel; ParallelFor wants >= 32 items.
        if (mr != 1 && mc == cols && m.get_scalar() == std::complex<double>(1.0, 0.0) && !m.is_conjugated()) {
            bool done = false;
            visit_matrix(m, [&](const auto& typed) {
                using M = std::decay_t<decltype(typed)>;
                using In = std::decay_t<decltype(typed.get(0, 0))>;
                if constexpr (std::is_same_v<In, T> && has_raw_rows_v<T, M>) {
                    dst.resize((i1 - i0) * cols);
                    const uint64_t n = r1 - r0, tile = std::clamp<uint64_t>(n / 32, 1, 32);
                    std::atomic<bool> ok{true};
                    ParallelFor(0, (n + tile - 1) / tile, [&](size_t t) {
                        const uint64_t a = r0 + t * tile, b = std::min(r1, a + tile);
                        if (!raw_rows<T>(typed, a, b, dst.data() + (a - r0) * cols)) ok = false;
                    });
                    done = ok;
                }
            });
            if (done) return true;
        }
        if constexpr (std::is_same_v<T, W>) {
            if (mr != 1 && mc == cols) {  // no broadcast: straight into dst, no copy pass
                dst.resize((i1 - i0) * cols);
                return pycauset::read_rows(m, r0, r1, dst.data());
            }
        }
        std::vector<W> w((r1 - r0) * mc);
        if (!pycauset::read_rows(m, r0, r1, w.data())) return false;
        dst.resize((i1 - i0) * cols);
        for (uint64_t i = i0; i < i1; ++i) {
            const W* src = w.data() + (mr == 1 ? 0 : i - i0) * mc;
            T* out = dst.data() + (i - i0) * cols;
            for (uint64_t j = 0; j < cols; ++j) {
                const W v = src[mc == 1 ? 0 : j];
                if constexpr (std::is_same_v<T, bool>) out[j] = v != W(0);
                else if constexpr (std::is_same_v<T, pycauset::float16_t>) out[j] = pycauset::float16_t(v);
                else out[j] = static_cast<T>(v);
            }
        }
        return true;
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

        // Operands through the typed row reader (complex128 rows, a block at a time), rounded
        // once to complex32; the per-element accessor only for operands it cannot read.
        {
            const uint64_t block = std::max<uint64_t>(1, (uint64_t{1} << 18) / std::max<uint64_t>(cols, 1));
            std::vector<std::complex<double>> ba, bb;
            bool typed = true;
            for (uint64_t i0 = 0; i0 < rows && typed; i0 += block) {
                const uint64_t i1 = std::min(rows, i0 + block);
                if (!(typed = load_rows<std::complex<double>>(a, i0, i1, cols, ba) &&
                              load_rows<std::complex<double>>(b, i0, i1, cols, bb))) break;
                ParallelFor(i0, i1, [&](size_t ii) {
                    const uint64_t i = static_cast<uint64_t>(ii);
                    for (uint64_t j = 0; j < cols; ++j) {
                        const std::complex<double> vc = op(ba[(i - i0) * cols + j], bb[(i - i0) * cols + j]);
                        const uint64_t idx = t_c ? (j * storage_cols + i) : (i * storage_cols + j);
                        rdst[idx] = pycauset::float16_t(vc.real());
                        idst[idx] = pycauset::float16_t(vc.imag());
                    }
                });
            }
            if (typed) {
                out.set_scalar(1.0);
                return;
            }
        }

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

}  // namespace

namespace {
    // The storage of a plain, full, unbroadcast DenseMatrix<T> (its rows are the logical rows),
    // else nullptr.
    template <typename T>
    const T* in_place_rows(const MatrixBase& m, uint64_t rows, uint64_t cols) {
        if constexpr (std::is_same_v<T, bool>) {
            return nullptr;  // bit storage
        } else {
            const auto* d = dynamic_cast<const DenseMatrix<T>*>(&m);
            if (!d || d->rows() != rows || d->cols() != cols || d->base_cols() != cols || d->is_transposed() ||
                d->is_subview() || d->is_conjugated() || d->get_scalar() != std::complex<double>(1.0, 0.0))
                return nullptr;
            return d->data();
        }
    }

    // a op b over result rows in blocks: operands through load_rows, then write(i, j, value)
    // for the stored columns [range(i).first, range(i).second) of each result row (only those
    // are computed: a symmetric or triangular result computes half). False if an operand is
    // unreadable.
    template <typename T, typename Op, typename Range, typename Write>
    bool rows_binary(const MatrixBase& a, const MatrixBase& b, uint64_t rows, uint64_t cols, Op op, Range range,
                     Write write) {
        const uint64_t block = std::max<uint64_t>(1, (uint64_t{1} << 18) / std::max<uint64_t>(cols, 1));
        std::vector<T> ba, bb;
        for (uint64_t i0 = 0; i0 < rows; i0 += block) {
            const uint64_t i1 = std::min(rows, i0 + block);
            // A plain full dense operand of type T is read in place (copying it into the
            // row buffer was a pass over memory for nothing).
            const T* pa = in_place_rows<T>(a, rows, cols);
            const T* pb = in_place_rows<T>(b, rows, cols);
            if ((!pa && !load_rows<T>(a, i0, i1, cols, ba)) || (!pb && !load_rows<T>(b, i0, i1, cols, bb))) return false;
            const T* xa = pa ? pa + i0 * cols : ba.data();
            const T* xb = pb ? pb + i0 * cols : bb.data();
            ParallelFor(i0, i1, [&](size_t ii) {
                const uint64_t i = static_cast<uint64_t>(ii);
                const T* x = xa + (i - i0) * cols;
                const T* y = xb + (i - i0) * cols;
                const auto [lo, hi] = range(i);
                for (uint64_t j = lo; j < hi; ++j) write(i, j, static_cast<T>(op(x[j], y[j])));
            });
        }
        return true;
    }

    // Reads exactly its storage: plain, or a transposed diagonal / symmetric matrix (transposing
    // those changes no element).
    bool reads_as_stored(const MatrixBase& m) {
        if (is_plain(m)) return true;
        const Structure s = m.get_structure();
        return (s == Structure::DIAGONAL || s == Structure::SYMMETRIC) && !m.is_subview() &&
               m.get_scalar() == std::complex<double>(1.0, 0.0) && !m.is_conjugated();
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
        
        // Structure rule: both operands and the result share one packed layout (same class and
        // element type, same triangle / symmetry kind, no view): the op runs over the stored
        // elements directly, contiguous and O(stored). Diagonal + diagonal read n^2 logical
        // elements for an O(n) result before (0.1x NumPy for every structured operand).
        if (a_full && b_full && reads_as_stored(a) && reads_as_stored(b) && is_plain(result)) {
            auto packed = [&](auto* res_c) -> bool {
                using C = std::remove_pointer_t<decltype(res_c)>;
                if constexpr (std::is_same_v<T, bool>) {
                    return false;  // bit storage: the word kernels (math/Bitwise.cpp)
                } else {
                const auto* ca = dynamic_cast<const C*>(&a);
                const auto* cb = dynamic_cast<const C*>(&b);
                if (!res_c || !ca || !cb) return false;
                if (a.get_structure() != result.get_structure() || b.get_structure() != result.get_structure())
                    return false;  // e.g. an IdentityMatrix is a DiagonalMatrix without storage
                if constexpr (requires { res_c->has_diagonal(); }) {
                    if (ca->has_diagonal() != res_c->has_diagonal() || cb->has_diagonal() != res_c->has_diagonal())
                        return false;
                }
                const size_t bytes = res_c->shared_mapper()->get_data_size();
                if (ca->shared_mapper()->get_data_size() != bytes || cb->shared_mapper()->get_data_size() != bytes)
                    return false;
                const T* x = ca->data();
                const T* y = cb->data();
                T* z = res_c->data();
                debug_trace::set_last("cpu.elementwise.packed");
                ParallelForRange(0, bytes / sizeof(T), [&](size_t k0, size_t k1) {
                    const T* __restrict xr = x;
                    const T* __restrict yr = y;
                    T* __restrict zr = z;
                    for (size_t k = k0; k < k1; ++k) zr[k] = static_cast<T>(op(xr[k], yr[k]));
                });
                return true;
                }
            };
            if (result.get_structure() == Structure::TRIANGULAR) {
                if (packed(dynamic_cast<TriangularMatrix<T>*>(&result))) return;
            } else if (result.get_structure() == Structure::DIAGONAL) {
                if (packed(dynamic_cast<DiagonalMatrix<T>*>(&result))) return;
            } else if (result.get_structure() == Structure::SYMMETRIC || result.get_structure() == Structure::ANTISYMMETRIC ||
                       result.get_structure() == Structure::HERMITIAN || result.get_structure() == Structure::SKEW_HERMITIAN) {
                if (packed(dynamic_cast<SymmetricMatrix<T>*>(&result))) return;
            }
        }

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
                
                ParallelForRange(0, rows * cols, [&](size_t i0, size_t i1) {
                    const T* __restrict xr = a_data;
                    const T* __restrict yr = b_data;
                    T* __restrict zr = res_data;
                    for (size_t i = i0; i < i1; ++i) zr[i] = op(xr[i], yr[i]);
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

            // Any other operand (bit, float16, structured, mixed dtypes): the typed row reader,
            // writing straight into a plain result (set() checks and dispatches per element).
            if (res_plain) {
                T* res_data = res_dense->data();
                if (rows_binary<T>(a, b, rows, cols, op, [cols](uint64_t) { return std::pair<uint64_t, uint64_t>(0, cols); },
                                   [res_data, cols](uint64_t i, uint64_t j, T v) { res_data[i * cols + j] = v; })) {
                    return;
                }
            } else if (rows_binary<T>(a, b, rows, cols, op, [cols](uint64_t) { return std::pair<uint64_t, uint64_t>(0, cols); },
                                      [&](uint64_t i, uint64_t j, T v) { res_dense->set(i, j, v); })) {
                return;
            }
            // Per-element fallback: lazy closures and windows of structured classes.
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
            // Structured results: the typed row reader over the stored positions; the virtual
            // accessor only for operands the reader cannot read.
            if (rows == cols) {
                // A fresh result (no view): written straight into its packed rows, one pointer per
                // row; set() per element paid ensure_unique and the triangle checks every time.
                const bool fresh = !std::is_same_v<T, bool> &&  // bit storage is packed words
                                   result.get_scalar() == std::complex<double>(1.0, 0.0) && !result.is_transposed() &&
                                   !result.is_subview() && !result.is_conjugated();
                if (auto* res_tri = dynamic_cast<TriangularMatrix<T>*>(&result)) {
                    const bool diag = res_tri->has_diagonal();
                    auto range = [diag, cols](uint64_t i) { return std::pair<uint64_t, uint64_t>(diag ? i : i + 1, cols); };
                    if (fresh) {
                        res_tri->ensure_unique();  // first: it may move the storage
                        auto* bytes = static_cast<uint8_t*>(res_tri->shared_mapper()->get_data());
                        if (rows_binary<T>(a, b, rows, cols, op, range, [&, bytes, diag](uint64_t i, uint64_t j, T v) {
                                reinterpret_cast<T*>(bytes + res_tri->get_row_offset(i))[j - (diag ? i : i + 1)] = v;
                            }))
                            return;
                    } else if (rows_binary<T>(a, b, rows, cols, op, range,
                                              [&](uint64_t i, uint64_t j, T v) { if (v != static_cast<T>(0)) res_tri->set(i, j, v); }))
                        return;
                } else if (auto* res_sym = dynamic_cast<SymmetricMatrix<T>*>(&result)) {
                    auto range = [cols](uint64_t i) { return std::pair<uint64_t, uint64_t>(i, cols); };
                    if (fresh) {
                        res_sym->ensure_unique();  // first: it may move the storage
                        auto* bytes = static_cast<uint8_t*>(res_sym->shared_mapper()->get_data());
                        // The diagonal through set(): it checks what the class requires there
                        // (0 for antisymmetric, real for Hermitian).
                        if (rows_binary<T>(a, b, rows, cols, op, range, [&, bytes](uint64_t i, uint64_t j, T v) {
                                if (j == i) res_sym->set(i, i, v);
                                else reinterpret_cast<T*>(bytes + res_sym->get_row_offset(i))[j - i] = v;
                            }))
                            return;
                    } else if (rows_binary<T>(a, b, rows, cols, op, range,
                                              [&](uint64_t i, uint64_t j, T v) { res_sym->set(i, j, v); }))
                        return;
                } else if (result.get_structure() == Structure::DIAGONAL) {
                    auto* res_diag = dynamic_cast<DiagonalMatrix<T>*>(&result);
                    if (res_diag && rows_binary<T>(a, b, rows, cols, op,
                                                   [](uint64_t i) { return std::pair<uint64_t, uint64_t>(i, i + 1); },
                                                   [&](uint64_t i, uint64_t, T v) { res_diag->set(i, i, v); }))
                        return;
                }
            }
            auto read_a = [&](uint64_t i, uint64_t j) -> T { return static_cast<T>(a.get_element_as_double(i, j)); };
            auto read_b = [&](uint64_t i, uint64_t j) -> T { return static_cast<T>(b.get_element_as_double(i, j)); };
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

    // bool * bool never reaches the device: the entry function's structure rule computes it
    // from the packed words (math/Bitwise.cpp, `bitwise(a, b, And)`).

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
            binary_op_complex16_impl(a, b, *out, [](auto x, auto y) { return pycauset::scalar::true_divide(x, y); });
            return;
        }
        throw std::runtime_error("CpuSolver::elementwise_divide complex_float16 result type mismatch");
    }
    if (dtype == DataType::COMPLEX_FLOAT32) {
        debug_trace::set_last("cpu.div.c32");
        binary_op_impl<std::complex<float>>(a, b, result, [](auto x, auto y) { return pycauset::scalar::true_divide(x, y); });
        return;
    }
    if (dtype == DataType::COMPLEX_FLOAT64) {
        debug_trace::set_last("cpu.div.c64");
        binary_op_impl<std::complex<double>>(a, b, result, [](auto x, auto y) { return pycauset::scalar::true_divide(x, y); });
        return;
    }

    if (dtype == DataType::BIT) {
        throw std::runtime_error("CpuSolver::elementwise_divide bit result type not supported");
    }

    if (dtype == DataType::FLOAT64) {
        debug_trace::set_last("cpu.div.f64");
        if (try_fast_simd<double>(a, b, result, get_div_kernel<double>())) return;
        binary_op_impl<double>(a, b, result, [](auto x, auto y) { return pycauset::scalar::true_divide(x, y); });
    } else if (dtype == DataType::FLOAT16) {
        debug_trace::set_last("cpu.div.f16");
        binary_op_impl<pycauset::float16_t>(a, b, result, [](auto x, auto y) { return pycauset::scalar::true_divide(x, y); });
    } else if (dtype == DataType::FLOAT32) {
        debug_trace::set_last("cpu.div.f32");
        if (try_fast_simd<float>(a, b, result, get_div_kernel<float>())) return;
        binary_op_impl<float>(a, b, result, [](auto x, auto y) { return pycauset::scalar::true_divide(x, y); });
    } else {
        // Integer and bool division gives a float result (DType System §2.5): an integer result
        // here is a caller bug (the integer branches truncated, and trapped on 0, review 3.4 C-17).
        throw std::logic_error("CpuSolver::elementwise_divide: result dtype must be float or complex");
    }
}

} // namespace pycauset
