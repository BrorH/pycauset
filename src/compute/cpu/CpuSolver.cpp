// CpuSolver: vector ops, norms, compute_k, svd and solve. Matmul, element-wise and the
// iterative eigen solvers live in CpuMatmul.cpp, CpuElementwise.cpp and CpuIterative.cpp.
#include <Spectra/SymEigsSolver.h>
#include <Spectra/GenEigsSolver.h>
#include "pycauset/core/Errors.hpp"
#include "pycauset/core/ScalarUtils.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/math/RawRows.hpp"
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
#include <optional>
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

double CpuSolver::dot(const VectorBase& a, const VectorBase& b) {
    uint64_t n = a.size();
    if (b.size() != n) throw std::invalid_argument("Vector dimensions mismatch");

    const auto dt_a = a.get_data_type();
    const auto dt_b = b.get_data_type();
    if (is_complex_dtype(dt_a) ||
        is_complex_dtype(dt_b)) {
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
        
        // The bits past n in the last word are not elements (bitwise_not sets them): mask them,
        // as the matmul kernels and sum do (review 3.4 C-1: dot(~u, ~u) was 64 for n = 5).
        const uint64_t tail = n % 64;
        const uint64_t last_mask = tail == 0 ? ~uint64_t{0} : (uint64_t{1} << tail) - 1;
        int64_t count = 0;
        #pragma omp parallel for reduction(+:count)
        for (int64_t i = 0; i < (int64_t)num_words; ++i) {
            const uint64_t w = a_data[i] & b_data[i];
            count += std::popcount(i + 1 == (int64_t)num_words ? (w & last_mask) : w);
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

    // Semantics: sum_i a[i] * b[i], no conjugation (np.dot), with BLAS's plain arithmetic.
    // Plain complex64 / complex128 pairs: BLAS ?dotu on the storage, as NumPy (also NumPy's
    // precision: complex64 accumulates in single).
    // A conjugated operand is BLAS ?dotc (which conjugates its first argument): conj(a).b =
    // dotc(a, b), a.conj(b) = dotc(b, a), conj(a).conj(b) = conj(dotu(a, b)). Exactly what NumPy
    // computes on the conjugated arrays. Scaled views take the typed path below.
    auto unscaled = [](const VectorBase& v) { return v.get_scalar() == std::complex<double>(1.0, 0.0); };
    if (unscaled(a) && unscaled(b)) {
        const bool ca = a.is_conjugated(), cb = b.is_conjugated();
        auto run = [&](auto* x, auto* y, auto dotu, auto dotc) -> std::optional<std::complex<double>> {
            if (!x || !y) return std::nullopt;
            using C = std::remove_cv_t<std::remove_pointer_t<decltype(x->data())>>;
            C r;
            const int len = static_cast<int>(n);
            if (!ca && !cb) dotu(len, x->data(), 1, y->data(), 1, &r);
            else if (ca && !cb) dotc(len, x->data(), 1, y->data(), 1, &r);
            else if (!ca && cb) dotc(len, y->data(), 1, x->data(), 1, &r);
            else { dotu(len, x->data(), 1, y->data(), 1, &r); r = std::conj(r); }
            return std::complex<double>(r.real(), r.imag());
        };
        using Z = DenseVector<std::complex<double>>;
        using Cf = DenseVector<std::complex<float>>;
        if (auto r = run(dynamic_cast<const Z*>(&a), dynamic_cast<const Z*>(&b),
                         [](auto... p) { cblas_zdotu_sub(p...); }, [](auto... p) { cblas_zdotc_sub(p...); })) {
            debug_trace::set_last("cpu.dot.zdot");
            return *r;
        }
        if (auto r = run(dynamic_cast<const Cf*>(&a), dynamic_cast<const Cf*>(&b),
                         [](auto... p) { cblas_cdotu_sub(p...); }, [](auto... p) { cblas_cdotc_sub(p...); })) {
            debug_trace::set_last("cpu.dot.cdot");
            return *r;
        }
    }
    // Any other pair: typed reads (every view rule), multiply-adds written out (std::complex's
    // operator* calls the C99 inf / NaN recovery routine per element: 50x slower than BLAS).
    {
        std::vector<std::complex<double>> x(n), y(n);
        if (read_rows(*as_row_matrix(a), 0, 1, x.data()) && read_rows(*as_row_matrix(b), 0, 1, y.data())) {
            debug_trace::set_last("cpu.dot.typed_complex");
            double re = 0.0, im = 0.0;
            for (uint64_t i = 0; i < n; ++i) {
                re += x[i].real() * y[i].real() - x[i].imag() * y[i].imag();
                im += x[i].real() * y[i].imag() + x[i].imag() * y[i].real();
            }
            return {re, im};
        }
    }
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
    auto row = as_row_matrix(v);  // the 1 x n bridge: one norm implementation
    return frobenius_norm(*row);
}

// Kernel contract: a lazy closure C +- C^T over causal bits (LAZY.md L2b), a real factor
// below 2^30, a real vector, a DenseVector<double> result. Two triangular passes over the
// bits, nothing materialized. Every other matrix-vector product is matmul (the matvec entry,
// math/LinearAlgebra.cpp).
void CpuSolver::matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, VectorBase& result) {
    const uint64_t rows = m.rows();
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
    // A structured (non-dense) matrix with a float64 / complex128 result: a block of rows at a
    // time through the typed row reader (every view rule), each block times x with BLAS gemv.
    // The general path wrote the whole n x n matrix out first (2-4 ms at n = 1024).
    auto blocked = [&](auto zero) -> bool {
        using W = decltype(zero);
        auto* out = dynamic_cast<DenseVector<W>*>(&result);
        if (!out) return false;
        const uint64_t cols = m.cols();
        std::vector<W> x(cols);
        if (!read_rows(*as_row_matrix(v), 0, 1, x.data())) return false;
        constexpr uint64_t kRows = 64;
        std::vector<W> probe(std::min<uint64_t>(rows, 1) * cols);
        if (rows > 0 && !read_rows(m, 0, 1, probe.data())) return false;  // readable at all?
        W* y = out->data();
        const uint64_t blocks = (rows + kRows - 1) / kRows;
        std::atomic<bool> ok{true};
        ParallelFor(0, blocks, [&](size_t blk) {
            const uint64_t i0 = blk * kRows, i1 = std::min(rows, i0 + kRows);
            std::vector<W> buf((i1 - i0) * cols);
            if (!read_rows(m, i0, i1, buf.data())) { ok = false; return; }
            const int r = static_cast<int>(i1 - i0), c = static_cast<int>(cols);
            if constexpr (std::is_same_v<W, double>) {
                cblas_dgemv(CblasRowMajor, CblasNoTrans, r, c, 1.0, buf.data(), c, x.data(), 1, 0.0, y + i0, 1);
            } else {
                const W one(1.0), zero_c(0.0);
                cblas_zgemv(CblasRowMajor, CblasNoTrans, r, c, &one, buf.data(), c, x.data(), 1, &zero_c, y + i0, 1);
            }
        });
        if (!ok) throw std::logic_error("matrix_vector_multiply: typed rows readable for the probe, not a block");
        out->set_scalar(1.0);
        debug_trace::set_last(std::is_same_v<W, double> ? "cpu.matvec.rows_f64" : "cpu.matvec.rows_c128");
        return true;
    };
    // Packed storage directly (float64 / complex128 elements, no scale, no conjugation, not a
    // window): a gather over each stored row (the upper triangle) and, for the mirrored
    // classes, a scatter of the same rows (the lower one, A_ji = mirror(A_ij)). Reads the n^2/2
    // stored elements once each; a transposed triangle is the scatter alone.
    auto packed = [&](auto zero) -> bool {
        using W = decltype(zero);
        auto* out = dynamic_cast<DenseVector<W>*>(&result);
        // The view: element = conj(raw * s) when conjugated, else raw * s (ViewValue.hpp), so
        // y = s * (A_raw x), or conj(s) * conj(A_raw conj(x)). Transposed mirrored classes are
        // often such views (A^T = -A, A^H = A), so they take this path too.
        const std::complex<double> sv = m.get_scalar();
        const bool cj = m.is_conjugated() && is_std_complex_v<W>;
        if (!out || m.is_subview() || (!is_std_complex_v<W> && sv.imag() != 0.0)) return false;
        const Structure st = m.get_structure();
        const uint64_t n = rows;
        std::vector<W> x(n);
        if (!read_rows(*as_row_matrix(v), 0, 1, x.data())) return false;
        if constexpr (is_std_complex_v<W>) if (cj) for (auto& e : x) e = std::conj(e);
        W* y = out->data();
        auto finish = [&](W raw) -> W {  // the view applied to (A_raw x)_i
            if constexpr (is_std_complex_v<W>) return cj ? std::conj(sv) * std::conj(raw) : W(sv) * raw;
            else return sv.real() * raw;
        };
        if (st == Structure::DIAGONAL) {
            auto* d = dynamic_cast<const DiagonalMatrix<W>*>(&m);
            if (!d) return false;
            for (uint64_t i = 0; i < n; ++i) y[i] = finish(d->get(i, i) * x[i]);
            out->set_scalar(1.0);
            debug_trace::set_last("cpu.matvec.diagonal");
            return true;
        }
        const TriangularMatrixBase* tri = nullptr;
        bool diag = true;
        if (auto* t = dynamic_cast<const TriangularMatrix<W>*>(&m)) { tri = t; diag = t->has_diagonal(); }
        else if (auto* sm = dynamic_cast<const SymmetricMatrix<W>*>(&m)) tri = sm;
        if (!tri) return false;
        const bool transposed = m.is_transposed();
        // Which parts: gather (A's upper rows times x), scatter (rows scattered into y), and the
        // mirror applied to the scattered values.
        bool gather = true, scatter = false, scatter_diag = false;
        int sign = 1;
        bool conj_scatter = false;
        if (st == Structure::TRIANGULAR) {
            if (transposed) { gather = false; scatter = true; scatter_diag = true; }
        } else if (st == Structure::SYMMETRIC) {
            scatter = true;
        } else if (st == Structure::ANTISYMMETRIC) {
            scatter = true; sign = -1;
        } else if constexpr (is_std_complex_v<W>) {
            if (transposed) return false;  // A^T is conj(A): the typed path
            if (st == Structure::HERMITIAN) { scatter = true; conj_scatter = true; }
            else if (st == Structure::SKEW_HERMITIAN) { scatter = true; conj_scatter = true; sign = -1; }
            else return false;
        } else {
            return false;
        }
        const char* base = static_cast<const char*>(m.shared_mapper()->get_data());
        auto row = [&](uint64_t i) { return reinterpret_cast<const W*>(base + tri->get_row_offset(i)); };
        auto first = [&](uint64_t i) { return diag ? i : i + 1; };  // first stored column of row i
        // One pass over the stored rows: each chunk takes rows c, c + chunks, ... (rows shrink
        // with i, so interleaving balances the work), gathers row . x into acc[i] and scatters
        // the row into its own buffer; the buffers are summed at the end (no shared writes).
        std::vector<W> acc(n, W(0));
        const uint64_t chunks = std::min<uint64_t>(n, 64);
        std::vector<std::vector<W>> part(scatter ? chunks : 0);
        // Real symmetric / antisymmetric (gather and scatter, rows from the diagonal): four rows
        // at a time, each stored element read once for both halves. The reductions vectorize
        // only with reassociation (omp simd); one accumulator per row ran at the add latency
        // and two passes (BLAS dot, then scatter) read the storage twice (0.3x NumPy's gemv).
        bool fused = false;
        if constexpr (!is_std_complex_v<W>) fused = gather && scatter && !scatter_diag && diag;
        if constexpr (!is_std_complex_v<W>) if (fused) {
            // 64 tasks (ParallelFor runs fewer than 32 inline) but only as many n-wide partial
            // buffers as threads: a task claims a free one (each is zeroed once and summed at the
            // end; 64 of them cost more than the products at n = 1024).
            // Below ~2^19 stored elements one thread is faster than waking the pool (one chunk,
            // which ParallelFor runs inline).
            const uint64_t quads = (n + 3) / 4, qchunks = n * n < (uint64_t{1} << 19) ? 1 : std::min<uint64_t>(quads, 64);
            const size_t nbuf = std::max<size_t>(1, ThreadPool::get_num_threads()) + 1;  // + the caller
            part.assign(nbuf, {});
            std::unique_ptr<std::atomic<bool>[]> busy(new std::atomic<bool>[nbuf]);
            for (size_t k = 0; k < nbuf; ++k) busy[k] = false;
            ParallelFor(0, qchunks, [&](size_t c) {
                size_t k = 0;
                for (;; k = (k + 1) % nbuf) {
                    bool expected = false;
                    if (busy[k].compare_exchange_weak(expected, true, std::memory_order_acquire)) break;
                }
                std::vector<W>& d = part[k];
                if (d.empty()) d.assign(n, W(0));
                W* __restrict dq = d.data();
                const W* __restrict xv = x.data();
                for (uint64_t q = c; q < quads; q += qchunks) {
                    const uint64_t i0 = 4 * q, i1 = std::min(n, i0 + 4);
                    // The corner among the group's own columns [i0, i1): row r stores columns
                    // from r; gather includes the diagonal, scatter only j > r.
                    for (uint64_t r = i0; r < i1; ++r) {
                        const W* rr = row(r);
                        for (uint64_t j = r; j < i1; ++j) {
                            acc[r] += rr[j - r] * xv[j];
                            if (j > r) dq[j] += rr[j - r] * xv[r];
                        }
                    }
                    if (i1 >= n) continue;
                    const uint64_t len = n - i1;
                    const W* __restrict xs = xv + i1;
                    W* __restrict ds = dq + i1;
                    if (i1 - i0 == 4) {
                        const W* __restrict p0 = row(i0) + (i1 - i0);
                        const W* __restrict p1 = row(i0 + 1) + (i1 - i0 - 1);
                        const W* __restrict p2 = row(i0 + 2) + (i1 - i0 - 2);
                        const W* __restrict p3 = row(i0 + 3) + (i1 - i0 - 3);
                        const W x0 = xv[i0], x1 = xv[i0 + 1], x2 = xv[i0 + 2], x3 = xv[i0 + 3];
                        W a0 = 0, a1 = 0, a2 = 0, a3 = 0;
#pragma omp simd reduction(+ : a0, a1, a2, a3)
                        for (uint64_t k = 0; k < len; ++k) {
                            const W v0 = p0[k], v1 = p1[k], v2 = p2[k], v3 = p3[k], xk = xs[k];
                            a0 += v0 * xk;
                            a1 += v1 * xk;
                            a2 += v2 * xk;
                            a3 += v3 * xk;
                            ds[k] += v0 * x0 + v1 * x1 + v2 * x2 + v3 * x3;
                        }
                        acc[i0] += a0; acc[i0 + 1] += a1; acc[i0 + 2] += a2; acc[i0 + 3] += a3;
                    } else {
                        for (uint64_t r = i0; r < i1; ++r) {
                            const W* __restrict pr = row(r) + (i1 - r);
                            const W xr = xv[r];
                            W a = 0;
#pragma omp simd reduction(+ : a)
                            for (uint64_t k = 0; k < len; ++k) {
                                a += pr[k] * xs[k];
                                ds[k] += pr[k] * xr;
                            }
                            acc[r] += a;
                        }
                    }
                }
                busy[k].store(false, std::memory_order_release);
            });
        }
        if (!fused) ParallelFor(0, chunks, [&](size_t c) {
            std::vector<W>* d = scatter ? &part[c] : nullptr;
            if (d) d->assign(n, W(0));
            for (uint64_t i = c; i < n; i += chunks) {
                const uint64_t c0 = first(i);
                if (c0 >= n) continue;
                const W* r = row(i);
                if (gather) {
                    const int len = static_cast<int>(n - c0);
                    if constexpr (is_std_complex_v<W>) {
                        W dd;
                        cblas_zdotu_sub(len, r, 1, x.data() + c0, 1, &dd);
                        acc[i] = dd;
                    } else {
                        acc[i] = cblas_ddot(len, r, 1, x.data() + c0, 1);
                    }
                }
                if (d) {
                    const uint64_t lo = scatter_diag ? c0 : std::max<uint64_t>(c0, i + 1);
                    if (lo >= n) continue;
                    const W* __restrict rr = r + (lo - c0);
                    W* __restrict dq = d->data() + lo;
                    const W xi = x[i];
                    const uint64_t len = n - lo;
                    if constexpr (is_std_complex_v<W>) {
                        const double xr = xi.real(), xm = xi.imag();
                        const double cs = conj_scatter ? -1.0 : 1.0;  // conj(a) flips a's imaginary part
                        for (uint64_t q = 0; q < len; ++q) {  // written out: std::complex's * is slow
                            const double ar = rr[q].real(), ai = cs * rr[q].imag();
                            dq[q] = W(dq[q].real() + ar * xr - ai * xm, dq[q].imag() + ar * xm + ai * xr);
                        }
                    } else {
                        for (uint64_t q = 0; q < len; ++q) dq[q] += rr[q] * xi;
                    }
                }
            }
        });
        if (scatter) {
            for (const auto& d : part)  // a buffer no task claimed is empty
                for (uint64_t j = 0; j < d.size(); ++j) acc[j] += static_cast<double>(sign) * d[j];
        }
        const double t_sign = (st == Structure::ANTISYMMETRIC && transposed) ? -1.0 : 1.0;  // A^T = -A
        for (uint64_t i = 0; i < n; ++i) y[i] = finish(t_sign * acc[i]);
        out->set_scalar(1.0);
        debug_trace::set_last(is_std_complex_v<W> ? "cpu.matvec.packed_c128" : "cpu.matvec.packed_f64");
        return true;
    };
    if (m.get_structure() != Structure::DENSE) {
        if (packed(0.0) || packed(std::complex<double>(0.0, 0.0))) return;
        if (blocked(0.0) || blocked(std::complex<double>(0.0, 0.0))) return;
    }
    throw std::invalid_argument("CpuSolver::matrix_vector_multiply: kernel contract is a bit closure or a structured matrix");
}

std::unique_ptr<TriangularMatrix<double>> CpuSolver::compute_k_matrix(
    const TriangularMatrix<bool>& C,
    double a,
    const std::string& output_path,
    int num_threads
) {
    (void)num_threads; // ParallelFor uses global ThreadPool settings
    if (C.has_diagonal()) {
        throw std::invalid_argument("compute_k_matrix expects a causal (strictly upper triangular) matrix");
    }

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

namespace {
// Sum of f(i, j) over a rows x cols index set in fixed blocks of the flat range, partials
// combined in block order: parallel for any shape (a 1 x n vector too) and deterministic.
template <typename F>
double block_sum(uint64_t rows, uint64_t cols, F&& f) {
    const uint64_t total = rows * cols;
    if (total == 0) return 0.0;
    constexpr uint64_t kBlock = 1u << 14;
    const uint64_t blocks = (total + kBlock - 1) / kBlock;
    std::vector<double> part(blocks, 0.0);
    ParallelFor(0, blocks, [&](size_t blk) {
        uint64_t idx = static_cast<uint64_t>(blk) * kBlock;
        const uint64_t end = std::min(total, idx + kBlock);
        uint64_t i = idx / cols, j = idx % cols;
        double acc = 0.0;
        for (; idx < end; ++idx) {
            acc += f(i, j);
            if (++j == cols) { j = 0; ++i; }
        }
        part[blk] = acc;
    });
    double sum = 0.0;
    for (double p : part) sum += p;
    return sum;
}
}  // namespace

// Frobenius norm of any matrix (L2 for vectors through the 1 x n bridge). The view rule
// (math/ViewRead.hpp) makes it a property of the raw values: transposition and conjugation
// do not change it and a factor f scales it by |f|, except integer storage with an integer
// factor, where NumPy wraps each element first. One class dispatch, typed reads; plain
// contiguous float32/float64 storage sums squares with BLAS dot in fixed blocks.
double CpuSolver::frobenius_norm(const MatrixBase& m) {
    const uint64_t rows = m.rows(), cols = m.cols();
    if (rows == 0 || cols == 0) return 0.0;
    const std::complex<double> f = m.get_scalar();
    // (Lazy closures go through visit_matrix too: the bulk path reads their bits.)
    return visit_matrix(m, [&](const auto& typed) -> double {
        using M = std::decay_t<decltype(typed)>;
        using T = std::decay_t<decltype(typed.get(0, 0))>;
        if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
            if (scalar::is_exact_int_factor(f) && f.real() != 1.0) {
                const int64_t k = static_cast<int64_t>(f.real());
                debug_trace::set_last("cpu.norm.int_view");
                return std::sqrt(block_sum(rows, cols, [&](uint64_t i, uint64_t j) {
                    const double x = static_cast<double>(scalar::wrap_scale(typed.get(i, j), k));
                    return x * x;
                }));
            }
        }
        const double af = std::abs(f);
        // The Frobenius norm is the same for A, A^T and conj(A): any dense matrix that is not
        // a window is its whole storage (a transposed view took the per-element path, 0.03x).
        const bool whole = !typed.is_subview() && rows * cols == typed.base_rows() * typed.base_cols();
        if constexpr (std::is_same_v<M, DenseMatrix<double>> || std::is_same_v<M, DenseMatrix<float>>) {
            if (whole) {
                debug_trace::set_last("cpu.norm.blas");
                const T* p = typed.data();
                const uint64_t total = rows * cols;
                constexpr uint64_t kBlock = 1u << 16;
                const uint64_t blocks = (total + kBlock - 1) / kBlock;
                std::vector<double> part(blocks, 0.0);
                ParallelFor(0, blocks, [&](size_t blk) {
                    const uint64_t lo = static_cast<uint64_t>(blk) * kBlock;
                    const int len = static_cast<int>(std::min(total, lo + kBlock) - lo);
                    if constexpr (std::is_same_v<T, double>) part[blk] = cblas_ddot(len, p + lo, 1, p + lo, 1);
                    else part[blk] = cblas_dsdot(len, p + lo, 1, p + lo, 1);
                });
                double sum = 0.0;
                for (double q : part) sum += q;
                return af * std::sqrt(sum);
            }
        }
        if constexpr (std::is_same_v<M, DenseMatrix<std::complex<double>>> || std::is_same_v<M, DenseMatrix<std::complex<float>>>) {
            if (whole) {  // sum |z|^2 = Re(z^H z): BLAS ?dotc per block
                debug_trace::set_last("cpu.norm.blas_complex");
                const T* p = typed.data();
                const uint64_t total = rows * cols;
                constexpr uint64_t kBlock = 1u << 16;
                const uint64_t blocks = (total + kBlock - 1) / kBlock;
                std::vector<double> part(blocks, 0.0);
                ParallelFor(0, blocks, [&](size_t blk) {
                    const uint64_t lo = static_cast<uint64_t>(blk) * kBlock;
                    const int len = static_cast<int>(std::min(total, lo + kBlock) - lo);
                    T r;
                    if constexpr (std::is_same_v<T, std::complex<double>>) cblas_zdotc_sub(len, p + lo, 1, p + lo, 1, &r);
                    else cblas_cdotc_sub(len, p + lo, 1, p + lo, 1, &r);
                    part[blk] = static_cast<double>(r.real());
                });
                double sum = 0.0;
                for (double q : part) sum += q;
                return af * std::sqrt(sum);
            }
        }
        if constexpr (std::is_same_v<M, DenseMatrix<T>> && !std::is_same_v<T, bool> &&
                      !std::is_same_v<T, pycauset::float16_t>) {
            if (whole) {  // other dense dtypes: one contiguous pass over the storage
                debug_trace::set_last("cpu.norm.dense");
                const T* p = typed.data();
                const uint64_t total = rows * cols;
                constexpr uint64_t kBlock = 1u << 16;
                const uint64_t blocks = (total + kBlock - 1) / kBlock;
                std::vector<double> part(blocks, 0.0);
                ParallelFor(0, blocks, [&](size_t blk) {
                    const uint64_t lo = static_cast<uint64_t>(blk) * kBlock, hi = std::min(total, lo + kBlock);
                    const T* __restrict q = p;
                    double acc = 0.0;
                    for (uint64_t k = lo; k < hi; ++k) {
                        if constexpr (is_std_complex_v<T>) {
                            const double re = static_cast<double>(q[k].real()), im = static_cast<double>(q[k].imag());
                            acc += re * re + im * im;
                        } else {
                            const double x = static_cast<double>(q[k]);
                            acc += x * x;
                        }
                    }
                    part[blk] = acc;
                });
                double sum = 0.0;
                for (double q : part) sum += q;
                return af * std::sqrt(sum);
            }
        }
        // Structured, transposed-window and bit storage: stored values in bulk, a tile of rows
        // at a time (math/RawRows.hpp). Squares do not see the mirror's sign or conjugation.
        if constexpr (has_raw_rows_v<T, M> && !std::is_same_v<T, pycauset::float16_t> &&
                      !std::is_same_v<T, pycauset::complex_half>) {
            const uint64_t tile = std::clamp<uint64_t>(rows / 32, 1, 32);
            std::vector<double> part((rows + tile - 1) / tile, 0.0);
            std::atomic<bool> ok{true};
            ParallelFor(0, part.size(), [&](size_t t) {
                const uint64_t r0 = t * tile, r1 = std::min(rows, r0 + tile);
                std::unique_ptr<T[]> raw(new T[(r1 - r0) * cols]);
                if (!raw_rows<T>(typed, r0, r1, raw.get())) { ok = false; return; }
                const T* __restrict q = raw.get();
                const uint64_t len = (r1 - r0) * cols;
                double a0 = 0, a1 = 0;
                for (uint64_t k = 0; k < len; ++k) {
                    if constexpr (is_std_complex_v<T>) {
                        const double re = static_cast<double>(q[k].real()), im = static_cast<double>(q[k].imag());
                        a0 += re * re;
                        a1 += im * im;
                    } else {
                        const double x = static_cast<double>(q[k]);
                        a0 += x * x;
                    }
                }
                part[t] = a0 + a1;
            });
            if (ok) {
                debug_trace::set_last("cpu.norm.bulk");
                double sum = 0.0;
                for (double q : part) sum += q;
                return af * std::sqrt(sum);
            }
        }
        debug_trace::set_last("cpu.norm.typed");
        return af * std::sqrt(block_sum(rows, cols, [&](uint64_t i, uint64_t j) {
            return std::norm(scalar::to_complex_double(typed.get(i, j)));
        }));
    });
}






namespace {
template <typename T> struct real_of { using type = T; };
template <typename T> struct real_of<std::complex<T>> { using type = T; };
template <typename T> using real_of_t = typename real_of<T>::type;

// SVD of dense row-major storage with LAPACK ?gesdd (divide and conquer, NumPy's routine),
// without LAPACKE's row-major transposes (F-108): a row-major m x n buffer is the
// column-major n x m matrix A^T. Factoring A^T = U' S V'^H gives A = conj(V') S U'^T, and the
// column-major outputs read row-major are exactly that: VT' (= V'^H) is U_A and U' is Vh_A.
// So the column-major call writes A's U into `u` and A's Vh into `vt`, no copies.
template <typename T>
lapack_int gesdd_transposed(char job, lapack_int m, lapack_int n, T* a, real_of_t<T>* s, T* u, lapack_int ldu, T* vt) {
    // Column-major problem: n x m matrix (lda n); its U' (n x .) goes to vt (ld n), its
    // VT' (. x m) to u (ld = A's U columns).
    if constexpr (std::is_same_v<T, double>) return LAPACKE_dgesdd(LAPACK_COL_MAJOR, job, n, m, a, std::max<lapack_int>(1, n), s, vt, std::max<lapack_int>(1, n), u, std::max<lapack_int>(1, ldu));
    else if constexpr (std::is_same_v<T, float>) return LAPACKE_sgesdd(LAPACK_COL_MAJOR, job, n, m, a, std::max<lapack_int>(1, n), s, vt, std::max<lapack_int>(1, n), u, std::max<lapack_int>(1, ldu));
    else if constexpr (std::is_same_v<T, std::complex<double>>) return LAPACKE_zgesdd(LAPACK_COL_MAJOR, job, n, m, a, std::max<lapack_int>(1, n), s, vt, std::max<lapack_int>(1, n), u, std::max<lapack_int>(1, ldu));
    else return LAPACKE_cgesdd(LAPACK_COL_MAJOR, job, n, m, a, std::max<lapack_int>(1, n), s, vt, std::max<lapack_int>(1, n), u, std::max<lapack_int>(1, ldu));
}

template <typename T>
bool svd_dense(const MatrixBase& in, MatrixBase& U, VectorBase& S, MatrixBase& VT) {
    using R = real_of_t<T>;
    auto* A = dynamic_cast<const DenseMatrix<T>*>(&in);
    auto* u = dynamic_cast<DenseMatrix<T>*>(&U);
    auto* vt = dynamic_cast<DenseMatrix<T>*>(&VT);
    auto* s = dynamic_cast<DenseVector<R>*>(&S);
    if (!A || !u || !vt || !s) return false;
    debug_trace::set_last("cpu.svd.gesdd");
    const uint64_t m = in.rows(), n = in.cols();
    std::vector<T> a(A->data(), A->data() + m * n);  // gesdd destroys its input
    // U's shape says which: (m, m) is NumPy's full_matrices=True (job 'A'), (m, k) reduced ('S').
    // U's shape says which: (m, m) is NumPy's full_matrices=True (job 'A'), (m, k) reduced
    // ('S'), (m, 0) singular values only ('N').
    const char job = U.cols() == 0 ? 'N'
                   : (U.cols() == m && m != std::min(m, n)) || (VT.rows() == n && n != std::min(m, n)) ? 'A' : 'S';
    const lapack_int info = gesdd_transposed<T>(job, (lapack_int)m, (lapack_int)n, a.data(), s->data(), u->data(),
                                                std::max<lapack_int>(1, (lapack_int)U.cols()), vt->data());
    if (info > 0) throw std::runtime_error("SVD did not converge");
    if (info < 0) throw std::runtime_error("SVD: illegal argument to gesdd");
    return true;
}
}  // namespace

// Input is dense, plain, float32/float64/complex64/complex128 (LinearAlgebra::svd writes it out).
void CpuSolver::svd(const MatrixBase& in, MatrixBase& U, VectorBase& S, MatrixBase& VT) {
    if (svd_dense<double>(in, U, S, VT) || svd_dense<float>(in, U, S, VT) ||
        svd_dense<std::complex<double>>(in, U, S, VT) || svd_dense<std::complex<float>>(in, U, S, VT)) {
        return;
    }
    throw std::runtime_error("svd: operand must be dense float32/float64/complex64/complex128");
}

namespace {
// A X = B for dense row-major storage (F-108), the algorithm of LAPACK ?gesv (LU with partial
// pivoting, NumPy's solve) without LAPACKE's row-major transposes of A, B and X. The row-major
// A buffer is the column-major A^T: ?getrf factors it in place, A^T = P L U. Then
// A X = B  <=>  X^T P L U = B^T, and the row-major B buffer is the column-major B^T (nrhs x n):
// two right-sided triangular solves (U, then unit L) leave X^T P, and applying the row
// interchanges in reverse gives X^T, i.e. X row-major. No copy besides A's (getrf
// overwrites it) and B's into X.
template <typename T>
lapack_int getrf_transposed(lapack_int n, T* a, lapack_int* ipiv) {
    if constexpr (std::is_same_v<T, double>) return LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, a, n, ipiv);
    else if constexpr (std::is_same_v<T, float>) return LAPACKE_sgetrf(LAPACK_COL_MAJOR, n, n, a, n, ipiv);
    else if constexpr (std::is_same_v<T, std::complex<double>>) return LAPACKE_zgetrf(LAPACK_COL_MAJOR, n, n, a, n, ipiv);
    else return LAPACKE_cgetrf(LAPACK_COL_MAJOR, n, n, a, n, ipiv);
}

// B := B * inv(op(F)) for the column-major nrhs x n matrix B and the triangle of F (n x n).
template <typename T>
void trsm_right(CBLAS_UPLO uplo, CBLAS_DIAG diag, int nrhs, int n, const T* f, T* b) {
    if constexpr (std::is_same_v<T, double>) cblas_dtrsm(CblasColMajor, CblasRight, uplo, CblasNoTrans, diag, nrhs, n, 1.0, f, n, b, nrhs);
    else if constexpr (std::is_same_v<T, float>) cblas_strsm(CblasColMajor, CblasRight, uplo, CblasNoTrans, diag, nrhs, n, 1.0f, f, n, b, nrhs);
    else {
        const T one(1);
        if constexpr (std::is_same_v<T, std::complex<double>>) cblas_ztrsm(CblasColMajor, CblasRight, uplo, CblasNoTrans, diag, nrhs, n, &one, f, n, b, nrhs);
        else cblas_ctrsm(CblasColMajor, CblasRight, uplo, CblasNoTrans, diag, nrhs, n, &one, f, n, b, nrhs);
    }
}

template <typename T>
bool solve_dense(const MatrixBase& a, const MatrixBase& b, MatrixBase& x) {
    auto* A = dynamic_cast<const DenseMatrix<T>*>(&a);
    auto* B = dynamic_cast<const DenseMatrix<T>*>(&b);
    auto* X = dynamic_cast<DenseMatrix<T>*>(&x);
    if (!A || !B || !X) return false;
    debug_trace::set_last("cpu.solve.getrf_trsm");
    const uint64_t n = a.rows(), nrhs = b.cols();
    std::vector<T> wa(A->data(), A->data() + n * n);  // getrf overwrites A with its LU factors
    std::vector<lapack_int> ipiv(std::max<uint64_t>(n, 1));
    T* xp = X->data();
    std::copy(B->data(), B->data() + n * nrhs, xp);
    if (n == 0) return true;
    const lapack_int info = getrf_transposed<T>((lapack_int)n, wa.data(), ipiv.data());
    if (info > 0) throw pycauset::linalg_error("solve: matrix is singular (zero pivot at " + std::to_string(info) + ")");
    if (info < 0) throw std::runtime_error("solve: illegal argument to getrf");
    if (nrhs == 0) return true;
    trsm_right<T>(CblasUpper, CblasNonUnit, (int)nrhs, (int)n, wa.data(), xp);
    trsm_right<T>(CblasLower, CblasUnit, (int)nrhs, (int)n, wa.data(), xp);
    for (uint64_t j = n; j-- > 0;) {  // X^T = (X^T P) P^T: row swaps of X, last interchange first
        const uint64_t p = static_cast<uint64_t>(ipiv[j] - 1);
        if (p != j) std::swap_ranges(xp + j * nrhs, xp + (j + 1) * nrhs, xp + p * nrhs);
    }
    return true;
}
}  // namespace

namespace {
// min ||A X - B|| with LAPACK ?gelsd (SVD, divide and conquer): NumPy's lstsq, with NumPy's
// cutoff rcond = eps * max(m, n). gelsd is column-major only in effect (its B must have
// max(m, n) rows), so A and B are copied column-major (gelsd destroys both anyway) and X is
// copied back: O(mn) next to the O(mn min(m, n)) solve.
template <typename T>
bool lstsq_dense(const MatrixBase& a, const MatrixBase& b, MatrixBase& x) {
    using R = real_of_t<T>;
    auto* A = dynamic_cast<const DenseMatrix<T>*>(&a);
    auto* B = dynamic_cast<const DenseMatrix<T>*>(&b);
    auto* X = dynamic_cast<DenseMatrix<T>*>(&x);
    if (!A || !B || !X) return false;
    debug_trace::set_last("cpu.lstsq.gelsd");
    const uint64_t m = a.rows(), n = a.cols(), k = b.cols();
    T* xp = X->data();
    std::fill(xp, xp + n * k, T(0));
    if (m == 0 || n == 0 || k == 0) return true;
    const uint64_t ldb = std::max(m, n);
    std::vector<T> ac(m * n), bc(ldb * k, T(0));
    const T* ap = A->data();
    const T* bp = B->data();
    for (uint64_t i = 0; i < m; ++i)
        for (uint64_t j = 0; j < n; ++j) ac[j * m + i] = ap[i * n + j];
    for (uint64_t i = 0; i < m; ++i)
        for (uint64_t c = 0; c < k; ++c) bc[c * ldb + i] = bp[i * k + c];
    std::vector<R> s(std::min(m, n));
    const R rcond = std::numeric_limits<R>::epsilon() * static_cast<R>(std::max(m, n));
    lapack_int rank = 0, info = 0;
    const auto M = static_cast<lapack_int>(m), N = static_cast<lapack_int>(n), K = static_cast<lapack_int>(k);
    const auto LDB = static_cast<lapack_int>(ldb);
    if constexpr (std::is_same_v<T, double>) info = LAPACKE_dgelsd(LAPACK_COL_MAJOR, M, N, K, ac.data(), M, bc.data(), LDB, s.data(), rcond, &rank);
    else if constexpr (std::is_same_v<T, float>) info = LAPACKE_sgelsd(LAPACK_COL_MAJOR, M, N, K, ac.data(), M, bc.data(), LDB, s.data(), rcond, &rank);
    else if constexpr (std::is_same_v<T, std::complex<double>>) info = LAPACKE_zgelsd(LAPACK_COL_MAJOR, M, N, K, ac.data(), M, bc.data(), LDB, s.data(), rcond, &rank);
    else info = LAPACKE_cgelsd(LAPACK_COL_MAJOR, M, N, K, ac.data(), M, bc.data(), LDB, s.data(), rcond, &rank);
    if (info > 0) throw std::runtime_error("lstsq: SVD did not converge");
    if (info < 0) throw std::runtime_error("lstsq: illegal argument to gelsd");
    for (uint64_t i = 0; i < n; ++i)
        for (uint64_t c = 0; c < k; ++c) xp[i * k + c] = bc[c * ldb + i];
    return true;
}
}  // namespace

void CpuSolver::lstsq(const MatrixBase& a, const MatrixBase& b, MatrixBase& x) {
    if (lstsq_dense<double>(a, b, x) || lstsq_dense<float>(a, b, x) ||
        lstsq_dense<std::complex<double>>(a, b, x) || lstsq_dense<std::complex<float>>(a, b, x)) {
        return;
    }
    throw std::runtime_error("lstsq: operands must be dense float32/float64/complex64/complex128");
}

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
