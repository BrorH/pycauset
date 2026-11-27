// CpuSolver: vector ops, norms, compute_k, svd and solve. Matmul, element-wise and the
// iterative eigen solvers live in CpuMatmul.cpp, CpuElementwise.cpp and CpuIterative.cpp.
#include <Spectra/SymEigsSolver.h>
#include <Spectra/GenEigsSolver.h>
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
    throw std::invalid_argument("CpuSolver::matrix_vector_multiply: kernel contract is a bit closure operand");
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
    if (dynamic_cast<const BitClosureMatrix*>(&m) != nullptr) {
        debug_trace::set_last("cpu.norm.closure");
        return std::sqrt(block_sum(rows, cols, [&](uint64_t i, uint64_t j) {
            return std::norm(m.get_element_as_complex(i, j));
        }));
    }
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
        if constexpr (std::is_same_v<M, DenseMatrix<double>> || std::is_same_v<M, DenseMatrix<float>>) {
            if (!typed.is_transposed() && !typed.is_subview() && cols == typed.base_cols()) {
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

// Reduced SVD A = U diag(S) VT of dense row-major storage (LAPACK ?gesdd, divide and conquer:
// what NumPy uses). S is real for complex input.
template <typename T>
lapack_int gesdd(char job, lapack_int m, lapack_int n, T* a, real_of_t<T>* s, T* u, lapack_int ldu, T* vt) {
    if constexpr (std::is_same_v<T, double>) return LAPACKE_dgesdd(LAPACK_ROW_MAJOR, job, m, n, a, n, s, u, ldu, vt, n);
    else if constexpr (std::is_same_v<T, float>) return LAPACKE_sgesdd(LAPACK_ROW_MAJOR, job, m, n, a, n, s, u, ldu, vt, n);
    else if constexpr (std::is_same_v<T, std::complex<double>>) return LAPACKE_zgesdd(LAPACK_ROW_MAJOR, job, m, n, a, n, s, u, ldu, vt, n);
    else return LAPACKE_cgesdd(LAPACK_ROW_MAJOR, job, m, n, a, n, s, u, ldu, vt, n);
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
    const char job = (U.cols() == m && m != std::min(m, n)) || (VT.rows() == n && n != std::min(m, n)) ? 'A' : 'S';
    const lapack_int info = gesdd<T>(job, (lapack_int)m, (lapack_int)n, a.data(), s->data(), u->data(),
                                     (lapack_int)U.cols(), vt->data());
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
