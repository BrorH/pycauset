// LAPACK kernels for eigen decompositions, Cholesky and QR (review PLAN N.3), one template per
// family for float, double, complex<float>, complex<double>.
//
// Kernel contract: the input is a plain DenseMatrix<T> (the entry functions in
// math/LinearAlgebra.cpp normalise every dtype, structure and view with to_dense in NumPy's
// LAPACK dtype); outputs are DenseMatrix / DenseVector of the documented element types.
// Anything else is a contract violation and throws. Replaces per-dtype hand-written kernels
// (float64/float32 only; no complex eigh or Cholesky), behind which the Python layer fell back
// to NumPy for every other input.
#include <complex>
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#if __has_include(<lapacke.h>)
#include <lapacke.h>
#else
#include <openblas/lapacke.h>
#endif

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "pycauset/core/Errors.hpp"
#include "pycauset/compute/cpu/CpuSolver.hpp"
#include "pycauset/core/DebugTrace.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/math/ScalarOps.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/vector/DenseVector.hpp"

namespace pycauset {
namespace {

// NumPy calls LAPACK on NaN input and returns NaN results; LAPACKE checks every input for NaN
// first and rejected it as an "illegal argument" (det, solve, inv, qr, cholesky, svd, eigh,
// ... all raised, and LAPACK printed to stderr; review 3.6 C-2). One process-wide switch,
// set when the engine loads.
[[maybe_unused]] const bool kLapackeNanCheckOff = [] {
    LAPACKE_set_nancheck(0);
    return true;
}();

template <typename T> struct real_of { using type = T; };
template <typename V> struct real_of<std::complex<V>> { using type = V; };
template <typename T> using real_t = typename real_of<T>::type;
template <typename T> constexpr bool is_complex_v = !std::is_same_v<T, real_t<T>>;
template <typename T> constexpr bool is_cplx_v = !std::is_same_v<T, real_t<T>>;

[[noreturn]] void contract(const char* op) {
    throw std::invalid_argument(std::string("CpuSolver::") + op +
                                ": kernel contract is a plain dense float32/float64/complex64/complex128 input");
}

// A plain dense input of element type T copied into a row-major buffer (LAPACK overwrites it).
template <typename T>
bool copy_input(const MatrixBase& in, std::vector<T>& buf) {
    auto* m = dynamic_cast<const DenseMatrix<T>*>(&in);
    if (m == nullptr) return false;
    if (!is_plain(in)) return false;  // the one definition (conj of real storage is plain)
    buf.assign(m->data(), m->data() + in.rows() * in.cols());
    return true;
}

template <typename T>
DenseMatrix<T>& out_matrix(MatrixBase& m, const char* op) {
    auto* d = dynamic_cast<DenseMatrix<T>*>(&m);
    if (d == nullptr) contract(op);
    d->set_scalar(1.0);
    return *d;
}

template <typename T>
DenseVector<T>& out_vector(VectorBase& v, const char* op) {
    auto* d = dynamic_cast<DenseVector<T>*>(&v);
    if (d == nullptr) contract(op);
    d->set_scalar(1.0);
    return *d;
}

// Every LAPACK call here is column-major on the row-major buffer, i.e. on A^T, so LAPACKE does
// not transpose operands in and out (F-108: that cost about half of each factorization).

// --- Hermitian / symmetric eigen decomposition (?syevd / ?heevd) reading A's lower triangle
// (NumPy's UPLO='L'), which is the upper triangle of the column-major A^T. A^T is A (real
// symmetric) or conj(A) (Hermitian): the same eigenvalues, and eigenvectors conj(v).
template <typename T>
lapack_int heevd(char jobz, lapack_int n, T* a, real_t<T>* w) {
    if constexpr (std::is_same_v<T, float>) return LAPACKE_ssyevd(LAPACK_COL_MAJOR, jobz, 'U', n, a, std::max<lapack_int>(1, n), w);
    else if constexpr (std::is_same_v<T, double>) return LAPACKE_dsyevd(LAPACK_COL_MAJOR, jobz, 'U', n, a, std::max<lapack_int>(1, n), w);
    else if constexpr (std::is_same_v<T, std::complex<float>>) return LAPACKE_cheevd(LAPACK_COL_MAJOR, jobz, 'U', n, a, std::max<lapack_int>(1, n), w);
    else return LAPACKE_zheevd(LAPACK_COL_MAJOR, jobz, 'U', n, a, std::max<lapack_int>(1, n), w);
}

// dst (row-major n x n) = src read column-major, conjugated for complex T: the eigenvectors
// of A from those of A^T. Blocked for cache use.
template <typename T>
void transpose_conj_into(const T* src, T* dst, size_t n) {
    constexpr size_t B = 64;
    ParallelFor(0, (n + B - 1) / B, [&](size_t bi) {
        const size_t i0 = bi * B, i1 = std::min(n, i0 + B);
        for (size_t j0 = 0; j0 < n; j0 += B) {
            const size_t j1 = std::min(n, j0 + B);
            for (size_t i = i0; i < i1; ++i)
                for (size_t j = j0; j < j1; ++j) {
                    if constexpr (is_complex_v<T>) dst[i * n + j] = std::conj(src[j * n + i]);
                    else dst[i * n + j] = src[j * n + i];
                }
        }
    });
}

template <typename T>
bool eigh_t(const MatrixBase& in, VectorBase& w, MatrixBase* v) {
    std::vector<T> a;
    if (!copy_input<T>(in, a)) return false;
    const lapack_int n = static_cast<lapack_int>(in.rows());
    auto& wv = out_vector<real_t<T>>(w, "eigh");
    debug_trace::set_last(v ? "cpu.eigh.heevd" : "cpu.eigvalsh.heevd");
    const lapack_int info = heevd<T>(v ? 'V' : 'N', n, a.data(), wv.data());
    if (info > 0) throw std::runtime_error("eigh: eigenvalues did not converge");
    if (info < 0) throw std::runtime_error("eigh: illegal argument to ?heevd");
    if (v) {
        auto& vm = out_matrix<T>(*v, "eigh");
        transpose_conj_into<T>(a.data(), vm.data(), static_cast<size_t>(n));
    }
    return true;
}

// --- General eigen decomposition (?geev), right eigenvectors.
template <typename T>
bool eig_t(const MatrixBase& in, VectorBase& w, MatrixBase* v) {
    using C = std::complex<real_t<T>>;
    std::vector<T> a;
    if (!copy_input<T>(in, a)) return false;
    const lapack_int n = static_cast<lapack_int>(in.rows());
    auto& wv = out_vector<C>(w, "eig");
    std::vector<T> vr(v ? static_cast<size_t>(n) * n : 1);
    const char jobvr = v ? 'V' : 'N';
    const lapack_int ldvr = v ? n : 1;
    lapack_int info = 0;
    require_finite(a.data(), a.size(), "Array must not contain infs or NaNs");  // NumPy's eig message
    debug_trace::set_last(v ? "cpu.eig.geev" : "cpu.eigvals.geev");
    if constexpr (is_cplx_v<T>) {
        if constexpr (std::is_same_v<T, std::complex<float>>)
            info = LAPACKE_cgeev(LAPACK_ROW_MAJOR, 'N', jobvr, n, a.data(), n, wv.data(), nullptr, 1, vr.data(), ldvr);
        else
            info = LAPACKE_zgeev(LAPACK_ROW_MAJOR, 'N', jobvr, n, a.data(), n, wv.data(), nullptr, 1, vr.data(), ldvr);
        if (v) {
            auto& vm = out_matrix<C>(*v, "eig");
            std::copy(vr.begin(), vr.end(), vm.data());
        }
    } else {
        std::vector<T> wr(n), wi(n);
        if constexpr (std::is_same_v<T, float>)
            info = LAPACKE_sgeev(LAPACK_ROW_MAJOR, 'N', jobvr, n, a.data(), n, wr.data(), wi.data(), nullptr, 1, vr.data(), ldvr);
        else
            info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', jobvr, n, a.data(), n, wr.data(), wi.data(), nullptr, 1, vr.data(), ldvr);
        C* wp = wv.data();
        for (lapack_int i = 0; i < n; ++i) wp[i] = C(wr[i], wi[i]);
        if (v) {
            // A complex pair (j, j+1) stores v_j = vr[:, j] + i vr[:, j+1]; v_{j+1} = conj(v_j).
            auto& vm = out_matrix<C>(*v, "eig");
            C* dst = vm.data();
            for (lapack_int j = 0; j < n; ++j) {
                if (wi[j] != 0 && j + 1 < n) {
                    for (lapack_int r = 0; r < n; ++r) {
                        const C z(vr[r * n + j], vr[r * n + j + 1]);
                        dst[r * n + j] = z;
                        dst[r * n + j + 1] = std::conj(z);
                    }
                    ++j;
                } else {
                    for (lapack_int r = 0; r < n; ++r) dst[r * n + j] = C(vr[r * n + j], 0);
                }
            }
        }
    }
    if (info > 0) throw linalg_error("Eigenvalues did not converge");
    if (info < 0) throw std::runtime_error("eig: illegal argument to ?geev");
    return true;
}

// --- Cholesky (?potrf), lower: A = L L^H.
template <typename T>
bool cholesky_t(const MatrixBase& in, MatrixBase& out) {
    std::vector<T> a;
    if (!copy_input<T>(in, a)) return false;
    const lapack_int n = static_cast<lapack_int>(in.rows());
    lapack_int info = 0;
    debug_trace::set_last("cpu.cholesky.potrf");
    // ?potrf('U') on the column-major A^T (its upper triangle is A's lower one): A^T = U^H U,
    // so A = U^T conj(U) and L = U^T, which is the column-major U read row-major. L is in the
    // buffer's lower triangle as it stands, real or complex.
    if constexpr (std::is_same_v<T, float>) info = LAPACKE_spotrf(LAPACK_COL_MAJOR, 'U', n, a.data(), std::max<lapack_int>(1, n));
    else if constexpr (std::is_same_v<T, double>) info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', n, a.data(), std::max<lapack_int>(1, n));
    else if constexpr (std::is_same_v<T, std::complex<float>>) info = LAPACKE_cpotrf(LAPACK_COL_MAJOR, 'U', n, a.data(), std::max<lapack_int>(1, n));
    else info = LAPACKE_zpotrf(LAPACK_COL_MAJOR, 'U', n, a.data(), std::max<lapack_int>(1, n));
    if (info > 0) throw pycauset::linalg_error("cholesky: matrix is not positive definite");
    if (info < 0) throw std::runtime_error("cholesky: illegal argument to ?potrf");
    auto& o = out_matrix<T>(out, "cholesky");
    T* dst = o.data();
    for (lapack_int i = 0; i < n; ++i)
        for (lapack_int j = 0; j < n; ++j) dst[i * n + j] = j <= i ? a[i * n + j] : T(0);
    return true;
}

// --- QR, k = min(m, n) reflectors. Reduced: Q m x k, R k x n. Complete: Q m x m, R m x n (the
// shapes of the outputs say which). The row-major buffer is the column-major A^T, and the LQ
// of A^T (?gelqf + ?orglq / ?unglq) is NumPy's QR of A (?geqrf + ?orgqr) transposed: the
// same reflectors and signs, A^T = L Q with R = L^T and Q_A = Q^T. Read row-major, the
// column-major L is R in the buffer's upper triangle and the column-major Q is Q_A.
template <typename T>
bool qr_t(const MatrixBase& in, MatrixBase& qm, MatrixBase& rm) {
    std::vector<T> a;
    if (!copy_input<T>(in, a)) return false;
    const lapack_int m = static_cast<lapack_int>(in.rows()), n = static_cast<lapack_int>(in.cols());
    const lapack_int k = std::min(m, n);
    const lapack_int qc = static_cast<lapack_int>(qm.cols()), rr = static_cast<lapack_int>(rm.rows());
    if (qm.rows() != in.rows() || (qc != k && qc != m) || rr != qc || rm.cols() != in.cols()) contract("qr");
    std::vector<T> tau(std::max<lapack_int>(k, 1));
    lapack_int info = 0;
    debug_trace::set_last("cpu.qr.gelqf");
    if (k > 0) {
        if constexpr (std::is_same_v<T, float>) info = LAPACKE_sgelqf(LAPACK_COL_MAJOR, n, m, a.data(), n, tau.data());
        else if constexpr (std::is_same_v<T, double>) info = LAPACKE_dgelqf(LAPACK_COL_MAJOR, n, m, a.data(), n, tau.data());
        else if constexpr (std::is_same_v<T, std::complex<float>>) info = LAPACKE_cgelqf(LAPACK_COL_MAJOR, n, m, a.data(), n, tau.data());
        else info = LAPACKE_zgelqf(LAPACK_COL_MAJOR, n, m, a.data(), n, tau.data());
        if (info != 0) throw std::runtime_error("qr: ?gelqf failed");
    }
    auto& r = out_matrix<T>(rm, "qr");
    T* rp = r.data();
    for (lapack_int i = 0; i < rr; ++i)
        for (lapack_int j = 0; j < n; ++j) rp[i * n + j] = (i < k && j >= i) ? a[i * n + j] : T(0);
    // The reflectors, as the column-major qc x m matrix (row-major m x qc) ?orglq expands in place.
    auto& qo = out_matrix<T>(qm, "qr");
    T* q = qo.data();
    std::fill(q, q + static_cast<size_t>(m) * qc, T(0));
    for (lapack_int i = 0; i < m; ++i)
        for (lapack_int j = 0; j < std::min(qc, n); ++j) q[i * qc + j] = a[i * n + j];
    if (qc > 0) {
        if constexpr (std::is_same_v<T, float>) info = LAPACKE_sorglq(LAPACK_COL_MAJOR, qc, m, k, q, qc, tau.data());
        else if constexpr (std::is_same_v<T, double>) info = LAPACKE_dorglq(LAPACK_COL_MAJOR, qc, m, k, q, qc, tau.data());
        else if constexpr (std::is_same_v<T, std::complex<float>>) info = LAPACKE_cunglq(LAPACK_COL_MAJOR, qc, m, k, q, qc, tau.data());
        else info = LAPACKE_zunglq(LAPACK_COL_MAJOR, qc, m, k, q, qc, tau.data());
        if (info != 0) throw std::runtime_error("qr: ?orglq failed");
    }
    return true;
}

// --- LU with partial pivoting (?getrf): A = P L U, P m x m (real), L m x k unit lower, U k x n.
// A singular A still factors (U has a zero pivot), as SciPy's lu.
template <typename T>
bool lu_t(const MatrixBase& in, MatrixBase& pm, MatrixBase& lm, MatrixBase& um) {
    std::vector<T> a;
    if (!copy_input<T>(in, a)) return false;
    const lapack_int m = static_cast<lapack_int>(in.rows()), n = static_cast<lapack_int>(in.cols());
    const lapack_int k = std::min(m, n);
    std::vector<lapack_int> ipiv(std::max<lapack_int>(k, 1));
    lapack_int info = 0;
    debug_trace::set_last("cpu.lu.getrf");
    if (k > 0) {
        if constexpr (std::is_same_v<T, float>) info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, m, n, a.data(), n, ipiv.data());
        else if constexpr (std::is_same_v<T, double>) info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, a.data(), n, ipiv.data());
        else if constexpr (std::is_same_v<T, std::complex<float>>) info = LAPACKE_cgetrf(LAPACK_ROW_MAJOR, m, n, a.data(), n, ipiv.data());
        else info = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, m, n, a.data(), n, ipiv.data());
    }
    if (info < 0) throw std::runtime_error("lu: illegal argument to ?getrf");
    T* u = out_matrix<T>(um, "lu").data();
    for (lapack_int i = 0; i < k; ++i)
        for (lapack_int j = 0; j < n; ++j) u[i * n + j] = j >= i ? a[i * n + j] : T(0);
    T* l = out_matrix<T>(lm, "lu").data();
    for (lapack_int i = 0; i < m; ++i)
        for (lapack_int j = 0; j < k; ++j) l[i * k + j] = j < i ? a[i * n + j] : (j == i ? T(1) : T(0));
    using R = real_t<T>;
    R* p = out_matrix<R>(pm, "lu").data();
    std::fill(p, p + static_cast<size_t>(m) * m, R(0));
    std::vector<lapack_int> perm(m);
    for (lapack_int i = 0; i < m; ++i) perm[i] = i;
    for (lapack_int i = 0; i < k; ++i) std::swap(perm[i], perm[ipiv[i] - 1]);
    for (lapack_int i = 0; i < m; ++i) p[perm[i] * m + i] = R(1);
    return true;
}

// --- LU pivots for det / slogdet (?getrf): the diagonal of U and the permutation's parity.
template <typename T>
bool pivots_t(const MatrixBase& in, LuPivots& out) {
    std::vector<T> a;
    if (!copy_input<T>(in, a)) return false;
    const lapack_int n = static_cast<lapack_int>(in.rows());
    std::vector<lapack_int> ipiv(std::max<lapack_int>(n, 1));
    lapack_int info = 0;
    debug_trace::set_last("cpu.det.getrf");
    if (n > 0) {
        // The LU of A^T (column-major on the buffer): det(A^T) = det(A), no transposes.
        if constexpr (std::is_same_v<T, float>) info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, n, n, a.data(), n, ipiv.data());
        else if constexpr (std::is_same_v<T, double>) info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, a.data(), n, ipiv.data());
        else if constexpr (std::is_same_v<T, std::complex<float>>) info = LAPACKE_cgetrf(LAPACK_COL_MAJOR, n, n, a.data(), n, ipiv.data());
        else info = LAPACKE_zgetrf(LAPACK_COL_MAJOR, n, n, a.data(), n, ipiv.data());
    }
    if (info < 0) throw std::runtime_error("det: illegal argument to ?getrf");  // info > 0: an exact zero pivot
    out.pivots.resize(n);
    out.odd_permutation = false;
    for (lapack_int i = 0; i < n; ++i) {
        out.pivots[i] = std::complex<double>(a[static_cast<size_t>(i) * n + i]);
        if (ipiv[i] != i + 1) out.odd_permutation = !out.odd_permutation;
    }
    return true;
}

}  // namespace

void CpuSolver::eigh(const MatrixBase& in, VectorBase& w, MatrixBase& v, char) {
    if (in.rows() != in.cols()) throw std::invalid_argument("eigh: matrix must be square");
    if (eigh_t<float>(in, w, &v) || eigh_t<double>(in, w, &v) || eigh_t<std::complex<float>>(in, w, &v) ||
        eigh_t<std::complex<double>>(in, w, &v)) return;
    contract("eigh");
}

void CpuSolver::eigvalsh(const MatrixBase& in, VectorBase& w, char) {
    if (in.rows() != in.cols()) throw std::invalid_argument("eigvalsh: matrix must be square");
    if (eigh_t<float>(in, w, nullptr) || eigh_t<double>(in, w, nullptr) || eigh_t<std::complex<float>>(in, w, nullptr) ||
        eigh_t<std::complex<double>>(in, w, nullptr)) return;
    contract("eigvalsh");
}

void CpuSolver::eig(const MatrixBase& in, VectorBase& w, MatrixBase& v) {
    if (in.rows() != in.cols()) throw std::invalid_argument("eig: matrix must be square");
    if (eig_t<float>(in, w, &v) || eig_t<double>(in, w, &v) || eig_t<std::complex<float>>(in, w, &v) ||
        eig_t<std::complex<double>>(in, w, &v)) return;
    contract("eig");
}

void CpuSolver::eigvals(const MatrixBase& in, VectorBase& w) {
    if (in.rows() != in.cols()) throw std::invalid_argument("eigvals: matrix must be square");
    if (eig_t<float>(in, w, nullptr) || eig_t<double>(in, w, nullptr) || eig_t<std::complex<float>>(in, w, nullptr) ||
        eig_t<std::complex<double>>(in, w, nullptr)) return;
    contract("eigvals");
}

void CpuSolver::cholesky(const MatrixBase& in, MatrixBase& out) {
    if (in.rows() != in.cols()) throw std::invalid_argument("cholesky: matrix must be square");
    if (cholesky_t<float>(in, out) || cholesky_t<double>(in, out) || cholesky_t<std::complex<float>>(in, out) ||
        cholesky_t<std::complex<double>>(in, out)) return;
    contract("cholesky");
}

LuPivots CpuSolver::lu_pivots(const MatrixBase& in) {
    if (in.rows() != in.cols()) throw std::invalid_argument("det: matrix must be square");
    LuPivots out;
    if (pivots_t<float>(in, out) || pivots_t<double>(in, out) || pivots_t<std::complex<float>>(in, out) ||
        pivots_t<std::complex<double>>(in, out)) return out;
    contract("lu_pivots");
}

void CpuSolver::lu(const MatrixBase& in, MatrixBase& p, MatrixBase& l, MatrixBase& u) {
    if (lu_t<float>(in, p, l, u) || lu_t<double>(in, p, l, u) || lu_t<std::complex<float>>(in, p, l, u) ||
        lu_t<std::complex<double>>(in, p, l, u)) return;
    contract("lu");
}

void CpuSolver::qr(const MatrixBase& in, MatrixBase& q, MatrixBase& r) {
    if (qr_t<float>(in, q, r) || qr_t<double>(in, q, r) || qr_t<std::complex<float>>(in, q, r) ||
        qr_t<std::complex<double>>(in, q, r)) return;
    contract("qr");
}

}  // namespace pycauset
