// CpuSolver: iterative eigen solvers (Spectra) and batched gemv (split from CpuSolver.cpp, review 4.2).
#include <Spectra/SymEigsSolver.h>
#include <Spectra/GenEigsSolver.h>
#include "pycauset/core/ScalarUtils.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/math/ViewRead.hpp"
#include "pycauset/compute/cpu/CpuSolver.hpp"
#include <algorithm>
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
std::vector<double> to_memory_flat_real_square(const MatrixBase& m) {
    const uint64_t rows = m.rows();
    const uint64_t cols = m.cols();
    if (rows != cols) {
        throw std::runtime_error("Operation requires square matrix");
    }

    const uint64_t n = rows;
    std::vector<double> mat(n * n);
    // The typed row reader (no per-element virtual call); the accessor only where it can't read.
    if (pycauset::read_rows(m, 0, n, mat.data())) return mat;
    ParallelFor(0, n, [&](size_t i) {
        for (size_t j = 0; j < n; ++j) {
            mat[i * n + j] = m.get_element_as_double(static_cast<uint64_t>(i), static_cast<uint64_t>(j));
        }
    });
    return mat;
}

} // namespace

namespace {

// y = A x through the matvec entry (math/LinearAlgebra.cpp), so every structure, dtype and
// storage (RAM or disk-backed) works here. Spectra drives it.
class MatVecOperator {
public:
    using Scalar = double;
    MatVecOperator(CpuSolver& solver, const MatrixBase& a)
        : solver_(solver), a_(a), n_(a.rows()), x_(n_, "") {}
    Eigen::Index rows() const { return static_cast<Eigen::Index>(n_); }
    Eigen::Index cols() const { return static_cast<Eigen::Index>(n_); }
    void perform_op(const double* x_in, double* y_out) const {
        std::memcpy(x_.data(), x_in, n_ * sizeof(double));
        auto y = pycauset::matrix_vector_multiply(a_, x_);
        if (auto* d = dynamic_cast<const DenseVector<double>*>(y.get()); d && d->get_scalar() == 1.0) {
            std::memcpy(y_out, d->data(), n_ * sizeof(double));
        } else {
            for (uint64_t i = 0; i < n_; ++i) y_out[i] = y->get_element_as_double(i);
        }
    }

private:
    CpuSolver& solver_;
    const MatrixBase& a_;
    uint64_t n_;
    mutable DenseVector<double> x_;
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

namespace {

// A real skew-symmetric A: M = -iA is Hermitian, and LAPACK's Hermitian solver gives its real
// eigenvalues mu with orthonormal eigenvectors u, A u = i mu u: exact +- pairs in real-arithmetic
// accuracy (?geev's eigenvectors were not orthogonal within a repeated eigenvalue, so the
// Sorkin-Johnston W built from them was wrong; review 3.4 C-3). The row-major buffer of iA,
// read column-major, is (iA)^T = -iA. `vectors`: also compute u (columns of `h`, column-major).
std::vector<double> skew_hermitian_solve(const MatrixBase& a, std::vector<std::complex<double>>& h, bool vectors,
                                         const char* op) {
    const uint64_t n = a.rows();
    auto data = to_memory_flat_real_square(a);  // folds in the matrix scalar
    h.assign(n * n, {});
    for (uint64_t idx = 0; idx < n * n; ++idx) h[idx] = std::complex<double>(0.0, data[idx]);
    data.clear();
    data.shrink_to_fit();
    std::vector<double> mu(n);
    lapack_int info = LAPACKE_zheevd(LAPACK_COL_MAJOR, vectors ? 'V' : 'N', 'U', static_cast<lapack_int>(n),
        reinterpret_cast<lapack_complex_double*>(h.data()), static_cast<lapack_int>(n), mu.data());
    if (info > 0) throw std::runtime_error(std::string(op) + " failed to converge");
    if (info < 0) throw std::runtime_error(std::string(op) + ": illegal argument");
    return mu;
}

// The order of the eigenvalues i*mu, shared by eigvals_skew and eig_skew: descending |mu|;
// values whose |mu| agree to 1e-12 of the largest are one pair (or cluster) and go +i first.
// (The two functions used different orders, and the pair loop skipped pairs it did not swap.)
std::vector<size_t> skew_order(const std::vector<double>& mu) {
    const size_t n = mu.size();
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](size_t x, size_t y) { return std::abs(mu[x]) > std::abs(mu[y]); });
    double scale = 0.0;
    for (double m : mu) scale = std::max(scale, std::abs(m));
    const double tol = 1e-12 * scale;
    for (size_t i = 0; i < n;) {
        size_t j = i + 1;
        while (j < n && std::abs(mu[idx[i]]) - std::abs(mu[idx[j]]) <= tol) ++j;
        std::stable_sort(idx.begin() + i, idx.begin() + j, [&](size_t x, size_t y) { return mu[x] > mu[y]; });
        i = j;
    }
    return idx;
}

}  // namespace

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
    std::vector<std::complex<double>> h;
    const std::vector<double> mu = skew_hermitian_solve(a, h, false, "eigvals_skew");
    const std::vector<size_t> order = skew_order(mu);
    for (int t = 0; t < kk; ++t) out_c->set(t, std::complex<double>(0.0, mu[order[t]]));
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

    std::vector<std::complex<double>> h;
    const std::vector<double> mu = skew_hermitian_solve(a, h, true, "eig_skew");
    const std::vector<size_t> order = skew_order(mu);
    for (int t = 0; t < kk; ++t) {
        const size_t src = order[t];
        out_w->set(t, std::complex<double>(0.0, mu[src]));
        for (uint64_t r = 0; r < n; ++r) out_v->set(r, t, h[src * n + r]);
    }
}

} // namespace pycauset
