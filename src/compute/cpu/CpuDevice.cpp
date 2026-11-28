#include "pycauset/compute/cpu/CpuDevice.hpp"
#include "pycauset/core/MemoryGovernor.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/matrix/IdentityMatrix.hpp"
#include <stdexcept>

namespace pycauset {

// Out of core is the mapped storage itself: every kernel runs on the operands' pages and the
// OS pages them in and out. (A tiled StreamingManager ran when operands did not fit; it was
// slower than paging in every case measured and allocated about 1.6x the free RAM, review 3.5
// C-1, C-2; removed.)

void CpuDevice::matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    solver_.matmul(a, b, result);
}

void CpuDevice::cholesky(const MatrixBase& in, MatrixBase& out) {
    solver_.cholesky(in, out);
}

void CpuDevice::matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, VectorBase& result) {
    solver_.matrix_vector_multiply(m, v, result);
}

void CpuDevice::add(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    solver_.add(a, b, result);
}

void CpuDevice::subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    solver_.subtract(a, b, result);
}

void CpuDevice::elementwise_multiply(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    solver_.elementwise_multiply(a, b, result);
}

void CpuDevice::elementwise_divide(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    solver_.elementwise_divide(a, b, result);
}

double CpuDevice::dot(const VectorBase& a, const VectorBase& b) {
    return solver_.dot(a, b);
}

std::complex<double> CpuDevice::dot_complex(const VectorBase& a, const VectorBase& b) {
    return solver_.dot_complex(a, b);
}


double CpuDevice::l2_norm(const VectorBase& v) {
    return solver_.l2_norm(v);
}






std::unique_ptr<TriangularMatrix<double>> CpuDevice::compute_k_matrix(
    const TriangularMatrix<bool>& C,
    double a,
    const std::string& output_path,
    int num_threads
) {
    return solver_.compute_k_matrix(C, a, output_path, num_threads);
}

double CpuDevice::frobenius_norm(const MatrixBase& m) {
    return solver_.frobenius_norm(m);
}



LuPivots CpuDevice::lu_pivots(const MatrixBase& m) {
    return solver_.lu_pivots(m);
}

void CpuDevice::qr(const MatrixBase& in, MatrixBase& Q, MatrixBase& R) {
    solver_.qr(in, Q, R);
}

void CpuDevice::lu(const MatrixBase& in, MatrixBase& P, MatrixBase& L, MatrixBase& U) {
    solver_.lu(in, P, L, U);
}

void CpuDevice::svd(const MatrixBase& in, MatrixBase& U, VectorBase& S, MatrixBase& VT) {
    solver_.svd(in, U, S, VT);
}

void CpuDevice::solve(const MatrixBase& A, const MatrixBase& B, MatrixBase& X) {
    solver_.solve(A, B, X);
}

void CpuDevice::lstsq(const MatrixBase& A, const MatrixBase& B, MatrixBase& X) {
    solver_.lstsq(A, B, X);
}

void CpuDevice::eigvals_arnoldi(const MatrixBase& a, VectorBase& out, int k, int m, double tol) {
    solver_.eigvals_arnoldi(a, out, k, m, tol);
}

void CpuDevice::eigvals_skew(const MatrixBase& a, VectorBase& out, int k) {
    solver_.eigvals_skew(a, out, k);
}

void CpuDevice::eig_skew(const MatrixBase& a, VectorBase& eigenvalues, MatrixBase& eigenvectors, int k) {
    solver_.eig_skew(a, eigenvalues, eigenvectors, k);
}

bool CpuDevice::fill_hardware_profile(HardwareProfile& profile, bool run_benchmarks) {
    (void)profile;
    (void)run_benchmarks;
    return false;
}

void CpuDevice::eigh(const MatrixBase& in, VectorBase& eigenvalues, MatrixBase& eigenvectors, char uplo) {
    solver_.eigh(in, eigenvalues, eigenvectors, uplo);
}

void CpuDevice::eigvalsh(const MatrixBase& in, VectorBase& eigenvalues, char uplo) {
    solver_.eigvalsh(in, eigenvalues, uplo);
}

void CpuDevice::eig(const MatrixBase& in, VectorBase& eigenvalues, MatrixBase& eigenvectors) {
    solver_.eig(in, eigenvalues, eigenvectors);
}

void CpuDevice::eigvals(const MatrixBase& in, VectorBase& eigenvalues) {
    solver_.eigvals(in, eigenvalues);
}

} // namespace pycauset
