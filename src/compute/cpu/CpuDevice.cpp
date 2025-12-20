#include "pycauset/compute/cpu/CpuDevice.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/matrix/IdentityMatrix.hpp"
#include <stdexcept>

namespace pycauset {

void CpuDevice::matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    solver_.matmul(a, b, result);
}

void CpuDevice::inverse(const MatrixBase& in, MatrixBase& out) {
    solver_.inverse(in, out);
}

void CpuDevice::batch_gemv(const MatrixBase& A, const double* x_data, double* y_data, size_t b) {
    solver_.batch_gemv(A, x_data, y_data, b);
}

void CpuDevice::matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, VectorBase& result) {
    solver_.matrix_vector_multiply(m, v, result);
}

void CpuDevice::vector_matrix_multiply(const VectorBase& v, const MatrixBase& m, VectorBase& result) {
    solver_.vector_matrix_multiply(v, m, result);
}

void CpuDevice::outer_product(const VectorBase& a, const VectorBase& b, MatrixBase& result) {
    solver_.outer_product(a, b, result);
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

void CpuDevice::multiply_scalar(const MatrixBase& a, double scalar, MatrixBase& result) {
    solver_.multiply_scalar(a, scalar, result);
}

double CpuDevice::dot(const VectorBase& a, const VectorBase& b) {
    return solver_.dot(a, b);
}

std::complex<double> CpuDevice::dot_complex(const VectorBase& a, const VectorBase& b) {
    return solver_.dot_complex(a, b);
}

std::complex<double> CpuDevice::sum(const VectorBase& v) {
    return solver_.sum(v);
}

double CpuDevice::l2_norm(const VectorBase& v) {
    return solver_.l2_norm(v);
}

void CpuDevice::add_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) {
    solver_.add_vector(a, b, result);
}

void CpuDevice::subtract_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) {
    solver_.subtract_vector(a, b, result);
}

void CpuDevice::scalar_multiply_vector(const VectorBase& a, double scalar, VectorBase& result) {
    solver_.scalar_multiply_vector(a, scalar, result);
}

void CpuDevice::scalar_multiply_vector_complex(const VectorBase& a, std::complex<double> scalar, VectorBase& result) {
    solver_.scalar_multiply_vector_complex(a, scalar, result);
}

void CpuDevice::scalar_add_vector(const VectorBase& a, double scalar, VectorBase& result) {
    solver_.scalar_add_vector(a, scalar, result);
}

void CpuDevice::cross_product(const VectorBase& a, const VectorBase& b, VectorBase& result) {
    solver_.cross_product(a, b, result);
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

std::complex<double> CpuDevice::sum(const MatrixBase& m) {
    return solver_.sum(m);
}

double CpuDevice::trace(const MatrixBase& m) {
    return solver_.trace(m);
}

double CpuDevice::determinant(const MatrixBase& m) {
    return solver_.determinant(m);
}

void CpuDevice::qr(const MatrixBase& in, MatrixBase& Q, MatrixBase& R) {
    solver_.qr(in, Q, R);
}

} // namespace pycauset
