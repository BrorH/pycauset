#include "CpuDevice.hpp"
#include "MatrixOperations.hpp"
#include "ParallelUtils.hpp"
#include "TriangularMatrix.hpp"
#include "DenseMatrix.hpp"
#include "DiagonalMatrix.hpp"
#include "IdentityMatrix.hpp"
#include <stdexcept>

namespace pycauset {

void CpuDevice::matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    solver_.matmul(a, b, result);
}

void CpuDevice::inverse(const MatrixBase& in, MatrixBase& out) {
    solver_.inverse(in, out);
}

void CpuDevice::eigvals(const MatrixBase& matrix, ComplexVector& result) {
    solver_.eigvals(matrix, result);
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

void CpuDevice::multiply_scalar(const MatrixBase& a, double scalar, MatrixBase& result) {
    solver_.multiply_scalar(a, scalar, result);
}

double CpuDevice::dot(const VectorBase& a, const VectorBase& b) {
    return solver_.dot(a, b);
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

void CpuDevice::scalar_add_vector(const VectorBase& a, double scalar, VectorBase& result) {
    solver_.scalar_add_vector(a, scalar, result);
}

} // namespace pycauset
