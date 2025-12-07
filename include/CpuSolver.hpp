#pragma once

#include "MatrixBase.hpp"
#include "ComplexVector.hpp"
#include <vector>

namespace pycauset {

class CpuSolver {
public:
    CpuSolver() = default;
    ~CpuSolver() = default;

    // Core Operations
    void matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);
    void inverse(const MatrixBase& in, MatrixBase& out);
    void eigvals(const MatrixBase& matrix, ComplexVector& result);
    void batch_gemv(const MatrixBase& A, const double* x_data, double* y_data, size_t b);

    void matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, VectorBase& result);
    void vector_matrix_multiply(const VectorBase& v, const MatrixBase& m, VectorBase& result);
    void outer_product(const VectorBase& a, const VectorBase& b, MatrixBase& result);

    void add(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);
    void subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);
    void elementwise_multiply(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);
    void multiply_scalar(const MatrixBase& a, double scalar, MatrixBase& result);

    // Vector Operations
    double dot(const VectorBase& a, const VectorBase& b);
    void add_vector(const VectorBase& a, const VectorBase& b, VectorBase& result);
    void subtract_vector(const VectorBase& a, const VectorBase& b, VectorBase& result);
    void scalar_multiply_vector(const VectorBase& a, double scalar, VectorBase& result);
    void scalar_add_vector(const VectorBase& a, double scalar, VectorBase& result);

private:
    // Helper for dense matrix multiplication
    void matmul_dense(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);
};

} // namespace pycauset
