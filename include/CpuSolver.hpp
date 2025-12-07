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

    void add(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);
    void subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);
    void multiply_scalar(const MatrixBase& a, double scalar, MatrixBase& result);

private:
    // Helper for dense matrix multiplication
    void matmul_dense(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);
};

} // namespace pycauset
