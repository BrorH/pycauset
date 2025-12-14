#pragma once

#include "CudaDevice.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include <vector>

namespace pycauset {

class CudaSolver {
public:
    CudaSolver(CudaDevice* device);
    
    // Main entry point for inversion
    void invert(const MatrixBase& in, MatrixBase& out);

    // Main entry point for eigenvalues
    void eigvals(const MatrixBase& matrix, ComplexVector& result);

    void add(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);
    void subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);
    void multiply_scalar(const MatrixBase& a, double scalar, MatrixBase& result);

    // BitMatrix Operations
    void matmul_bit(const DenseMatrix<bool>& a, const DenseMatrix<bool>& b, DenseMatrix<int32_t>& result);

private:
    CudaDevice* device_;
    
    // Generalized Streaming GEMM for C = alpha*A*B + beta*C
    // A: m x k, B: k x n, C: m x n
    // All matrices are in Host Memory (Memory Mapped)
    
    // Double Precision
    void gemm_streaming(
        size_t m, size_t n, size_t k,
        double alpha,
        const double* A, size_t lda,
        const double* B, size_t ldb,
        double beta,
        double* C, size_t ldc
    );

    // Single Precision
    void gemm_streaming(
        size_t m, size_t n, size_t k,
        float alpha,
        const float* A, size_t lda,
        const float* B, size_t ldb,
        float beta,
        float* C, size_t ldc
    );



    // --- Helper Methods for Blocked LU ---

    // Factors the panel A[k:N, k:k+b] on GPU.
    // Returns pivots.
    std::vector<int> factor_panel(double* data, size_t N, size_t k, size_t b);
    std::vector<int> factor_panel(float* data, size_t N, size_t k, size_t b);

    // Applies pivots to the trailing submatrix.
    // If full_row is true, applies to columns 0..N (for RHS).
    // If full_row is false, applies to columns k+b..N (for LU).
    void apply_pivots(double* data, size_t N, size_t k, size_t b, const std::vector<int>& pivots, bool full_row = false);
    void apply_pivots(float* data, size_t N, size_t k, size_t b, const std::vector<int>& pivots, bool full_row = false);

    // Solves L11 * U12 = A12.
    void solve_row_panel(double* data, size_t N, size_t k, size_t b);
    void solve_row_panel(float* data, size_t N, size_t k, size_t b);

    // Solves L * X = B (Forward Substitution)
    // L is unit lower triangular (from LU factorization in 'lu_data')
    // B is 'rhs_data' (overwritten with X)
    void solve_forward(const double* lu_data, double* rhs_data, size_t N, size_t block_size);
    void solve_forward(const float* lu_data, float* rhs_data, size_t N, size_t block_size);

    // Solves U * X = B (Backward Substitution)
    // U is upper triangular (from LU factorization in 'lu_data')
    // B is 'rhs_data' (overwritten with X)
    void solve_backward(const double* lu_data, double* rhs_data, size_t N, size_t block_size);
    void solve_backward(const float* lu_data, float* rhs_data, size_t N, size_t block_size);
};

} // namespace pycauset
