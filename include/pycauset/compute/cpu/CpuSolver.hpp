#pragma once

#include "pycauset/compute/ComputeDevice.hpp"
#include "pycauset/matrix/MatrixBase.hpp"
#include <complex>
#include <memory>
#include <string>
#include <vector>

namespace pycauset {
class VectorBase; // Forward declaration
template <typename T>
class TriangularMatrix;

class CpuSolver {
public:
    CpuSolver() = default;
    ~CpuSolver() = default;

    // Core Operations
    void matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);
    /**
     * @brief Generalized Matrix Multiplication (C = alpha*A*B + beta*C)
     * Exposed for use by CpuComputeWorker (tiled execution).
     */
    void gemm(const MatrixBase& a, const MatrixBase& b, MatrixBase& c, double alpha, double beta);

    void cholesky(const MatrixBase& in, MatrixBase& out);
    void batch_gemv(const MatrixBase& A, const double* x_data, double* y_data, size_t b);

    void matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, VectorBase& result);

    void add(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);
    void subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);
    void elementwise_multiply(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);
    void elementwise_divide(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);

    double dot(const VectorBase& a, const VectorBase& b);
    std::complex<double> dot_complex(const VectorBase& a, const VectorBase& b);
    double l2_norm(const VectorBase& v);


    std::unique_ptr<TriangularMatrix<double>> compute_k_matrix(
        const TriangularMatrix<bool>& C,
        double a,
        const std::string& output_path,
        int num_threads);

    double frobenius_norm(const MatrixBase& m);

    LuPivots lu_pivots(const MatrixBase& m);
    void qr(const MatrixBase& in, MatrixBase& Q, MatrixBase& R);
    void lu(const MatrixBase& in, MatrixBase& P, MatrixBase& L, MatrixBase& U);
    void svd(const MatrixBase& in, MatrixBase& U, VectorBase& S, MatrixBase& VT);
    void solve(const MatrixBase& A, const MatrixBase& B, MatrixBase& X);

    // Eigen Solvers
    // Hermitian/Symmetric (Real eigenvalues)
    void eigh(const MatrixBase& in, VectorBase& eigenvalues, MatrixBase& eigenvectors, char uplo = 'L');
    void eigvalsh(const MatrixBase& in, VectorBase& eigenvalues, char uplo = 'L');
    
    // General (Complex eigenvalues)
    void eig(const MatrixBase& in, VectorBase& eigenvalues, MatrixBase& eigenvector_right);
    void eigvals(const MatrixBase& in, VectorBase& eigenvalues);

    void eigvals_arnoldi(const MatrixBase& a, VectorBase& out, int k, int m, double tol);
    void eigvals_skew(const MatrixBase& a, VectorBase& out, int k);
    void eig_skew(const MatrixBase& a, VectorBase& eigenvalues, MatrixBase& eigenvectors, int k);

private:
};

} // namespace pycauset
