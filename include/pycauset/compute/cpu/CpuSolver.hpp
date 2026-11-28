#pragma once

#include "pycauset/matrix/MatrixBase.hpp"
#include <complex>
#include <memory>
#include <string>
#include <vector>

namespace pycauset {
class VectorBase; // Forward declaration
template <typename T>
class TriangularMatrix;

// LU pivots (U's diagonal) and permutation parity of a plain square matrix: det and slogdet
// both come from these (math/Eigen.cpp). Exactly singular: an exact 0 pivot.
struct LuPivots {
    std::vector<std::complex<double>> pivots;
    bool odd_permutation = false;
};

// The CPU kernels: the one compute device (compute/ComputeContext.hpp). Each kernel's contract
// (operand classes and views it reads, result it writes) is at its definition.
class CpuSolver {
public:
    CpuSolver() = default;
    ~CpuSolver() = default;

    // Core Operations
    void matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);
    void cholesky(const MatrixBase& in, MatrixBase& out);

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
        const std::string& output_path);

    double frobenius_norm(const MatrixBase& m);

    LuPivots lu_pivots(const MatrixBase& m);
    void qr(const MatrixBase& in, MatrixBase& Q, MatrixBase& R);
    void lu(const MatrixBase& in, MatrixBase& P, MatrixBase& L, MatrixBase& U);
    void svd(const MatrixBase& in, MatrixBase& U, VectorBase& S, MatrixBase& VT);
    void solve(const MatrixBase& A, const MatrixBase& B, MatrixBase& X);
    void lstsq(const MatrixBase& A, const MatrixBase& B, MatrixBase& X);

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
};

} // namespace pycauset
