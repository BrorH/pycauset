#pragma once

#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/vector/VectorBase.hpp"
#include <complex>
#include <memory>
#include <string>

namespace pycauset {

template <typename T>
class TriangularMatrix;
struct HardwareProfile;

class ComputeDevice {
public:
    virtual ~ComputeDevice() = default;

    // Core Operations
    virtual void matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) = 0;
    
    // Inversion
    virtual void inverse(const MatrixBase& in, MatrixBase& out) = 0;


    // Batch Matrix-Vector Multiplication (A * X -> Y)
    // A is N x N
    // X is N x b (Row Major)
    // Y is N x b (Row Major)
    virtual void batch_gemv(const MatrixBase& A, const double* x_data, double* y_data, size_t b) = 0;

    // Matrix-Vector Operations
    virtual void matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, VectorBase& result) = 0;
    virtual void vector_matrix_multiply(const VectorBase& v, const MatrixBase& m, VectorBase& result) = 0;
    virtual void outer_product(const VectorBase& a, const VectorBase& b, MatrixBase& result) = 0;

    // Element-wise Operations
    virtual void add(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) = 0;
    virtual void subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) = 0;
    virtual void elementwise_multiply(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) = 0;
    virtual void elementwise_divide(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) = 0;
    virtual void multiply_scalar(const MatrixBase& a, double scalar, MatrixBase& result) = 0;

    // Vector Operations
    virtual double dot(const VectorBase& a, const VectorBase& b) = 0;
    virtual std::complex<double> dot_complex(const VectorBase& a, const VectorBase& b) = 0;
    virtual std::complex<double> sum(const VectorBase& v) = 0;
    virtual double l2_norm(const VectorBase& v) = 0;
    virtual void add_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) = 0;
    virtual void subtract_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) = 0;
    virtual void scalar_multiply_vector(const VectorBase& a, double scalar, VectorBase& result) = 0;
    virtual void scalar_multiply_vector_complex(const VectorBase& a, std::complex<double> scalar, VectorBase& result) = 0;
    virtual void scalar_add_vector(const VectorBase& a, double scalar, VectorBase& result) = 0;

    // Specialized vector ops
    virtual void cross_product(const VectorBase& a, const VectorBase& b, VectorBase& result) = 0;

    // Special solvers / structured ops
    virtual std::unique_ptr<TriangularMatrix<double>> compute_k_matrix(
        const TriangularMatrix<bool>& C,
        double a,
        const std::string& output_path,
        int num_threads) = 0;

    // Reductions / norms
    virtual double frobenius_norm(const MatrixBase& m) = 0;
    virtual std::complex<double> sum(const MatrixBase& m) = 0;

    // Linalg utilities
    virtual double trace(const MatrixBase& m) = 0;
    virtual double determinant(const MatrixBase& m) = 0;

    // Factorizations
    // Computes a QR factorization of `in` into `Q` and `R`.
    // Contract: in is square float64; Q and R are pre-allocated square float64 dense matrices.
    virtual void qr(const MatrixBase& in, MatrixBase& Q, MatrixBase& R) = 0;

    // Computes Cholesky factorization of `in` into `out`.
    // Contract: in is SPD. out is lower triangular (if supported) or full with zeros.
    virtual void cholesky(const MatrixBase& in, MatrixBase& out) = 0;

    // Computes LU factorization of `in`.
    // Contract: P (permutation), L (lower unit), U (upper). P is permutation vector or matrix?
    // For now, let's just output P, L, U as matrices.
    virtual void lu(const MatrixBase& in, MatrixBase& P, MatrixBase& L, MatrixBase& U) = 0;

    // Computes SVD
    virtual void svd(const MatrixBase& in, MatrixBase& U, VectorBase& S, MatrixBase& VT) = 0;
    
    // Solves linear system AX = B
    virtual void solve(const MatrixBase& A, const MatrixBase& B, MatrixBase& X) = 0;

    // Eigenvalues (Arnoldi/Lanczos-style top-k)
    // Contract: out is a pre-allocated real vector of length k.
    virtual void eigvals_arnoldi(const MatrixBase& a, VectorBase& out, int k, int m, double tol) = 0;

    // Real Symmetric / Complex Hermitian Eigenvalue Decomposition
    // Contract: `in` is symmetric/hermitian. `eigenvalues` is vector of length N. `eigenvectors` is NxN matrix.
    // 'uplo' = 'L' or 'U' (default 'L').
    virtual void eigh(const MatrixBase& in, VectorBase& eigenvalues, MatrixBase& eigenvectors, char uplo = 'L') = 0;

    // Eigenvalues only
    virtual void eigvalsh(const MatrixBase& in, VectorBase& eigenvalues, char uplo = 'L') = 0;

    // General Non-Symmetric Eigenvalue Decomposition
    // Contract: `in` is square. `eigenvalues` and `eigenvectors` should handle complex types.
    virtual void eig(const MatrixBase& in, VectorBase& eigenvalues, MatrixBase& eigenvectors) = 0;
    
    // General Non-Symmetric Eigenvalues only
    virtual void eigvals(const MatrixBase& in, VectorBase& eigenvalues) = 0;

    // Device Info
    virtual std::string name() const = 0;
    virtual bool is_gpu() const = 0;
    
    // 0 = Unknown, 1 = Float32, 2 = Float64
    virtual int preferred_precision() const { return 1; } // Default to Float32

    // Hardware profiling (GPU implementations may populate this)
    virtual bool fill_hardware_profile(HardwareProfile& profile, bool run_benchmarks) { return false; }

    // Memory Management
    virtual void* allocate_pinned(size_t size) { return nullptr; }
    virtual void free_pinned(void* ptr) {}
    virtual void register_host_memory(void* ptr, size_t size) {}
    virtual void unregister_host_memory(void* ptr) {}
};

} // namespace pycauset
