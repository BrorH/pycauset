#pragma once

#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/vector/VectorBase.hpp"
#include <complex>
#include <memory>
#include <string>

namespace pycauset {

template <typename T>
class TriangularMatrix;

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

    // Device Info
    virtual std::string name() const = 0;
    virtual bool is_gpu() const = 0;
    
    // 0 = Unknown, 1 = Float32, 2 = Float64
    virtual int preferred_precision() const { return 1; } // Default to Float32

    // Memory Management
    virtual void* allocate_pinned(size_t size) { return nullptr; }
    virtual void free_pinned(void* ptr) {}
    virtual void register_host_memory(void* ptr, size_t size) {}
    virtual void unregister_host_memory(void* ptr) {}
};

} // namespace pycauset
