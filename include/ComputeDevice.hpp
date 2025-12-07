#pragma once

#include "MatrixBase.hpp"
#include "VectorBase.hpp"
#include "ComplexVector.hpp"
#include <memory>
#include <string>

namespace pycauset {

class ComputeDevice {
public:
    virtual ~ComputeDevice() = default;

    // Core Operations
    virtual void matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) = 0;
    
    // Inversion
    virtual void inverse(const MatrixBase& in, MatrixBase& out) = 0;

    // Eigenvalue Solvers
    virtual void eigvals(const MatrixBase& matrix, ComplexVector& result) = 0;

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
    virtual void multiply_scalar(const MatrixBase& a, double scalar, MatrixBase& result) = 0;

    // Vector Operations
    virtual double dot(const VectorBase& a, const VectorBase& b) = 0;
    virtual void add_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) = 0;
    virtual void subtract_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) = 0;
    virtual void scalar_multiply_vector(const VectorBase& a, double scalar, VectorBase& result) = 0;
    virtual void scalar_add_vector(const VectorBase& a, double scalar, VectorBase& result) = 0;

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
