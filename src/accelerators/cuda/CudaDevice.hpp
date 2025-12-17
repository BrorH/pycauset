#pragma once

#include "pycauset/compute/ComputeDevice.hpp"
#include "pycauset/compute/AcceleratorConfig.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace pycauset {

class CudaDevice : public ComputeDevice {
    friend class CudaSolver;
public:
    CudaDevice(const AcceleratorConfig& config);
    ~CudaDevice();

    // Core Operations
    void matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) override;
    void inverse(const MatrixBase& in, MatrixBase& out) override;
    
    // Internal helper for in-core inversion (called by CudaSolver)
    void inverse_incore(const MatrixBase& in, MatrixBase& out);
    void batch_gemv(const MatrixBase& A, const double* x_data, double* y_data, size_t b) override;

    // Matrix-Vector Operations
    void matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, VectorBase& result) override;
    void vector_matrix_multiply(const VectorBase& v, const MatrixBase& m, VectorBase& result) override;
    void outer_product(const VectorBase& a, const VectorBase& b, MatrixBase& result) override;

    // Element-wise Operations
    void add(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) override;
    void subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) override;
    void elementwise_multiply(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) override;
    void elementwise_divide(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) override;
    void multiply_scalar(const MatrixBase& a, double scalar, MatrixBase& result) override;

    double dot(const VectorBase& a, const VectorBase& b) override;
    double l2_norm(const VectorBase& v) override;
    void add_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) override;
    void subtract_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) override;
    void scalar_multiply_vector(const VectorBase& a, double scalar, VectorBase& result) override;
    void scalar_add_vector(const VectorBase& a, double scalar, VectorBase& result) override;

    double frobenius_norm(const MatrixBase& m) override;

    std::string name() const override { return "CUDA (NVIDIA GPU)"; }
    bool is_gpu() const override { return true; }
    
    // Hardware Capability
    int get_compute_capability_major() const { return cc_major_; }
    int get_compute_capability_minor() const { return cc_minor_; }

    // Accessors for Solver
    cublasHandle_t get_cublas_handle() const { return cublas_handle_; }
    cusolverDnHandle_t get_cusolver_handle() const { return cusolver_handle_; }
    size_t get_available_memory_bytes() { return get_available_memory(); }
    const AcceleratorConfig& get_config() const { return config_; }

    // Public helper for checking errors
    void check_cuda_error(cudaError_t result, const char* func) { check_cuda(result, func); }
    void check_cublas_error(cublasStatus_t result, const char* func) { check_cublas(result, func); }
    void check_cusolver_error(cusolverStatus_t result, const char* func) { check_cusolver(result, func); }

    int preferred_precision() const override {
        if (cc_major_ == 6 && cc_minor_ == 0) return 2; // P100
        if (cc_major_ == 7 && cc_minor_ == 0) return 2; // V100
        if (cc_major_ == 8 && cc_minor_ == 0) return 2; // A100
        
        return 1; // Float32
    }

    // Memory Management
    void* allocate_pinned(size_t size) override;
    void free_pinned(void* ptr) override;
    void register_host_memory(void* ptr, size_t size) override;
    void unregister_host_memory(void* ptr) override;

private:
    cublasHandle_t cublas_handle_;
    cusolverDnHandle_t cusolver_handle_;
    int cc_major_ = 0;
    int cc_minor_ = 0;
    AcceleratorConfig config_;

    // Helper to check CUDA errors
    void check_cuda(cudaError_t result, const char* func);
    void check_cublas(cublasStatus_t result, const char* func);
    void check_cusolver(cusolverStatus_t result, const char* func);

    // Memory Management
    size_t get_available_memory();

    // Streaming implementations
    void batch_gemv_streaming(const MatrixBase& A, const double* x_data, double* y_data, size_t b, size_t available_mem);
    void matmul_streaming(const DenseMatrix<double>* a, const DenseMatrix<double>* b, DenseMatrix<double>* c, size_t available_mem);
    void matmul_streaming(const DenseMatrix<float>* a, const DenseMatrix<float>* b, DenseMatrix<float>* c, size_t available_mem);

    // Persistent buffers
    double *d_A_ = nullptr;
    double *d_B_ = nullptr;

    double *d_C_ = nullptr;
    size_t buffer_size_ = 0; // Number of elements (doubles) allocated per buffer

    // Persistent buffers for Float32
    float *d_A_float_ = nullptr;
    float *d_B_float_ = nullptr;
    float *d_C_float_ = nullptr;
    size_t buffer_size_float_ = 0;

    void ensure_buffers(size_t n_elements);
    void ensure_float_buffers(size_t n_elements);
    void free_buffers();
};

// Factory function to be exported
extern "C" {
    __declspec(dllexport) ComputeDevice* create_cuda_device(const AcceleratorConfig* config);
}

} // namespace pycauset
