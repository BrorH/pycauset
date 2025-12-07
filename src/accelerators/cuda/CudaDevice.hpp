#pragma once

#include "ComputeDevice.hpp"
#include "AcceleratorConfig.hpp"
#include "DenseMatrix.hpp"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace pycauset {

class CudaDevice : public ComputeDevice {
public:
    CudaDevice(const AcceleratorConfig& config);
    ~CudaDevice();

    // Core Operations
    void matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) override;
    void inverse(const MatrixBase& in, MatrixBase& out) override;
    
    // Internal helper for in-core inversion (called by CudaSolver)
    void inverse_incore(const MatrixBase& in, MatrixBase& out);

    void eigvals(const MatrixBase& matrix, ComplexVector& result) override;
    void batch_gemv(const MatrixBase& A, const double* x_data, double* y_data, size_t b) override;

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
        // Pascal (6.x) and Maxwell (5.x) have poor FP64 performance (1/32)
        // Volta (7.0) has 1/2 FP64 rate (very good)
        // Turing (7.5) has 1/32 FP64 rate
        // Ampere (8.0/8.6) has 1/2 (A100) or 1/64 (Consumer)
        
        // Heuristic: Consumer cards usually prefer Float32.
        // Only Tesla V100/A100/H100 prefer Float64 or have good support.
        
        // For now, default to Float32 for everything unless we detect specific high-end cards?
        // Actually, user wants "if you detect that a user would have better performance... I WANT THAT AUTOMATICALLY DETECTED"
        
        // Simple heuristic:
        // If CC == 6.0 (P100) -> Good FP64 (1/2)
        // If CC == 6.1 (GTX 10 series) -> Bad FP64 (1/32) -> Prefer Float32
        // If CC == 7.0 (V100) -> Good FP64 (1/2)
        // If CC == 7.5 (RTX 20 series) -> Bad FP64 (1/32) -> Prefer Float32
        // If CC == 8.0 (A100) -> Good FP64 (1/2)
        // If CC == 8.6 (RTX 30 series) -> Bad FP64 (1/64) -> Prefer Float32
        
        if (cc_major_ == 6 && cc_minor_ == 0) return 2; // P100
        if (cc_major_ == 7 && cc_minor_ == 0) return 2; // V100
        if (cc_major_ == 8 && cc_minor_ == 0) return 2; // A100
        
        return 1; // Float32
    }

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
