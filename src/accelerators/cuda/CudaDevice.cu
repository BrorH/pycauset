#include "CudaDevice.hpp"
#include "CudaSolver.hpp"
#include "AsyncStreamer.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/vector/ComplexVector.hpp"
#include "pycauset/math/Eigen.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <chrono>

namespace pycauset {

CudaDevice::CudaDevice(const AcceleratorConfig& config) : config_(config) {
    // Initialize CUDA context
    cudaError_t err = cudaSetDevice(config_.device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaSetDevice(" + std::to_string(config_.device_id) + ") failed: " + std::string(cudaGetErrorString(err)));
    }

    // Force context initialization
    cudaFree(0);

    // Get Device Properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, config_.device_id);
    cc_major_ = prop.major;
    cc_minor_ = prop.minor;
    // std::cout << "[PyCauset] Detected GPU Compute Capability: " << cc_major_ << "." << cc_minor_ << std::endl;

    // std::cout << "[PyCauset] Initializing cuBLAS..." << std::endl;
    try {
        check_cublas(cublasCreate(&cublas_handle_), "cublasCreate");
    } catch (const std::exception& e) {
        std::cerr << "[PyCauset] CRITICAL ERROR: cuBLAS initialization failed." << std::endl;
        std::cerr << "  This often indicates that your GPU architecture is not supported by the installed CUDA version." << std::endl;
        std::cerr << "  CUDA 13.0+ requires Volta (sm_70) or newer. Pascal (sm_61, e.g., GTX 10 series) is NOT supported." << std::endl;
        throw;
    }
    // std::cout << "[PyCauset] Initializing cuSolver..." << std::endl;
    check_cusolver(cusolverDnCreate(&cusolver_handle_), "cusolverDnCreate");
}

CudaDevice::~CudaDevice() {
    free_buffers();
    cublasDestroy(cublas_handle_);
    cusolverDnDestroy(cusolver_handle_);
}

void* CudaDevice::allocate_pinned(size_t size) {
    void* ptr = nullptr;
    // cudaHostAllocPortable: memory is portable to all CUDA contexts
    // cudaHostAllocMapped: maps allocation into device address space (Zero Copy) - Optional, but good for integrated GPUs
    // For discrete GPUs, we just want pinned memory for faster transfer.
    cudaError_t err = cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        // Fallback or return null?
        // std::cerr << "cudaHostAlloc failed: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    return ptr;
}

void CudaDevice::free_pinned(void* ptr) {
    if (ptr) {
        cudaFreeHost(ptr);
    }
}

void CudaDevice::register_host_memory(void* ptr, size_t size) {
    if (ptr && size > 0) {
        cudaHostRegister(ptr, size, cudaHostRegisterDefault);
    }
}

void CudaDevice::unregister_host_memory(void* ptr) {
    if (ptr) {
        cudaHostUnregister(ptr);
    }
}

void CudaDevice::free_buffers() {
    if (d_A_) { cudaFree(d_A_); d_A_ = nullptr; }
    if (d_B_) { cudaFree(d_B_); d_B_ = nullptr; }
    if (d_C_) { cudaFree(d_C_); d_C_ = nullptr; }
    buffer_size_ = 0;

    if (d_A_float_) { cudaFree(d_A_float_); d_A_float_ = nullptr; }
    if (d_B_float_) { cudaFree(d_B_float_); d_B_float_ = nullptr; }
    if (d_C_float_) { cudaFree(d_C_float_); d_C_float_ = nullptr; }
    buffer_size_float_ = 0;

    if (d_A_half_) { cudaFree(d_A_half_); d_A_half_ = nullptr; }
    if (d_B_half_) { cudaFree(d_B_half_); d_B_half_ = nullptr; }
    if (d_C_half_) { cudaFree(d_C_half_); d_C_half_ = nullptr; }
    buffer_size_half_ = 0;
}

void CudaDevice::ensure_buffers(size_t n_elements) {
    if (n_elements <= buffer_size_) return;

    // Only free double buffers
    if (d_A_) { cudaFree(d_A_); d_A_ = nullptr; }
    if (d_B_) { cudaFree(d_B_); d_B_ = nullptr; }
    if (d_C_) { cudaFree(d_C_); d_C_ = nullptr; }
    buffer_size_ = 0;

    size_t size_bytes = n_elements * sizeof(double);
    check_cuda(cudaMalloc(&d_A_, size_bytes), "cudaMalloc A");
    check_cuda(cudaMalloc(&d_B_, size_bytes), "cudaMalloc B");
    check_cuda(cudaMalloc(&d_C_, size_bytes), "cudaMalloc C");
    buffer_size_ = n_elements;
}

void CudaDevice::ensure_float_buffers(size_t n_elements) {
    if (n_elements <= buffer_size_float_) return;

    // Only free float buffers
    if (d_A_float_) { cudaFree(d_A_float_); d_A_float_ = nullptr; }
    if (d_B_float_) { cudaFree(d_B_float_); d_B_float_ = nullptr; }
    if (d_C_float_) { cudaFree(d_C_float_); d_C_float_ = nullptr; }
    buffer_size_float_ = 0;

    size_t size_bytes = n_elements * sizeof(float);
    check_cuda(cudaMalloc(&d_A_float_, size_bytes), "cudaMalloc A float");
    check_cuda(cudaMalloc(&d_B_float_, size_bytes), "cudaMalloc B float");
    check_cuda(cudaMalloc(&d_C_float_, size_bytes), "cudaMalloc C float");
    buffer_size_float_ = n_elements;
}

void CudaDevice::ensure_half_buffers(size_t n_elements) {
    if (n_elements <= buffer_size_half_) return;

    // Only free half buffers
    if (d_A_half_) { cudaFree(d_A_half_); d_A_half_ = nullptr; }
    if (d_B_half_) { cudaFree(d_B_half_); d_B_half_ = nullptr; }
    if (d_C_half_) { cudaFree(d_C_half_); d_C_half_ = nullptr; }
    buffer_size_half_ = 0;

    size_t size_bytes = n_elements * sizeof(__half);
    check_cuda(cudaMalloc(&d_A_half_, size_bytes), "cudaMalloc A half");
    check_cuda(cudaMalloc(&d_B_half_, size_bytes), "cudaMalloc B half");
    check_cuda(cudaMalloc(&d_C_half_, size_bytes), "cudaMalloc C half");
    buffer_size_half_ = n_elements;
}

size_t CudaDevice::get_available_memory() {
    size_t free_byte, total_byte;
    check_cuda(cudaMemGetInfo(&free_byte, &total_byte), "cudaMemGetInfo");
    
    size_t limit = free_byte;
    if (config_.memory_limit_bytes > 0 && config_.memory_limit_bytes < free_byte) {
        limit = config_.memory_limit_bytes;
    }

    // Reserve 10% or 500MB for system/overhead if using full VRAM (auto-detect mode)
    if (config_.memory_limit_bytes == 0) {
        size_t reserve = 500 * 1024 * 1024;
        if (limit > reserve) return limit - reserve;
        return 0;
    }
    
    return limit;
}

void CudaDevice::check_cuda(cudaError_t result, const char* func) {
    if (result != cudaSuccess) {
        std::string msg = "CUDA Error in " + std::string(func) + ": " + cudaGetErrorString(result);
        throw std::runtime_error(msg);
    }
}

void CudaDevice::check_cublas(cublasStatus_t result, const char* func) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS Error in " + std::string(func));
    }
}

void CudaDevice::check_cusolver(cusolverStatus_t result, const char* func) {
    if (result != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error("cuSOLVER Error in " + std::string(func));
    }
}

void CudaDevice::matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    // Try BitMatrix
    auto* a_bit = dynamic_cast<const DenseBitMatrix*>(&a);
    auto* b_bit = dynamic_cast<const DenseBitMatrix*>(&b);
    auto* c_int = dynamic_cast<DenseMatrix<int32_t>*>(&result);

    if (a_bit && b_bit && c_int) {
        CudaSolver solver(this);
        solver.matmul_bit(*a_bit, *b_bit, *c_int);
        return;
    }

    uint64_t n = a.size();
    if (b.size() != n || result.size() != n) {
        throw std::invalid_argument("Dimension mismatch");
    }
    size_t free_mem = get_available_memory();

    // Try Float32
    auto* a_float = dynamic_cast<const DenseMatrix<float>*>(&a);
    auto* b_float = dynamic_cast<const DenseMatrix<float>*>(&b);
    auto* c_float = dynamic_cast<DenseMatrix<float>*>(&result);

    if (a_float && b_float && c_float) {
        size_t size_bytes = n * n * sizeof(float);
        
        if (n * n > buffer_size_float_) {
            if (3 * size_bytes > free_mem + (buffer_size_float_ * sizeof(float) * 3)) {
                 matmul_streaming(a_float, b_float, c_float, free_mem);
                 return;
            }
            ensure_float_buffers(n * n);
        }

        float *d_A = d_A_float_;
        float *d_B = d_B_float_;
        float *d_C = d_C_float_;

        check_cuda(cudaMemcpy(d_A, a_float->data(), size_bytes, cudaMemcpyHostToDevice), "cudaMemcpy A float");
        check_cuda(cudaMemcpy(d_B, b_float->data(), size_bytes, cudaMemcpyHostToDevice), "cudaMemcpy B float");

        float alpha = 1.0f;
        float beta = 0.0f;

        check_cublas(cublasSgemm(cublas_handle_,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 n, n, n,
                                 &alpha,
                                 d_B, n,
                                 d_A, n,
                                 &beta,
                                 d_C, n), "cublasSgemm");
        
        check_cuda(cudaMemcpy(c_float->data(), d_C, size_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy C float");
        c_float->set_scalar(a_float->get_scalar() * b_float->get_scalar());
        return;
    }

    // Try Float16
    auto* a_half = dynamic_cast<const DenseMatrix<Float16>*>(&a);
    auto* b_half = dynamic_cast<const DenseMatrix<Float16>*>(&b);
    auto* c_half = dynamic_cast<DenseMatrix<Float16>*>(&result);

    if (a_half && b_half && c_half) {
        size_t size_bytes = n * n * sizeof(uint16_t);
        
        if (n * n > buffer_size_half_) {
            if (3 * size_bytes > free_mem + (buffer_size_half_ * sizeof(uint16_t) * 3)) {
                 matmul_streaming(a_half, b_half, c_half, free_mem);
                 return;
            }
            ensure_half_buffers(n * n);
        }

        __half *d_A = d_A_half_;
        __half *d_B = d_B_half_;
        __half *d_C = d_C_half_;

        check_cuda(cudaMemcpy(d_A, a_half->data(), size_bytes, cudaMemcpyHostToDevice), "cudaMemcpy A half");
        check_cuda(cudaMemcpy(d_B, b_half->data(), size_bytes, cudaMemcpyHostToDevice), "cudaMemcpy B half");

        __half alpha = __float2half(1.0f);
        __half beta = __float2half(0.0f);

        check_cublas(cublasHgemm(cublas_handle_,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 n, n, n,
                                 &alpha,
                                 d_B, n,
                                 d_A, n,
                                 &beta,
                                 d_C, n), "cublasHgemm");
        
        check_cuda(cudaMemcpy(c_half->data(), d_C, size_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy C half");
        c_half->set_scalar(a_half->get_scalar() * b_half->get_scalar());
        return;
    }

    // Try Double
    auto* a_dense = dynamic_cast<const DenseMatrix<double>*>(&a);
    auto* b_dense = dynamic_cast<const DenseMatrix<double>*>(&b);
    auto* c_dense = dynamic_cast<DenseMatrix<double>*>(&result);

    if (!a_dense || !b_dense || !c_dense) {
        throw std::runtime_error("CudaDevice::matmul only supports DenseMatrix<double>, <float>, or <Float16>");
    }

    size_t size_bytes = n * n * sizeof(double);
    
    if (n * n > buffer_size_) {
        if (3 * size_bytes > free_mem + (buffer_size_ * sizeof(double) * 3)) {
             matmul_streaming(a_dense, b_dense, c_dense, free_mem);
             return;
        }
        ensure_buffers(n * n);
    }

    double *d_A = d_A_;
    double *d_B = d_B_;
    double *d_C = d_C_;

    check_cuda(cudaMemcpy(d_A, a_dense->data(), size_bytes, cudaMemcpyHostToDevice), "cudaMemcpy A");
    check_cuda(cudaMemcpy(d_B, b_dense->data(), size_bytes, cudaMemcpyHostToDevice), "cudaMemcpy B");

    double alpha = 1.0;
    double beta = 0.0;

    check_cublas(cublasDgemm(cublas_handle_,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             n, n, n,
                             &alpha,
                             d_B, n,
                             d_A, n,
                             &beta,
                             d_C, n), "cublasDgemm");
    
    cudaDeviceSynchronize();
    check_cuda(cudaMemcpy(c_dense->data(), d_C, size_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy C");
    c_dense->set_scalar(a_dense->get_scalar() * b_dense->get_scalar());
}

void CudaDevice::inverse(const MatrixBase& in, MatrixBase& out) {
    CudaSolver solver(this);
    solver.invert(in, out);
}

void CudaDevice::inverse_incore(const MatrixBase& in, MatrixBase& out) {
    try {
        // Support for Double Precision
        if (auto* in_dense = dynamic_cast<const DenseMatrix<double>*>(&in)) {
            auto* out_dense = dynamic_cast<DenseMatrix<double>*>(&out);
            if (!out_dense) throw std::runtime_error("CudaDevice::inverse output must match input type (double)");

            uint64_t n = in.size();
            size_t size_bytes = n * n * sizeof(double);

            double *d_A;
            int *d_Ipiv, *d_Info;
            
            check_cuda(cudaMalloc(&d_A, size_bytes), "cudaMalloc A");
            check_cuda(cudaMalloc(&d_Ipiv, n * sizeof(int)), "cudaMalloc Ipiv");
            check_cuda(cudaMalloc(&d_Info, sizeof(int)), "cudaMalloc Info");

            check_cuda(cudaMemcpy(d_A, in_dense->data(), size_bytes, cudaMemcpyHostToDevice), "cudaMemcpy A");

            int lwork_getrf = 0;
            check_cusolver(cusolverDnDgetrf_bufferSize(cusolver_handle_, n, n, d_A, n, &lwork_getrf), "getrf_bufferSize");
            
            double *d_B;
            check_cuda(cudaMalloc(&d_B, size_bytes), "cudaMalloc B (Identity)");
            std::vector<double> h_I(n * n, 0.0);
            for(size_t i=0; i<n; ++i) h_I[i*n + i] = 1.0;
            check_cuda(cudaMemcpy(d_B, h_I.data(), size_bytes, cudaMemcpyHostToDevice), "cudaMemcpy Identity");

            double *d_Work;
            check_cuda(cudaMalloc(&d_Work, lwork_getrf * sizeof(double)), "cudaMalloc Work");

            check_cusolver(cusolverDnDgetrf(cusolver_handle_, n, n, d_A, n, d_Work, d_Ipiv, d_Info), "getrf");

            int info = 0;
            check_cuda(cudaMemcpy(&info, d_Info, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy Info");
            if (info < 0) throw std::runtime_error("LU Factorization failed: Illegal value");
            if (info > 0) throw std::runtime_error("Matrix is singular (LU failed)");

            check_cusolver(cusolverDnDgetrs(cusolver_handle_, CUBLAS_OP_N, n, n, d_A, n, d_Ipiv, d_B, n, d_Info), "getrs");

            check_cuda(cudaMemcpy(&info, d_Info, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy Info 2");
            if (info != 0) throw std::runtime_error("Matrix inversion failed");

            check_cuda(cudaMemcpy(out_dense->data(), d_B, size_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy Result");

            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_Ipiv);
            cudaFree(d_Info);
            cudaFree(d_Work);

            out_dense->set_scalar(1.0 / in_dense->get_scalar());
            return;
        }

        // Support for Single Precision (Float32)
        if (auto* in_dense = dynamic_cast<const DenseMatrix<float>*>(&in)) {
            auto* out_dense = dynamic_cast<DenseMatrix<float>*>(&out);
            if (!out_dense) throw std::runtime_error("CudaDevice::inverse output must match input type (float)");

            uint64_t n = in.size();
            size_t size_bytes = n * n * sizeof(float);

            float *d_A;
            int *d_Ipiv, *d_Info;
            
            check_cuda(cudaMalloc(&d_A, size_bytes), "cudaMalloc A");
            check_cuda(cudaMalloc(&d_Ipiv, n * sizeof(int)), "cudaMalloc Ipiv");
            check_cuda(cudaMalloc(&d_Info, sizeof(int)), "cudaMalloc Info");

            check_cuda(cudaMemcpy(d_A, in_dense->data(), size_bytes, cudaMemcpyHostToDevice), "cudaMemcpy A");

            int lwork_getrf = 0;
            check_cusolver(cusolverDnSgetrf_bufferSize(cusolver_handle_, n, n, d_A, n, &lwork_getrf), "getrf_bufferSize");
            
            float *d_B;
            check_cuda(cudaMalloc(&d_B, size_bytes), "cudaMalloc B (Identity)");
            std::vector<float> h_I(n * n, 0.0f);
            for(size_t i=0; i<n; ++i) h_I[i*n + i] = 1.0f;
            check_cuda(cudaMemcpy(d_B, h_I.data(), size_bytes, cudaMemcpyHostToDevice), "cudaMemcpy Identity");
            
            float *d_Work;
            check_cuda(cudaMalloc(&d_Work, lwork_getrf * sizeof(float)), "cudaMalloc Work");

            check_cusolver(cusolverDnSgetrf(cusolver_handle_, n, n, d_A, n, d_Work, d_Ipiv, d_Info), "getrf");

            int info = 0;
            check_cuda(cudaMemcpy(&info, d_Info, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy Info");
            if (info != 0) throw std::runtime_error("Matrix is singular (LU failed)");

            check_cusolver(cusolverDnSgetrs(cusolver_handle_, CUBLAS_OP_N, n, n, d_A, n, d_Ipiv, d_B, n, d_Info), "getrs");

            check_cuda(cudaMemcpy(&info, d_Info, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy Info 2");
            if (info != 0) throw std::runtime_error("Matrix inversion failed");

            check_cuda(cudaMemcpy(out_dense->data(), d_B, size_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy Result");

            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_Ipiv);
            cudaFree(d_Info);
            cudaFree(d_Work);

            out_dense->set_scalar(1.0 / in_dense->get_scalar());
            return;
        }
    } catch (const std::exception& e) {
        std::cerr << "[PyCauset] GPU Inverse failed (falling back to CPU): " << e.what() << std::endl;
        
        // Fallback to CPU implementation
        if (auto* in_dense = dynamic_cast<const DenseMatrix<double>*>(&in)) {
            auto* out_dense = dynamic_cast<DenseMatrix<double>*>(&out);
            if (out_dense) {
                auto res = in_dense->inverse(); // CPU Parallel
                auto* res_dense = dynamic_cast<DenseMatrix<double>*>(res.get());
                std::copy(res_dense->data(), res_dense->data() + in.size()*in.size(), out_dense->data());
                out_dense->set_scalar(res_dense->get_scalar());
                return;
            }
        }
        if (auto* in_dense = dynamic_cast<const DenseMatrix<float>*>(&in)) {
            // DenseMatrix<float> doesn't have inverse() implemented in header?
            // Wait, DenseMatrix is templated. Yes it does.
            auto* out_dense = dynamic_cast<DenseMatrix<float>*>(&out);
            if (out_dense) {
                auto res = in_dense->inverse();
                auto* res_dense = dynamic_cast<DenseMatrix<float>*>(res.get());
                std::copy(res_dense->data(), res_dense->data() + in.size()*in.size(), out_dense->data());
                out_dense->set_scalar(res_dense->get_scalar());
                return;
            }
        }
        throw; // Re-throw if fallback fails
    }

    throw std::runtime_error("CudaDevice::inverse only supports DenseMatrix<double> or DenseMatrix<float>");
}

void CudaDevice::eigvals(const MatrixBase& matrix, ComplexVector& result) {
    CudaSolver solver(this);
    solver.eigvals(matrix, result);
}

void CudaDevice::batch_gemv(const MatrixBase& A, const double* x_data, double* y_data, size_t b) {
    auto* a_double = dynamic_cast<const DenseMatrix<double>*>(&A);
    auto* a_float = dynamic_cast<const DenseMatrix<float>*>(&A);
    
    if (!a_double && !a_float) throw std::runtime_error("CudaDevice::batch_gemv only supports DenseMatrix<double> or DenseMatrix<float>");
    
    uint64_t n = A.size();
    size_t free_mem = get_available_memory();

    if (a_float) {
        size_t size_A = n * n * sizeof(float);
        size_t size_X_float = n * b * sizeof(float);
        size_t required_mem = size_A + 2 * size_X_float;
        
        if (required_mem > free_mem) {
            batch_gemv_streaming(A, x_data, y_data, b, free_mem);
            return;
        }
        
        float *d_A, *d_X, *d_Y;
        check_cuda(cudaMalloc(&d_A, size_A), "cudaMalloc A");
        check_cuda(cudaMalloc(&d_X, size_X_float), "cudaMalloc X");
        check_cuda(cudaMalloc(&d_Y, size_X_float), "cudaMalloc Y");
        
        check_cuda(cudaMemcpy(d_A, a_float->data(), size_A, cudaMemcpyHostToDevice), "cudaMemcpy A");
        
        std::vector<float> x_float(n * b);
        for(size_t i=0; i<n*b; ++i) x_float[i] = (float)x_data[i];
        check_cuda(cudaMemcpy(d_X, x_float.data(), size_X_float, cudaMemcpyHostToDevice), "cudaMemcpy X");
        
        float alpha = 1.0f;
        float beta = 0.0f;
        
        check_cublas(cublasSgemm(cublas_handle_,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 b, n, n,
                                 &alpha,
                                 d_X, b,
                                 d_A, n,
                                 &beta,
                                 d_Y, b), "cublasSgemm");
                                 
        std::vector<float> y_float(n * b);
        check_cuda(cudaMemcpy(y_float.data(), d_Y, size_X_float, cudaMemcpyDeviceToHost), "cudaMemcpy Y");
        for(size_t i=0; i<n*b; ++i) y_data[i] = (double)y_float[i];
        
        cudaFree(d_A);
        cudaFree(d_X);
        cudaFree(d_Y);
        return;
    }

    size_t size_A = n * n * sizeof(double);
    size_t size_X = n * b * sizeof(double);
    size_t required_mem = size_A + 2 * size_X;

    if (required_mem > free_mem) {
        batch_gemv_streaming(A, x_data, y_data, b, free_mem);
        return;
    }

    double *d_A, *d_X, *d_Y;
    check_cuda(cudaMalloc(&d_A, size_A), "cudaMalloc A");
    check_cuda(cudaMalloc(&d_X, size_X), "cudaMalloc X");
    check_cuda(cudaMalloc(&d_Y, size_X), "cudaMalloc Y");

    check_cuda(cudaMemcpy(d_A, a_double->data(), size_A, cudaMemcpyHostToDevice), "cudaMemcpy A");
    check_cuda(cudaMemcpy(d_X, x_data, size_X, cudaMemcpyHostToDevice), "cudaMemcpy X");

    double alpha = 1.0;
    double beta = 0.0;

    check_cublas(cublasDgemm(cublas_handle_,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             b, n, n,
                             &alpha,
                             d_X, b,
                             d_A, n,
                             &beta,
                             d_Y, b), "cublasDgemm");

    check_cuda(cudaMemcpy(y_data, d_Y, size_X, cudaMemcpyDeviceToHost), "cudaMemcpy Y");

    cudaFree(d_A);
    cudaFree(d_X);
    cudaFree(d_Y);
}

void CudaDevice::batch_gemv_streaming(const MatrixBase& A, const double* x_data, double* y_data, size_t b, size_t available_mem) {
    auto* a_double = dynamic_cast<const DenseMatrix<double>*>(&A);
    auto* a_float = dynamic_cast<const DenseMatrix<float>*>(&A);
    
    uint64_t n = A.size();
    
    if (a_float) {
        size_t size_X_float = n * b * sizeof(float);
        if (2 * size_X_float > available_mem) throw std::runtime_error("Not enough GPU memory");
        
        float *d_X, *d_Y;
        check_cuda(cudaMalloc(&d_X, size_X_float), "cudaMalloc X");
        check_cuda(cudaMalloc(&d_Y, size_X_float), "cudaMalloc Y");
        
        std::vector<float> x_float(n * b);
        for(size_t i=0; i<n*b; ++i) x_float[i] = (float)x_data[i];
        check_cuda(cudaMemcpy(d_X, x_float.data(), size_X_float, cudaMemcpyHostToDevice), "cudaMemcpy X");
        
        size_t mem_for_A = available_mem - 2 * size_X_float;
        size_t row_size = n * sizeof(float);
        size_t max_rows = mem_for_A / row_size / 2;
        if (max_rows == 0) throw std::runtime_error("Not enough GPU memory");
        
        size_t chunk_rows = (max_rows / 32) * 32;
        if (chunk_rows == 0) chunk_rows = max_rows;
        if (chunk_rows > n) chunk_rows = n;
        
        AsyncStreamer<float> streamer(chunk_rows * n, config_.device_id, config_.enable_async);
        
        cudaStream_t compute_stream;
        check_cuda(cudaStreamCreate(&compute_stream), "cudaStreamCreate");
        check_cublas(cublasSetStream(cublas_handle_, compute_stream), "cublasSetStream");
        
        const float* a_ptr = a_float->data();
        float alpha = 1.0f;
        float beta = 0.0f;
        
        for (size_t i = 0; i < n; i += chunk_rows) {
            size_t current_rows = std::min(chunk_rows, n - i);
            streamer.wait_for_write_buffer();
            float* h_pinned = streamer.get_host_write_buffer();
            std::copy(a_ptr + i * n, a_ptr + i * n + current_rows * n, h_pinned);
            streamer.submit_transfer(current_rows * n);
            float* d_A_chunk = streamer.get_device_read_buffer(compute_stream);
            
            check_cublas(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                     b, current_rows, n, &alpha, d_X, b, d_A_chunk, n, &beta, d_Y + i * b, b), "cublasSgemm");
            streamer.release_device_buffer(compute_stream);
        }
        check_cuda(cudaStreamSynchronize(compute_stream), "Sync Compute");
        
        std::vector<float> y_float(n * b);
        check_cuda(cudaMemcpy(y_float.data(), d_Y, size_X_float, cudaMemcpyDeviceToHost), "cudaMemcpy Y");
        for(size_t i=0; i<n*b; ++i) y_data[i] = (double)y_float[i];
        
        cudaFree(d_X);
        cudaFree(d_Y);
        cudaStreamDestroy(compute_stream);
        return;
    }

    // Double implementation
    size_t size_X = n * b * sizeof(double);
    if (2 * size_X > available_mem) throw std::runtime_error("Not enough GPU memory");
    
    double *d_X, *d_Y;
    check_cuda(cudaMalloc(&d_X, size_X), "cudaMalloc X");
    check_cuda(cudaMalloc(&d_Y, size_X), "cudaMalloc Y");
    
    check_cuda(cudaMemcpy(d_X, x_data, size_X, cudaMemcpyHostToDevice), "cudaMemcpy X");
    
    size_t mem_for_A = available_mem - 2 * size_X;
    size_t row_size = n * sizeof(double);
    size_t max_rows = mem_for_A / row_size / 2;
    if (max_rows == 0) throw std::runtime_error("Not enough GPU memory");
    
    size_t chunk_rows = (max_rows / 32) * 32;
    if (chunk_rows == 0) chunk_rows = max_rows;
    if (chunk_rows > n) chunk_rows = n;
    
    AsyncStreamer<double> streamer(chunk_rows * n, config_.device_id, config_.enable_async);
    
    cudaStream_t compute_stream;
    check_cuda(cudaStreamCreate(&compute_stream), "cudaStreamCreate");
    check_cublas(cublasSetStream(cublas_handle_, compute_stream), "cublasSetStream");
    
    const double* a_ptr = a_double->data();
    double alpha = 1.0;
    double beta = 0.0;
    
    for (size_t i = 0; i < n; i += chunk_rows) {
        size_t current_rows = std::min(chunk_rows, n - i);
        streamer.wait_for_write_buffer();
        double* h_pinned = streamer.get_host_write_buffer();
        std::copy(a_ptr + i * n, a_ptr + i * n + current_rows * n, h_pinned);
        streamer.submit_transfer(current_rows * n);
        double* d_A_chunk = streamer.get_device_read_buffer(compute_stream);
        
        check_cublas(cublasDgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                 b, current_rows, n, &alpha, d_X, b, d_A_chunk, n, &beta, d_Y + i * b, b), "cublasDgemm");
        streamer.release_device_buffer(compute_stream);
    }
    check_cuda(cudaStreamSynchronize(compute_stream), "Sync Compute");
    
    check_cuda(cudaMemcpy(y_data, d_Y, size_X, cudaMemcpyDeviceToHost), "cudaMemcpy Y");
    
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaStreamDestroy(compute_stream);
}

void CudaDevice::matmul_streaming(const DenseMatrix<double>* a, const DenseMatrix<double>* b, DenseMatrix<double>* c, size_t available_mem) {
    // Tiled Matrix Multiplication C = A * B
    // We divide C into tiles C_ij.
    // C_ij = Sum_k (A_ik * B_kj)
    
    // To minimize I/O, we want to load a block of A and reuse it as much as possible,
    // or load a block of B and reuse it.
    // Standard approach: Blocked GEMM.
    
    // Constraints:
    // We need at least one block of A, one block of B, and one block of C in memory.
    // Ideally, we keep a large block of C in memory and accumulate into it.
    
    uint64_t n = a->size();
    size_t row_size = n * sizeof(double);
    
    // Let's try to compute C in horizontal strips (rows).
    // To compute a strip of C (size R x N), we need:
    // - The corresponding strip of A (size R x N).
    // - The entire matrix B (size N x N).
    // If B fits in memory, great. If not, we need to tile B too.
    
    // If B is too large, we must tile both dimensions.
    // Let's define a tile size T x T.
    // We need 3 * T^2 * sizeof(double) < available_mem.
    // We want T to be as large as possible.
    
    // Max tile size
    // We need space for 2 buffers (A+B) in AsyncStreamer (double buffered = 4 total)
    // Plus 1 buffer for C (on device)
    // Plus 1 buffer for C (on host, pinned)
    
    // AsyncStreamer uses 2 * buffer_size on Host and 2 * buffer_size on Device.
    // We want buffer_size to hold ONE tile of A and ONE tile of B.
    // So buffer_size = 2 * tile_dim^2.
    // Total Device Mem = 2 * (2 * tile_dim^2) + tile_dim^2 (for C) = 5 * tile_dim^2.
    
    size_t max_elements = available_mem / sizeof(double) / 5;
    size_t tile_dim = (size_t)std::sqrt(max_elements);
    
    // Align tile_dim
    tile_dim = (tile_dim / 32) * 32;
    if (tile_dim == 0) tile_dim = 32;
    if (tile_dim > n) tile_dim = n;
    
    size_t tile_elements = tile_dim * tile_dim;
    size_t tile_bytes = tile_elements * sizeof(double);
    
    // C buffer (Accumulator)
    double *d_C;
    check_cuda(cudaMalloc(&d_C, tile_bytes), "cudaMalloc Tile C");
    
    double *h_pinned_C;
    check_cuda(cudaMallocHost(&h_pinned_C, tile_bytes), "cudaMallocHost Tile C");
    
    // Async Streamer for A and B
    // Buffer size = 2 * tile_elements (First half A, Second half B)
    AsyncStreamer<double> streamer(2 * tile_elements, config_.device_id, config_.enable_async);
    
    // Create a compute stream for kernels
    cudaStream_t compute_stream;
    check_cuda(cudaStreamCreate(&compute_stream), "cudaStreamCreate");
    check_cublas(cublasSetStream(cublas_handle_, compute_stream), "cublasSetStream");

    const double* a_ptr = a->data();
    const double* b_ptr = b->data();
    double* c_ptr = c->data();
    
    double alpha = 1.0;
    double beta = 1.0; // Accumulate
    
    // Loop over tiles of C (i, j)
    for (size_t i = 0; i < n; i += tile_dim) {
        size_t h = std::min(tile_dim, n - i); // Height of C tile
        
        for (size_t j = 0; j < n; j += tile_dim) {
            size_t w = std::min(tile_dim, n - j); // Width of C tile
            
            // Initialize C tile to 0 on GPU
            check_cuda(cudaMemsetAsync(d_C, 0, tile_bytes, compute_stream), "cudaMemset C");
            
            // Pipeline Loop
            for (size_t k = 0; k < n; k += tile_dim) {
                size_t d = std::min(tile_dim, n - k); // Depth
                
                // 1. Wait for a free write buffer (CPU sync)
                streamer.wait_for_write_buffer();
                
                // 2. Fill the buffer (CPU)
                double* h_buf = streamer.get_host_write_buffer();
                double* h_A = h_buf;
                double* h_B = h_buf + tile_elements;
                
                // Gather A_ik (h x d)
                ParallelFor(0, h, [&](size_t r) {
                    std::copy(a_ptr + (i + r) * n + k, 
                              a_ptr + (i + r) * n + k + d, 
                              h_A + r * tile_dim); 
                });
                
                // Gather B_kj (d x w)
                ParallelFor(0, d, [&](size_t r) {
                    std::copy(b_ptr + (k + r) * n + j, 
                              b_ptr + (k + r) * n + j + w, 
                              h_B + r * tile_dim);
                });
                
                // 3. Submit Transfer (H2D on transfer stream)
                streamer.submit_transfer(2 * tile_elements);
                
                // 4. Get Device Buffer (Injects wait on compute stream)
                double* d_buf = streamer.get_device_read_buffer(compute_stream);
                double* d_A = d_buf;
                double* d_B = d_buf + tile_elements;
                
                // 5. Compute (GPU on compute stream)
                // cublasDgemm(handle, OP_N, OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
                // A = d_B (w x d), B = d_A (d x h), C = d_C (w x h)
                check_cublas(cublasDgemm(cublas_handle_,
                                         CUBLAS_OP_N, CUBLAS_OP_N,
                                         w, h, d,
                                         &alpha,
                                         d_B, tile_dim,
                                         d_A, tile_dim,
                                         &beta,
                                         d_C, tile_dim), "GEMM Tile");
                                         
                // 6. Release Device Buffer (Injects event record)
                streamer.release_device_buffer(compute_stream);
            }
            
            // Download C_ij (Synchronous for now, or use another stream?)
            // We reuse the compute stream to ensure GEMMs are done.
            check_cuda(cudaMemcpyAsync(h_pinned_C, d_C, tile_bytes, cudaMemcpyDeviceToHost, compute_stream), "Memcpy C");
            check_cuda(cudaStreamSynchronize(compute_stream), "Sync C");
            
            // Scatter C_ij back to global C
            ParallelFor(0, h, [&](size_t r) {
                std::copy(h_pinned_C + r * tile_dim, 
                          h_pinned_C + r * tile_dim + w, 
                          c_ptr + (i + r) * n + j);
            });
        }
    }
    
    // Cleanup
    cudaFree(d_C);
    cudaFreeHost(h_pinned_C);
    cudaStreamDestroy(compute_stream);
    
    c->set_scalar(a->get_scalar() * b->get_scalar());
}

void CudaDevice::matmul_streaming(const DenseMatrix<float>* a, const DenseMatrix<float>* b, DenseMatrix<float>* c, size_t available_mem) {
    size_t n = a->size();
    size_t tile_dim = 1024;
    
    while (5 * tile_dim * tile_dim * sizeof(float) > available_mem * 0.8 && tile_dim > 32) {
        tile_dim /= 2;
    }
    
    size_t tile_elements = tile_dim * tile_dim;
    size_t tile_bytes = tile_elements * sizeof(float);
    
    float *d_C;
    check_cuda(cudaMalloc(&d_C, tile_bytes), "cudaMalloc Tile C float");
    
    float *h_pinned_C;
    check_cuda(cudaMallocHost(&h_pinned_C, tile_bytes), "cudaMallocHost Tile C float");
    
    AsyncStreamer<float> streamer(2 * tile_elements, config_.device_id, config_.enable_async);
    
    cudaStream_t compute_stream;
    check_cuda(cudaStreamCreate(&compute_stream), "cudaStreamCreate");
    check_cublas(cublasSetStream(cublas_handle_, compute_stream), "cublasSetStream");

    const float* a_ptr = a->data();
    const float* b_ptr = b->data();
    float* c_ptr = c->data();
    
    float alpha = 1.0f;
    float beta = 1.0f;
    
    for (size_t i = 0; i < n; i += tile_dim) {
        size_t h = std::min(tile_dim, n - i);
        
        for (size_t j = 0; j < n; j += tile_dim) {
            size_t w = std::min(tile_dim, n - j);
            
            check_cuda(cudaMemsetAsync(d_C, 0, tile_bytes, compute_stream), "cudaMemset C float");
            
            for (size_t k = 0; k < n; k += tile_dim) {
                size_t d = std::min(tile_dim, n - k);
                
                streamer.wait_for_write_buffer();
                
                float* h_buf = streamer.get_host_write_buffer();
                float* h_A = h_buf;
                float* h_B = h_buf + tile_elements;
                
                ParallelFor(0, h, [&](size_t r) {
                    std::copy(a_ptr + (i + r) * n + k, 
                              a_ptr + (i + r) * n + k + d, 
                              h_A + r * tile_dim); 
                });
                
                ParallelFor(0, d, [&](size_t r) {
                    std::copy(b_ptr + (k + r) * n + j, 
                              b_ptr + (k + r) * n + j + w, 
                              h_B + r * tile_dim);
                });
                
                streamer.submit_transfer(2 * tile_elements);
                
                float* d_buf = streamer.get_device_read_buffer(compute_stream);
                float* d_A = d_buf;
                float* d_B = d_buf + tile_elements;
                
                check_cublas(cublasSgemm(cublas_handle_,
                                         CUBLAS_OP_N, CUBLAS_OP_N,
                                         w, h, d,
                                         &alpha,
                                         d_B, tile_dim,
                                         d_A, tile_dim,
                                         &beta,
                                         d_C, tile_dim), "GEMM Tile float");
                                         
                streamer.release_device_buffer(compute_stream);
            }
            
            check_cuda(cudaMemcpyAsync(h_pinned_C, d_C, tile_bytes, cudaMemcpyDeviceToHost, compute_stream), "Memcpy C float");
            check_cuda(cudaStreamSynchronize(compute_stream), "Sync C float");
            
            ParallelFor(0, h, [&](size_t r) {
                std::copy(h_pinned_C + r * tile_dim, 
                          h_pinned_C + r * tile_dim + w, 
                          c_ptr + (i + r) * n + j);
            });
        }
    }
    
    cudaFree(d_C);
    cudaFreeHost(h_pinned_C);
    cudaStreamDestroy(compute_stream);
    
    c->set_scalar(a->get_scalar() * b->get_scalar());
}

void CudaDevice::matmul_streaming(const DenseMatrix<Float16>* a, const DenseMatrix<Float16>* b, DenseMatrix<Float16>* c, size_t available_mem) {
    size_t n = a->size();
    size_t tile_dim = 1024;
    
    while (5 * tile_dim * tile_dim * sizeof(uint16_t) > available_mem * 0.8 && tile_dim > 32) {
        tile_dim /= 2;
    }
    
    size_t tile_elements = tile_dim * tile_dim;
    size_t tile_bytes = tile_elements * sizeof(uint16_t);
    
    __half *d_C;
    check_cuda(cudaMalloc(&d_C, tile_bytes), "cudaMalloc Tile C half");
    
    uint16_t *h_pinned_C;
    check_cuda(cudaMallocHost(&h_pinned_C, tile_bytes), "cudaMallocHost Tile C half");
    
    AsyncStreamer<uint16_t> streamer(2 * tile_elements, config_.device_id, config_.enable_async);
    
    cudaStream_t compute_stream;
    check_cuda(cudaStreamCreate(&compute_stream), "cudaStreamCreate");
    check_cublas(cublasSetStream(cublas_handle_, compute_stream), "cublasSetStream");

    const Float16* a_ptr = a->data();
    const Float16* b_ptr = b->data();
    Float16* c_ptr = c->data();
    
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(1.0f);
    
    for (size_t i = 0; i < n; i += tile_dim) {
        size_t h = std::min(tile_dim, n - i);
        
        for (size_t j = 0; j < n; j += tile_dim) {
            size_t w = std::min(tile_dim, n - j);
            
            check_cuda(cudaMemsetAsync(d_C, 0, tile_bytes, compute_stream), "cudaMemset C half");
            
            for (size_t k = 0; k < n; k += tile_dim) {
                size_t d = std::min(tile_dim, n - k);
                
                streamer.wait_for_write_buffer();
                
                uint16_t* h_buf = streamer.get_host_write_buffer();
                uint16_t* h_A = h_buf;
                uint16_t* h_B = h_buf + tile_elements;
                
                ParallelFor(0, h, [&](size_t r) {
                    const uint16_t* src = (const uint16_t*)(a_ptr + (i + r) * n + k);
                    std::copy(src, src + d, h_A + r * tile_dim); 
                });
                
                ParallelFor(0, d, [&](size_t r) {
                    const uint16_t* src = (const uint16_t*)(b_ptr + (k + r) * n + j);
                    std::copy(src, src + w, h_B + r * tile_dim);
                });
                
                streamer.submit_transfer(2 * tile_elements);
                
                uint16_t* d_buf = streamer.get_device_read_buffer(compute_stream);
                __half* d_A = (__half*)d_buf;
                __half* d_B = (__half*)(d_buf + tile_elements);
                
                check_cublas(cublasHgemm(cublas_handle_,
                                         CUBLAS_OP_N, CUBLAS_OP_N,
                                         w, h, d,
                                         &alpha,
                                         d_B, tile_dim,
                                         d_A, tile_dim,
                                         &beta,
                                         d_C, tile_dim), "GEMM Tile half");
                                         
                streamer.release_device_buffer(compute_stream);
            }
            
            check_cuda(cudaMemcpyAsync(h_pinned_C, d_C, tile_bytes, cudaMemcpyDeviceToHost, compute_stream), "Memcpy C half");
            check_cuda(cudaStreamSynchronize(compute_stream), "Sync C half");
            
            ParallelFor(0, h, [&](size_t r) {
                uint16_t* dst = (uint16_t*)(c_ptr + (i + r) * n + j);
                std::copy(h_pinned_C + r * tile_dim, 
                          h_pinned_C + r * tile_dim + w, 
                          dst);
            });
        }
    }
    
    cudaFree(d_C);
    cudaFreeHost(h_pinned_C);
    cudaStreamDestroy(compute_stream);
    
    c->set_scalar(a->get_scalar() * b->get_scalar());
}

extern "C" {
    __declspec(dllexport) ComputeDevice* create_cuda_device(const AcceleratorConfig* config) {
        return new CudaDevice(*config);
    }
}

void CudaDevice::add(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    CudaSolver solver(this);
    solver.add(a, b, result);
}

void CudaDevice::subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    CudaSolver solver(this);
    solver.subtract(a, b, result);
}

void CudaDevice::multiply_scalar(const MatrixBase& a, double scalar, MatrixBase& result) {
    CudaSolver solver(this);
    solver.multiply_scalar(a, scalar, result);
}

double CudaDevice::dot(const VectorBase& a, const VectorBase& b) {
    throw std::runtime_error("CudaDevice::dot not implemented");
}

void CudaDevice::add_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) {
    throw std::runtime_error("CudaDevice::add_vector not implemented");
}

void CudaDevice::subtract_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) {
    throw std::runtime_error("CudaDevice::subtract_vector not implemented");
}

void CudaDevice::scalar_multiply_vector(const VectorBase& a, double scalar, VectorBase& result) {
    throw std::runtime_error("CudaDevice::scalar_multiply_vector not implemented");
}

void CudaDevice::scalar_add_vector(const VectorBase& a, double scalar, VectorBase& result) {
    throw std::runtime_error("CudaDevice::scalar_add_vector not implemented");
}

} // namespace pycauset

