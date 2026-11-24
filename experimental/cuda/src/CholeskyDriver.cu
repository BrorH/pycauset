// CholeskyDriver
// What: GPU-backed Cholesky factorization driver for dense float matrices.
// Why: Provides a plug-and-play acceleration routine for SPD factorizations.
// Dependencies: CudaDevice (cuSOLVER/cuBLAS) and dense matrix types.

#include "CholeskyDriver.hpp"
#include "CudaDevice.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"

#include <vector>
#include <stdexcept>
#include <algorithm>

namespace pycauset {

namespace {
    void zero_upper_triangle(double* data, uint64_t n) {
        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t j = i + 1; j < n; ++j) {
                data[i * n + j] = 0.0;
            }
        }
    }

    void zero_upper_triangle(float* data, uint64_t n) {
        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t j = i + 1; j < n; ++j) {
                data[i * n + j] = 0.0f;
            }
        }
    }
}

void CholeskyDriver::run(CudaDevice& device, const MatrixBase& in, MatrixBase& out) {
    if (in.rows() != in.cols()) {
        throw std::invalid_argument("CholeskyDriver: matrix must be square");
    }

    const uint64_t n = in.rows();
    size_t free_mem = device.get_available_memory_bytes();

    // Float64 path
    if (auto* in_dense = dynamic_cast<const DenseMatrix<double>*>(&in)) {
        auto* out_dense = dynamic_cast<DenseMatrix<double>*>(&out);
        if (!out_dense) {
            throw std::runtime_error("CholeskyDriver: output must be DenseMatrix<double>");
        }

        const size_t size_bytes = n * n * sizeof(double);
        if (size_bytes * 2 > free_mem) {
            throw std::runtime_error("CholeskyDriver: matrix too large for GPU in-core factorization");
        }

        double* d_A = nullptr;
        int* d_Info = nullptr;
        double* d_Work = nullptr;
        int lwork = 0;

        device.check_cuda_error(cudaMalloc(&d_A, size_bytes), "cudaMalloc A");
        device.check_cuda_error(cudaMemcpy(d_A, in_dense->data(), size_bytes, cudaMemcpyHostToDevice), "cudaMemcpy A");
        device.check_cuda_error(cudaMalloc(&d_Info, sizeof(int)), "cudaMalloc Info");

        device.check_cusolver_error(
            cusolverDnDpotrf_bufferSize(device.get_cusolver_handle(), CUBLAS_FILL_MODE_LOWER, static_cast<int>(n), d_A, static_cast<int>(n), &lwork),
            "potrf_bufferSize");
        device.check_cuda_error(cudaMalloc(&d_Work, static_cast<size_t>(lwork) * sizeof(double)), "cudaMalloc Work");

        device.check_cusolver_error(
            cusolverDnDpotrf(device.get_cusolver_handle(), CUBLAS_FILL_MODE_LOWER, static_cast<int>(n), d_A, static_cast<int>(n), d_Work, lwork, d_Info),
            "potrf");

        int info = 0;
        device.check_cuda_error(cudaMemcpy(&info, d_Info, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy Info");
        if (info != 0) {
            cudaFree(d_A);
            cudaFree(d_Info);
            cudaFree(d_Work);
            throw std::runtime_error("CholeskyDriver: potrf failed (matrix not SPD?)");
        }

        device.check_cuda_error(cudaMemcpy(out_dense->data(), d_A, size_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy Result");
        zero_upper_triangle(out_dense->data(), n);

        cudaFree(d_A);
        cudaFree(d_Info);
        cudaFree(d_Work);
        return;
    }

    // Float32 path
    if (auto* in_dense = dynamic_cast<const DenseMatrix<float>*>(&in)) {
        auto* out_dense = dynamic_cast<DenseMatrix<float>*>(&out);
        if (!out_dense) {
            throw std::runtime_error("CholeskyDriver: output must be DenseMatrix<float>");
        }

        const size_t size_bytes = n * n * sizeof(float);
        if (size_bytes * 2 > free_mem) {
            throw std::runtime_error("CholeskyDriver: matrix too large for GPU in-core factorization");
        }

        float* d_A = nullptr;
        int* d_Info = nullptr;
        float* d_Work = nullptr;
        int lwork = 0;

        device.check_cuda_error(cudaMalloc(&d_A, size_bytes), "cudaMalloc A");
        device.check_cuda_error(cudaMemcpy(d_A, in_dense->data(), size_bytes, cudaMemcpyHostToDevice), "cudaMemcpy A");
        device.check_cuda_error(cudaMalloc(&d_Info, sizeof(int)), "cudaMalloc Info");

        device.check_cusolver_error(
            cusolverDnSpotrf_bufferSize(device.get_cusolver_handle(), CUBLAS_FILL_MODE_LOWER, static_cast<int>(n), d_A, static_cast<int>(n), &lwork),
            "potrf_bufferSize");
        device.check_cuda_error(cudaMalloc(&d_Work, static_cast<size_t>(lwork) * sizeof(float)), "cudaMalloc Work");

        device.check_cusolver_error(
            cusolverDnSpotrf(device.get_cusolver_handle(), CUBLAS_FILL_MODE_LOWER, static_cast<int>(n), d_A, static_cast<int>(n), d_Work, lwork, d_Info),
            "potrf");

        int info = 0;
        device.check_cuda_error(cudaMemcpy(&info, d_Info, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy Info");
        if (info != 0) {
            cudaFree(d_A);
            cudaFree(d_Info);
            cudaFree(d_Work);
            throw std::runtime_error("CholeskyDriver: potrf failed (matrix not SPD?)");
        }

        device.check_cuda_error(cudaMemcpy(out_dense->data(), d_A, size_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy Result");
        zero_upper_triangle(out_dense->data(), n);

        cudaFree(d_A);
        cudaFree(d_Info);
        cudaFree(d_Work);
        return;
    }

    throw std::runtime_error("CholeskyDriver: only DenseMatrix<float/double> supported");
}

} // namespace pycauset
