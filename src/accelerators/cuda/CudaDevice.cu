#include "CudaDevice.hpp"
#include "CudaSolver.hpp"
#include "AsyncStreamer.hpp"
#include "MatmulDriver.hpp"
#include "CholeskyDriver.hpp"
#include "ArnoldiDriver.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/math/Eigen.hpp"
#include "pycauset/core/MemoryGovernor.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include <iostream>
#include <stdexcept>
#include <optional>
#include <sstream>
#include <cctype>
#include <cstdlib>
#include <vector>
#include <algorithm>

namespace {

// In-place transpose of an n x n dense matrix. Used to convert row-major host
// data into the column-major layout cuSOLVER's geev expects, so the solver sees
// A (not A^T) and returns the correct right eigenvectors.
template <typename T>
__global__ void k_transpose_dense(T* A, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n && i < j) {
        T tmp = A[i * n + j];
        A[i * n + j] = A[j * n + i];
        A[j * n + i] = tmp;
    }
}

// Generic (64-bit) geev wrapper. T is the real element type (float/double) and
// CT the matching cuSOLVER complex type (cuComplex/cuDoubleComplex). Eigenvalues
// come back in host_W (size n, complex), and, when want_vr, right eigenvectors
// in host_VR (size n*n, real, with conjugate pairs in adjacent columns like
// LAPACK).
template <typename T, typename CT>
void run_xgeev(pycauset::CudaDevice& dev, int64_t n, cudaDataType dtA, cudaDataType dtW,
               const T* host_A, CT* host_W, T* host_VR, bool want_vr) {
    const cusolverEigMode_t jobvl = CUSOLVER_EIG_MODE_NOVECTOR;
    const cusolverEigMode_t jobvr = want_vr ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

    cusolverDnParams_t params = nullptr;
    dev.check_cusolver_error(cusolverDnCreateParams(&params), "cusolverDnCreateParams");

    const size_t bytes_A = static_cast<size_t>(n) * n * sizeof(T);
    T* d_A = nullptr;
    CT* d_W = nullptr;
    T* d_VR = nullptr;
    int* d_Info = nullptr;
    void* d_work = nullptr;
    void* h_work = nullptr;

    try {
        dev.check_cuda_error(cudaMalloc(&d_A, bytes_A), "cudaMalloc A");
        dev.check_cuda_error(cudaMalloc(&d_W, n * sizeof(CT)), "cudaMalloc W");
        if (want_vr) dev.check_cuda_error(cudaMalloc(&d_VR, bytes_A), "cudaMalloc VR");
        dev.check_cuda_error(cudaMalloc(&d_Info, sizeof(int)), "cudaMalloc Info");
        dev.check_cuda_error(cudaMemcpy(d_A, host_A, bytes_A, cudaMemcpyHostToDevice), "cudaMemcpy A");

        if (want_vr) {
            dim3 block(16, 16);
            dim3 grid(static_cast<unsigned>((n + 15) / 16), static_cast<unsigned>((n + 15) / 16));
            k_transpose_dense<<<grid, block>>>(d_A, static_cast<int>(n));
        }

        size_t ws_dev = 0, ws_host = 0;
        dev.check_cusolver_error(
            cusolverDnXgeev_bufferSize(dev.get_cusolver_handle(), params, jobvl, jobvr, n,
                                       dtA, d_A, n,
                                       dtW, d_W,
                                       dtA, nullptr, n,
                                       dtA, want_vr ? d_VR : nullptr, n,
                                       dtA, &ws_dev, &ws_host),
            "Xgeev_bufferSize");

        dev.check_cuda_error(cudaMalloc(&d_work, ws_dev), "cudaMalloc work");
        if (ws_host > 0) h_work = std::malloc(ws_host);

        dev.check_cusolver_error(
            cusolverDnXgeev(dev.get_cusolver_handle(), params, jobvl, jobvr, n,
                            dtA, d_A, n,
                            dtW, d_W,
                            dtA, nullptr, n,
                            dtA, want_vr ? d_VR : nullptr, n,
                            dtA, d_work, ws_dev, h_work, ws_host, d_Info),
            "Xgeev");

        int info = 0;
        dev.check_cuda_error(cudaMemcpy(&info, d_Info, sizeof(int), cudaMemcpyDeviceToHost), "copy Info");
        if (info != 0) {
            throw std::runtime_error("cusolverDnXgeev failed (info=" + std::to_string(info) + ")");
        }

        dev.check_cuda_error(cudaMemcpy(host_W, d_W, n * sizeof(CT), cudaMemcpyDeviceToHost), "copy W");
        if (want_vr) dev.check_cuda_error(cudaMemcpy(host_VR, d_VR, bytes_A, cudaMemcpyDeviceToHost), "copy VR");
    } catch (...) {
        if (d_A) cudaFree(d_A);
        if (d_W) cudaFree(d_W);
        if (d_VR) cudaFree(d_VR);
        if (d_Info) cudaFree(d_Info);
        if (d_work) cudaFree(d_work);
        if (h_work) std::free(h_work);
        cusolverDnDestroyParams(params);
        throw;
    }

    cudaFree(d_A);
    cudaFree(d_W);
    if (d_VR) cudaFree(d_VR);
    cudaFree(d_Info);
    cudaFree(d_work);
    if (h_work) std::free(h_work);
    cusolverDnDestroyParams(params);
}

} // namespace

namespace pycauset {

std::complex<double> CudaDevice::dot_complex(const VectorBase& a, const VectorBase& b) {
    (void)a;
    (void)b;
    throw std::runtime_error("CUDA dot_complex not implemented");
}

std::complex<double> CudaDevice::sum(const VectorBase& v) {
    (void)v;
    throw std::runtime_error("CudaDevice::sum(VectorBase) not implemented");
}

void CudaDevice::scalar_multiply_vector_complex(const VectorBase& a, std::complex<double> scalar, VectorBase& result) {
    (void)a;
    (void)scalar;
    (void)result;
    throw std::runtime_error("CUDA scalar_multiply_vector_complex not implemented");
}

std::complex<double> CudaDevice::sum(const MatrixBase& m) {
    (void)m;
    throw std::runtime_error("CudaDevice::sum(MatrixBase) not implemented");
}

double CudaDevice::trace(const MatrixBase& m) {
    (void)m;
    throw std::runtime_error("CudaDevice::trace not implemented");
}

double CudaDevice::determinant(const MatrixBase& m) {
    (void)m;
    throw std::runtime_error("CudaDevice::determinant not implemented");
}

void CudaDevice::qr(const MatrixBase& in, MatrixBase& Q, MatrixBase& R) {
    // GPU QR for square dense float/double matrices. Rectangular input throws so
    // AutoSolver falls back to the CPU path (which handles M x N thin QR).
    const uint64_t m = in.rows();
    const uint64_t n = in.cols();
    if (m != n) {
        throw std::runtime_error("CudaDevice::qr only supports square matrices (CPU handles rectangular)");
    }
    const uint64_t N = m;

    // Double precision.
    if (auto* in_d = dynamic_cast<const DenseMatrix<double>*>(&in)) {
        auto* q_out = dynamic_cast<DenseMatrix<double>*>(&Q);
        auto* r_out = dynamic_cast<DenseMatrix<double>*>(&R);
        if (!q_out || !r_out) throw std::runtime_error("QR output type mismatch (double)");

        const size_t bytes = N * N * sizeof(double);
        std::vector<double> h_a(N * N);
        for (uint64_t i = 0; i < N; ++i)
            for (uint64_t j = 0; j < N; ++j)
                h_a[i * N + j] = in_d->get(i, j);

        double* d_A = nullptr;
        double* d_Tau = nullptr;
        int* d_Info = nullptr;
        double* d_Work = nullptr;
        try {
            check_cuda(cudaMalloc(&d_A, bytes), "cudaMalloc A");
            check_cuda(cudaMalloc(&d_Tau, N * sizeof(double)), "cudaMalloc Tau");
            check_cuda(cudaMalloc(&d_Info, sizeof(int)), "cudaMalloc Info");
            check_cuda(cudaMemcpy(d_A, h_a.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy A");

            // Row-major -> column-major (in-place transpose) so geqrf sees A.
            dim3 block(16, 16);
            dim3 grid(static_cast<unsigned>((N + 15) / 16), static_cast<unsigned>((N + 15) / 16));
            k_transpose_dense<<<grid, block>>>(d_A, static_cast<int>(N));

            int lwork = 0;
            check_cusolver(cusolverDnDgeqrf_bufferSize(cusolver_handle_, static_cast<int>(N), static_cast<int>(N),
                                                       d_A, static_cast<int>(N), &lwork), "geqrf_bufferSize");
            check_cuda(cudaMalloc(&d_Work, lwork * sizeof(double)), "cudaMalloc Work");
            check_cusolver(cusolverDnDgeqrf(cusolver_handle_, static_cast<int>(N), static_cast<int>(N),
                                            d_A, static_cast<int>(N), d_Tau, d_Work, lwork, d_Info), "geqrf");

            int info = 0;
            check_cuda(cudaMemcpy(&info, d_Info, sizeof(int), cudaMemcpyDeviceToHost), "copy Info");
            if (info != 0) throw std::runtime_error("QR factorization failed (geqrf)");

            // Extract R (upper triangular) from the factorized matrix BEFORE orgqr
            // overwrites it with Q. d_A is column-major: entry (r,c) = d_A[c*N + r].
            std::vector<double> h_r(N * N);
            check_cuda(cudaMemcpy(h_r.data(), d_A, bytes, cudaMemcpyDeviceToHost), "copy R");
            double* r_ptr = r_out->data();
            std::fill(r_ptr, r_ptr + N * N, 0.0);
            for (uint64_t r = 0; r < N; ++r)
                for (uint64_t c = r; c < N; ++c)
                    r_ptr[r * N + c] = h_r[c * N + r];

            // orgqr reconstructs Q from the Householder reflectors in place.
            check_cusolver(cusolverDnDorgqr_bufferSize(cusolver_handle_, static_cast<int>(N), static_cast<int>(N),
                                                       static_cast<int>(N), d_A, static_cast<int>(N), d_Tau, &lwork), "orgqr_bufferSize");
            if (lwork * sizeof(double) > 0) {
                cudaFree(d_Work);
                d_Work = nullptr;
                check_cuda(cudaMalloc(&d_Work, lwork * sizeof(double)), "cudaMalloc Work (orgqr)");
            }
            check_cusolver(cusolverDnDorgqr(cusolver_handle_, static_cast<int>(N), static_cast<int>(N),
                                            static_cast<int>(N), d_A, static_cast<int>(N), d_Tau, d_Work, lwork, d_Info), "orgqr");
            check_cuda(cudaMemcpy(&info, d_Info, sizeof(int), cudaMemcpyDeviceToHost), "copy Info 2");
            if (info != 0) throw std::runtime_error("QR factorization failed (orgqr)");

            std::vector<double> h_q(N * N);
            check_cuda(cudaMemcpy(h_q.data(), d_A, bytes, cudaMemcpyDeviceToHost), "copy Q");
            double* q_ptr = q_out->data();
            for (uint64_t r = 0; r < N; ++r)
                for (uint64_t c = 0; c < N; ++c)
                    q_ptr[r * N + c] = h_q[c * N + r];

            cudaFree(d_A); cudaFree(d_Tau); cudaFree(d_Info); cudaFree(d_Work);
            return;
        } catch (...) {
            if (d_A) cudaFree(d_A);
            if (d_Tau) cudaFree(d_Tau);
            if (d_Info) cudaFree(d_Info);
            if (d_Work) cudaFree(d_Work);
            throw;
        }
    }

    // Single precision.
    if (auto* in_f = dynamic_cast<const DenseMatrix<float>*>(&in)) {
        auto* q_out = dynamic_cast<DenseMatrix<float>*>(&Q);
        auto* r_out = dynamic_cast<DenseMatrix<float>*>(&R);
        if (!q_out || !r_out) throw std::runtime_error("QR output type mismatch (float)");

        const size_t bytes = N * N * sizeof(float);
        std::vector<float> h_a(N * N);
        for (uint64_t i = 0; i < N; ++i)
            for (uint64_t j = 0; j < N; ++j)
                h_a[i * N + j] = in_f->get(i, j);

        float* d_A = nullptr;
        float* d_Tau = nullptr;
        int* d_Info = nullptr;
        float* d_Work = nullptr;
        try {
            check_cuda(cudaMalloc(&d_A, bytes), "cudaMalloc A");
            check_cuda(cudaMalloc(&d_Tau, N * sizeof(float)), "cudaMalloc Tau");
            check_cuda(cudaMalloc(&d_Info, sizeof(int)), "cudaMalloc Info");
            check_cuda(cudaMemcpy(d_A, h_a.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy A");

            dim3 block(16, 16);
            dim3 grid(static_cast<unsigned>((N + 15) / 16), static_cast<unsigned>((N + 15) / 16));
            k_transpose_dense<<<grid, block>>>(d_A, static_cast<int>(N));

            int lwork = 0;
            check_cusolver(cusolverDnSgeqrf_bufferSize(cusolver_handle_, static_cast<int>(N), static_cast<int>(N),
                                                       d_A, static_cast<int>(N), &lwork), "geqrf_bufferSize");
            check_cuda(cudaMalloc(&d_Work, lwork * sizeof(float)), "cudaMalloc Work");
            check_cusolver(cusolverDnSgeqrf(cusolver_handle_, static_cast<int>(N), static_cast<int>(N),
                                            d_A, static_cast<int>(N), d_Tau, d_Work, lwork, d_Info), "geqrf");

            int info = 0;
            check_cuda(cudaMemcpy(&info, d_Info, sizeof(int), cudaMemcpyDeviceToHost), "copy Info");
            if (info != 0) throw std::runtime_error("QR factorization failed (geqrf)");

            std::vector<float> h_r(N * N);
            check_cuda(cudaMemcpy(h_r.data(), d_A, bytes, cudaMemcpyDeviceToHost), "copy R");
            float* r_ptr = r_out->data();
            std::fill(r_ptr, r_ptr + N * N, 0.0f);
            for (uint64_t r = 0; r < N; ++r)
                for (uint64_t c = r; c < N; ++c)
                    r_ptr[r * N + c] = h_r[c * N + r];

            check_cusolver(cusolverDnSorgqr_bufferSize(cusolver_handle_, static_cast<int>(N), static_cast<int>(N),
                                                       static_cast<int>(N), d_A, static_cast<int>(N), d_Tau, &lwork), "orgqr_bufferSize");
            if (lwork * sizeof(float) > 0) {
                cudaFree(d_Work);
                d_Work = nullptr;
                check_cuda(cudaMalloc(&d_Work, lwork * sizeof(float)), "cudaMalloc Work (orgqr)");
            }
            check_cusolver(cusolverDnSorgqr(cusolver_handle_, static_cast<int>(N), static_cast<int>(N),
                                            static_cast<int>(N), d_A, static_cast<int>(N), d_Tau, d_Work, lwork, d_Info), "orgqr");
            check_cuda(cudaMemcpy(&info, d_Info, sizeof(int), cudaMemcpyDeviceToHost), "copy Info 2");
            if (info != 0) throw std::runtime_error("QR factorization failed (orgqr)");

            std::vector<float> h_q(N * N);
            check_cuda(cudaMemcpy(h_q.data(), d_A, bytes, cudaMemcpyDeviceToHost), "copy Q");
            float* q_ptr = q_out->data();
            for (uint64_t r = 0; r < N; ++r)
                for (uint64_t c = 0; c < N; ++c)
                    q_ptr[r * N + c] = h_q[c * N + r];

            cudaFree(d_A); cudaFree(d_Tau); cudaFree(d_Info); cudaFree(d_Work);
            return;
        } catch (...) {
            if (d_A) cudaFree(d_A);
            if (d_Tau) cudaFree(d_Tau);
            if (d_Info) cudaFree(d_Info);
            if (d_Work) cudaFree(d_Work);
            throw;
        }
    }

    throw std::runtime_error("CudaDevice::qr only implemented for float/double");
}

void CudaDevice::lu(const MatrixBase& in, MatrixBase& P, MatrixBase& L, MatrixBase& U) {
    // GPU LU for square dense float/double matrices. Rectangular input (and any
    // non-dense dtype) throws so AutoSolver falls back to the CPU path, which
    // handles the general M x N case. Square is the causal-set case (relation and
    // adjacency matrices are n x n).
    const uint64_t m = in.rows();
    const uint64_t n = in.cols();
    if (m != n) {
        throw std::runtime_error("CudaDevice::lu only supports square matrices (CPU handles rectangular)");
    }
    const uint64_t N = m;

    // Double precision.
    if (auto* in_d = dynamic_cast<const DenseMatrix<double>*>(&in)) {
        auto* p_out = dynamic_cast<DenseMatrix<double>*>(&P);
        auto* l_out = dynamic_cast<DenseMatrix<double>*>(&L);
        auto* u_out = dynamic_cast<DenseMatrix<double>*>(&U);
        if (!p_out || !l_out || !u_out) throw std::runtime_error("LU output type mismatch (double)");

        const size_t bytes = N * N * sizeof(double);
        // Materialize through get() so scalar/transpose/view metadata is applied.
        std::vector<double> h_a(N * N);
        for (uint64_t i = 0; i < N; ++i)
            for (uint64_t j = 0; j < N; ++j)
                h_a[i * N + j] = in_d->get(i, j);

        double* d_A = nullptr;
        int* d_Ipiv = nullptr;
        int* d_Info = nullptr;
        double* d_Work = nullptr;
        try {
            check_cuda(cudaMalloc(&d_A, bytes), "cudaMalloc A");
            check_cuda(cudaMalloc(&d_Ipiv, N * sizeof(int)), "cudaMalloc Ipiv");
            check_cuda(cudaMalloc(&d_Info, sizeof(int)), "cudaMalloc Info");
            check_cuda(cudaMemcpy(d_A, h_a.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy A");

            // Row-major host data -> column-major layout getrf expects (in-place
            // transpose), so the solver factorizes A rather than A^T.
            dim3 block(16, 16);
            dim3 grid(static_cast<unsigned>((N + 15) / 16), static_cast<unsigned>((N + 15) / 16));
            k_transpose_dense<<<grid, block>>>(d_A, static_cast<int>(N));

            int lwork = 0;
            check_cusolver(cusolverDnDgetrf_bufferSize(cusolver_handle_, static_cast<int>(N), static_cast<int>(N),
                                                       d_A, static_cast<int>(N), &lwork), "getrf_bufferSize");
            check_cuda(cudaMalloc(&d_Work, lwork * sizeof(double)), "cudaMalloc Work");
            check_cusolver(cusolverDnDgetrf(cusolver_handle_, static_cast<int>(N), static_cast<int>(N),
                                            d_A, static_cast<int>(N), d_Work, d_Ipiv, d_Info), "getrf");

            int info = 0;
            check_cuda(cudaMemcpy(&info, d_Info, sizeof(int), cudaMemcpyDeviceToHost), "copy Info");
            if (info < 0) throw std::runtime_error("LU factorization failed: illegal value");
            // info > 0 => singular; the factorization is still returned, matching CPU.

            std::vector<double> h_lu(N * N);
            std::vector<int> h_ipiv(N);
            check_cuda(cudaMemcpy(h_lu.data(), d_A, bytes, cudaMemcpyDeviceToHost), "copy LU");
            check_cuda(cudaMemcpy(h_ipiv.data(), d_Ipiv, N * sizeof(int), cudaMemcpyDeviceToHost), "copy Ipiv");

            // Extract L (unit lower) and U (upper). h_lu is column-major, so the
            // (r,c) entry lives at h_lu[c*N + r]; the output matrices are row-major.
            double* l_ptr = l_out->data();
            double* u_ptr = u_out->data();
            std::fill(l_ptr, l_ptr + N * N, 0.0);
            std::fill(u_ptr, u_ptr + N * N, 0.0);
            for (uint64_t r = 0; r < N; ++r) {
                for (uint64_t c = 0; c < N; ++c) {
                    if (r > c) l_ptr[r * N + c] = h_lu[c * N + r];
                    else if (r == c) l_ptr[r * N + c] = 1.0;
                    if (r <= c) u_ptr[r * N + c] = h_lu[c * N + r];
                }
            }

            // Build P from the 1-based pivot array (row-swap semantics, same as CPU).
            double* p_ptr = p_out->data();
            std::fill(p_ptr, p_ptr + N * N, 0.0);
            std::vector<int> p_idx(N);
            for (uint64_t i = 0; i < N; ++i) p_idx[i] = static_cast<int>(i);
            for (uint64_t i = 0; i < N; ++i) std::swap(p_idx[i], p_idx[h_ipiv[i] - 1]);
            for (uint64_t i = 0; i < N; ++i) p_ptr[static_cast<uint64_t>(p_idx[i]) * N + i] = 1.0;

            cudaFree(d_A); cudaFree(d_Ipiv); cudaFree(d_Info); cudaFree(d_Work);
            return;
        } catch (...) {
            if (d_A) cudaFree(d_A);
            if (d_Ipiv) cudaFree(d_Ipiv);
            if (d_Info) cudaFree(d_Info);
            if (d_Work) cudaFree(d_Work);
            throw;
        }
    }

    // Single precision.
    if (auto* in_f = dynamic_cast<const DenseMatrix<float>*>(&in)) {
        auto* p_out = dynamic_cast<DenseMatrix<float>*>(&P);
        auto* l_out = dynamic_cast<DenseMatrix<float>*>(&L);
        auto* u_out = dynamic_cast<DenseMatrix<float>*>(&U);
        if (!p_out || !l_out || !u_out) throw std::runtime_error("LU output type mismatch (float)");

        const size_t bytes = N * N * sizeof(float);
        std::vector<float> h_a(N * N);
        for (uint64_t i = 0; i < N; ++i)
            for (uint64_t j = 0; j < N; ++j)
                h_a[i * N + j] = in_f->get(i, j);

        float* d_A = nullptr;
        int* d_Ipiv = nullptr;
        int* d_Info = nullptr;
        float* d_Work = nullptr;
        try {
            check_cuda(cudaMalloc(&d_A, bytes), "cudaMalloc A");
            check_cuda(cudaMalloc(&d_Ipiv, N * sizeof(int)), "cudaMalloc Ipiv");
            check_cuda(cudaMalloc(&d_Info, sizeof(int)), "cudaMalloc Info");
            check_cuda(cudaMemcpy(d_A, h_a.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy A");

            dim3 block(16, 16);
            dim3 grid(static_cast<unsigned>((N + 15) / 16), static_cast<unsigned>((N + 15) / 16));
            k_transpose_dense<<<grid, block>>>(d_A, static_cast<int>(N));

            int lwork = 0;
            check_cusolver(cusolverDnSgetrf_bufferSize(cusolver_handle_, static_cast<int>(N), static_cast<int>(N),
                                                       d_A, static_cast<int>(N), &lwork), "getrf_bufferSize");
            check_cuda(cudaMalloc(&d_Work, lwork * sizeof(float)), "cudaMalloc Work");
            check_cusolver(cusolverDnSgetrf(cusolver_handle_, static_cast<int>(N), static_cast<int>(N),
                                            d_A, static_cast<int>(N), d_Work, d_Ipiv, d_Info), "getrf");

            int info = 0;
            check_cuda(cudaMemcpy(&info, d_Info, sizeof(int), cudaMemcpyDeviceToHost), "copy Info");
            if (info < 0) throw std::runtime_error("LU factorization failed: illegal value");

            std::vector<float> h_lu(N * N);
            std::vector<int> h_ipiv(N);
            check_cuda(cudaMemcpy(h_lu.data(), d_A, bytes, cudaMemcpyDeviceToHost), "copy LU");
            check_cuda(cudaMemcpy(h_ipiv.data(), d_Ipiv, N * sizeof(int), cudaMemcpyDeviceToHost), "copy Ipiv");

            float* l_ptr = l_out->data();
            float* u_ptr = u_out->data();
            std::fill(l_ptr, l_ptr + N * N, 0.0f);
            std::fill(u_ptr, u_ptr + N * N, 0.0f);
            for (uint64_t r = 0; r < N; ++r) {
                for (uint64_t c = 0; c < N; ++c) {
                    if (r > c) l_ptr[r * N + c] = h_lu[c * N + r];
                    else if (r == c) l_ptr[r * N + c] = 1.0f;
                    if (r <= c) u_ptr[r * N + c] = h_lu[c * N + r];
                }
            }

            float* p_ptr = p_out->data();
            std::fill(p_ptr, p_ptr + N * N, 0.0f);
            std::vector<int> p_idx(N);
            for (uint64_t i = 0; i < N; ++i) p_idx[i] = static_cast<int>(i);
            for (uint64_t i = 0; i < N; ++i) std::swap(p_idx[i], p_idx[h_ipiv[i] - 1]);
            for (uint64_t i = 0; i < N; ++i) p_ptr[static_cast<uint64_t>(p_idx[i]) * N + i] = 1.0f;

            cudaFree(d_A); cudaFree(d_Ipiv); cudaFree(d_Info); cudaFree(d_Work);
            return;
        } catch (...) {
            if (d_A) cudaFree(d_A);
            if (d_Ipiv) cudaFree(d_Ipiv);
            if (d_Info) cudaFree(d_Info);
            if (d_Work) cudaFree(d_Work);
            throw;
        }
    }

    throw std::runtime_error("CudaDevice::lu only implemented for float/double");
}

void CudaDevice::svd(const MatrixBase& in, MatrixBase& U, VectorBase& S, MatrixBase& VT) {
    (void)in; (void)U; (void)S; (void)VT;
    throw std::runtime_error("CudaDevice::svd not implemented (use CPU)");
}

void CudaDevice::solve(const MatrixBase& A, const MatrixBase& B, MatrixBase& X) {
    (void)A; (void)B; (void)X;
    throw std::runtime_error("CudaDevice::solve not implemented (use CPU)");
}

} // namespace pycauset
#include <vector>
#include <algorithm>
#include <chrono>

namespace pycauset {

namespace {

std::string trim_copy(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

void strip_quotes(std::string& s) {
    if (s.size() >= 2) {
        char first = s.front();
        char last = s.back();
        if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
            s = s.substr(1, s.size() - 2);
        }
    }
}

std::optional<HardwareProfile> parse_mock_profile_env() {
    const char* env = std::getenv("PYCAUSET_TEST_CUDA_PROFILE");
    if (!env || !*env) {
        return std::nullopt;
    }

    HardwareProfile profile;
    profile.version = 1;
    profile.device_id = 0;
    profile.device_name = "Mock CUDA Device";

    std::string raw(env);
    for (char& ch : raw) {
        if (ch == ';') {
            ch = ',';
        }
    }

    std::stringstream ss(raw);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim_copy(token);
        if (token.empty()) continue;
        auto eq = token.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim_copy(token.substr(0, eq));
        std::string value = trim_copy(token.substr(eq + 1));
        strip_quotes(value);
        if (key == "version") {
            try { profile.version = std::stoi(value); } catch (...) {}
        } else if (key == "device_id") {
            try { profile.device_id = std::stoi(value); } catch (...) {}
        } else if (key == "device_name") {
            if (!value.empty()) profile.device_name = value;
        } else if (key == "cc_major") {
            try { profile.cc_major = std::stoi(value); } catch (...) {}
        } else if (key == "cc_minor") {
            try { profile.cc_minor = std::stoi(value); } catch (...) {}
        } else if (key == "pci_bandwidth_gbps") {
            try { profile.pci_bandwidth_gbps = std::stod(value); } catch (...) {}
        } else if (key == "sgemm_gflops") {
            try { profile.sgemm_gflops = std::stod(value); } catch (...) {}
        } else if (key == "dgemm_gflops") {
            try { profile.dgemm_gflops = std::stod(value); } catch (...) {}
        } else if (key == "timestamp_unix") {
            try { profile.timestamp_unix = static_cast<uint64_t>(std::stoull(value)); } catch (...) {}
        }
    }

    return profile;
}

} // namespace

// --- Discovery API ---

int CudaDevice::get_device_count() {
    if (parse_mock_profile_env().has_value()) {
        return 1;
    }
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaSuccess) {
        return count;
    } else {
        // If we can't get the count, assume 0 (no driver, or no gpu).
        // Clear error to avoid polluting subsequent calls?
        cudaGetLastError(); 
        return 0;
    }
}

std::string CudaDevice::get_device_name(int device_id) {
    if (auto mock = parse_mock_profile_env()) {
        (void)device_id;
        return mock->device_name;
    }
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err == cudaSuccess) {
        return std::string(prop.name);
    } else {
        return "Unknown Device (Error: " + std::string(cudaGetErrorString(err)) + ")"; 
    }
}

bool CudaDevice::is_available() {
    if (parse_mock_profile_env().has_value()) {
        return true;
    }
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

bool CudaDevice::fill_hardware_profile(HardwareProfile& profile, bool run_benchmarks) {
    if (auto mock = parse_mock_profile_env()) {
        profile = *mock;
        if (run_benchmarks) {
            // If the mock does not include benchmark values, leave them as-is.
            // Tests can supply sgemm/dgemm/pci_bandwidth_gbps in the env string.
        }
        return true;
    }
    cudaError_t err = cudaSetDevice(config_.device_id);
    if (err != cudaSuccess) return false;

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, config_.device_id);
    if (err != cudaSuccess) return false;

    profile.version = 1;
    profile.device_id = config_.device_id;
    profile.device_name = std::string(prop.name);
    profile.cc_major = prop.major;
    profile.cc_minor = prop.minor;

    if (run_benchmarks) {
        profile.pci_bandwidth_gbps = benchmark_pci_bandwidth_gbps();
        profile.sgemm_gflops = benchmark_gemm_gflops(true);
        profile.dgemm_gflops = benchmark_gemm_gflops(false);
    }

    return true;
}

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
    if (!core::MemoryGovernor::instance().try_pin_memory(size)) {
        return nullptr;
    }

    void* ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        core::MemoryGovernor::instance().unpin_memory(size);
        return nullptr;
    }

    {
        std::lock_guard<std::mutex> lock(pinned_allocations_mutex_);
        pinned_allocations_[ptr] = size;
    }
    return ptr;
}

void CudaDevice::free_pinned(void* ptr) {
    if (!ptr) return;

    cudaFreeHost(ptr);

    size_t size = 0;
    {
        std::lock_guard<std::mutex> lock(pinned_allocations_mutex_);
        auto it = pinned_allocations_.find(ptr);
        if (it != pinned_allocations_.end()) {
            size = it->second;
            pinned_allocations_.erase(it);
        }
    }

    if (size > 0) {
        core::MemoryGovernor::instance().unpin_memory(size);
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

double CudaDevice::benchmark_pci_bandwidth_gbps() {
    size_t free_byte = 0;
    size_t total_byte = 0;
    if (cudaMemGetInfo(&free_byte, &total_byte) != cudaSuccess) {
        cudaGetLastError();
        return 0.0;
    }

    size_t bytes = 256ULL * 1024 * 1024;
    if (free_byte / 4 < bytes) {
        bytes = free_byte / 4;
    }
    if (bytes < 8ULL * 1024 * 1024) {
        return 0.0;
    }

    if (!core::MemoryGovernor::instance().try_pin_memory(bytes)) {
        return 0.0;
    }

    void* h_ptr = nullptr;
    void* d_ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&h_ptr, bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        core::MemoryGovernor::instance().unpin_memory(bytes);
        return 0.0;
    }
    err = cudaMalloc(&d_ptr, bytes);
    if (err != cudaSuccess) {
        cudaFreeHost(h_ptr);
        core::MemoryGovernor::instance().unpin_memory(bytes);
        return 0.0;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double total_gbps = 0.0;
    const int iters = 2;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start, 0);
        cudaMemcpyAsync(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice, 0);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        if (ms > 0.0f) {
            total_gbps += (static_cast<double>(bytes) / 1e9) / (ms / 1000.0);
        }

        cudaEventRecord(start, 0);
        cudaMemcpyAsync(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost, 0);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        if (ms > 0.0f) {
            total_gbps += (static_cast<double>(bytes) / 1e9) / (ms / 1000.0);
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_ptr);
    cudaFreeHost(h_ptr);
    core::MemoryGovernor::instance().unpin_memory(bytes);

    const int samples = iters * 2;
    return samples > 0 ? (total_gbps / samples) : 0.0;
}

double CudaDevice::benchmark_gemm_gflops(bool use_float32) {
    size_t free_byte = 0;
    size_t total_byte = 0;
    if (cudaMemGetInfo(&free_byte, &total_byte) != cudaSuccess) {
        cudaGetLastError();
        return 0.0;
    }

    int n = use_float32 ? 2048 : 1024;
    size_t elem_size = use_float32 ? sizeof(float) : sizeof(double);
    while (n > 256) {
        size_t required = static_cast<size_t>(n) * n * elem_size * 3;
        if (required < free_byte / 2) break;
        n /= 2;
    }
    if (n < 256) return 0.0;

    void* d_A = nullptr;
    void* d_B = nullptr;
    void* d_C = nullptr;
    size_t bytes = static_cast<size_t>(n) * n * elem_size;
    if (cudaMalloc(&d_A, bytes) != cudaSuccess) return 0.0;
    if (cudaMalloc(&d_B, bytes) != cudaSuccess) { cudaFree(d_A); return 0.0; }
    if (cudaMalloc(&d_C, bytes) != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); return 0.0; }

    cudaMemset(d_A, 0, bytes);
    cudaMemset(d_B, 0, bytes);
    cudaMemset(d_C, 0, bytes);

    const int iters = 3;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (use_float32) {
        float alpha = 1.0f;
        float beta = 0.0f;
        if (cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                        static_cast<float*>(d_A), n, static_cast<float*>(d_B), n, &beta,
                        static_cast<float*>(d_C), n) != CUBLAS_STATUS_SUCCESS) {
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            return 0.0;
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start, 0);
        for (int i = 0; i < iters; ++i) {
            if (cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                            static_cast<float*>(d_A), n, static_cast<float*>(d_B), n, &beta,
                            static_cast<float*>(d_C), n) != CUBLAS_STATUS_SUCCESS) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                cudaFree(d_A);
                cudaFree(d_B);
                cudaFree(d_C);
                return 0.0;
            }
        }
        cudaEventRecord(stop, 0);
    } else {
        double alpha = 1.0;
        double beta = 0.0;
        if (cublasDgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                        static_cast<double*>(d_A), n, static_cast<double*>(d_B), n, &beta,
                        static_cast<double*>(d_C), n) != CUBLAS_STATUS_SUCCESS) {
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            return 0.0;
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start, 0);
        for (int i = 0; i < iters; ++i) {
            if (cublasDgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                            static_cast<double*>(d_A), n, static_cast<double*>(d_B), n, &beta,
                            static_cast<double*>(d_C), n) != CUBLAS_STATUS_SUCCESS) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                cudaFree(d_A);
                cudaFree(d_B);
                cudaFree(d_C);
                return 0.0;
            }
        }
        cudaEventRecord(stop, 0);
    }

    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    if (ms <= 0.0f) return 0.0;
    double seconds = (ms / 1000.0) / iters;
    double ops = 2.0 * static_cast<double>(n) * n * n;
    return (ops / seconds) / 1e9;
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

    // NOTE: a.size() is the element count (n*n), not the dimension. Use rows().
    uint64_t n = a.rows();
    // The GPU matmul buffers are sized n*n and the gemm is issued square, so
    // non-square inputs are routed back to the CPU via AutoSolver's fallback.
    if (a.cols() != n || b.rows() != n || b.cols() != n ||
        result.rows() != n || result.cols() != n) {
        throw std::invalid_argument("CudaDevice::matmul requires square matrices");
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
                  MatmulDriver::run(*this, *a_float, *b_float, *c_float, free_mem);
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



    // Try Double
    auto* a_dense = dynamic_cast<const DenseMatrix<double>*>(&a);
    auto* b_dense = dynamic_cast<const DenseMatrix<double>*>(&b);
    auto* c_dense = dynamic_cast<DenseMatrix<double>*>(&result);

    if (!a_dense || !b_dense || !c_dense) {
        throw std::runtime_error("CudaDevice::matmul only supports DenseMatrix<double> or <float>");
    }

    size_t size_bytes = n * n * sizeof(double);
    
    if (n * n > buffer_size_) {
           if (3 * size_bytes > free_mem + (buffer_size_ * sizeof(double) * 3)) {
               MatmulDriver::run(*this, *a_dense, *b_dense, *c_dense, free_mem);
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

void CudaDevice::cholesky(const MatrixBase& in, MatrixBase& out) {
    CholeskyDriver::run(*this, in, out);
}

void CudaDevice::inverse_incore(const MatrixBase& in, MatrixBase& out) {
    try {
        // Support for Double Precision
        if (auto* in_dense = dynamic_cast<const DenseMatrix<double>*>(&in)) {
            auto* out_dense = dynamic_cast<DenseMatrix<double>*>(&out);
            if (!out_dense) throw std::runtime_error("CudaDevice::inverse output must match input type (double)");

            uint64_t n = in.rows();  // dimension, not element count (in.size() == n*n)
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

            uint64_t n = in.rows();  // dimension, not element count (in.size() == n*n)
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
                std::copy(res_dense->data(), res_dense->data() + in.size(), out_dense->data());
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
                std::copy(res_dense->data(), res_dense->data() + in.size(), out_dense->data());
                out_dense->set_scalar(res_dense->get_scalar());
                return;
            }
        }
        throw; // Re-throw if fallback fails
    }

    throw std::runtime_error("CudaDevice::inverse only supports DenseMatrix<double> or DenseMatrix<float>");
}

void CudaDevice::batch_gemv(const MatrixBase& A, const double* x_data, double* y_data, size_t b) {
    auto* a_double = dynamic_cast<const DenseMatrix<double>*>(&A);
    auto* a_float = dynamic_cast<const DenseMatrix<float>*>(&A);
    
    if (!a_double && !a_float) throw std::runtime_error("CudaDevice::batch_gemv only supports DenseMatrix<double> or DenseMatrix<float>");
    
    uint64_t n = A.rows();  // dimension, not element count (A.size() == n*n)
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

void CudaDevice::eigvals_arnoldi(const MatrixBase& a, VectorBase& out, int k, int m, double tol) {
    ArnoldiDriver::run(*this, a, out, k, m, tol);
}

void CudaDevice::eigvals_skew(const MatrixBase& a, VectorBase& out, int k) {
    (void)a; (void)out; (void)k;
    throw std::runtime_error("CudaDevice::eigvals_skew not implemented (use CPU or wait for update)");
}

void CudaDevice::eig_skew(const MatrixBase& a, VectorBase& eigenvalues, MatrixBase& eigenvectors, int k) {
    (void)a; (void)eigenvalues; (void)eigenvectors; (void)k;
    throw std::runtime_error("CudaDevice::eig_skew not implemented (use CPU)");
}

void CudaDevice::eigh(const MatrixBase& in, VectorBase& eigenvalues, MatrixBase& eigenvectors, char uplo) {
    throw std::runtime_error("CudaDevice::eigh not implemented (use CPU or wait for update)");
}

void CudaDevice::eigvalsh(const MatrixBase& in, VectorBase& eigenvalues, char uplo) {
    throw std::runtime_error("CudaDevice::eigvalsh not implemented (use CPU or wait for update)");
}

void CudaDevice::eig(const MatrixBase& in, VectorBase& eigenvalues, MatrixBase& eigenvectors) {
    if (in.rows() != in.cols()) throw std::invalid_argument("eig requires square matrix");
    const uint64_t n = in.rows();
    if (eigenvectors.rows() != n || eigenvectors.cols() != n) throw std::invalid_argument("Eigenvectors matrix bad shape");
    if (eigenvalues.size() != n) throw std::invalid_argument("Eigenvalues vector bad size");

    const int64_t nn = static_cast<int64_t>(n);

    // Helper to unpack the (real) right-eigenvector buffer VR from geev into the
    // complex output, reconstructing conjugate pairs exactly like the CPU path.
    auto unpack_eigenvectors = [&](const auto& wi, const auto* vr, MatrixBase& out) {
        if (auto* out_cd = dynamic_cast<DenseMatrix<std::complex<double>>*>(&out)) {
            std::complex<double>* out_ptr = out_cd->data();
            const bool contiguous = (out_cd->row_offset() == 0 && out_cd->col_offset() == 0 && out_cd->base_cols() == n);
            for (size_t j = 0; j < n; ++j) {
                if (wi[j] == 0.0) {
                    for (size_t i = 0; i < n; ++i) {
                        std::complex<double> val = { static_cast<double>(vr[j * n + i]), 0.0 };
                        if (contiguous) out_ptr[i * n + j] = val; else out_cd->set(i, j, val);
                    }
                } else {
                    for (size_t i = 0; i < n; ++i) {
                        double re = static_cast<double>(vr[j * n + i]);
                        double im = static_cast<double>(vr[(j + 1) * n + i]);
                        if (contiguous) { out_ptr[i * n + j] = {re, im}; out_ptr[i * n + (j + 1)] = {re, -im}; }
                        else { out_cd->set(i, j, {re, im}); out_cd->set(i, j + 1, {re, -im}); }
                    }
                    ++j;
                }
            }
        } else if (auto* out_cf = dynamic_cast<DenseMatrix<std::complex<float>>*>(&out)) {
            std::complex<float>* out_ptr = out_cf->data();
            const bool contiguous = (out_cf->row_offset() == 0 && out_cf->col_offset() == 0 && out_cf->base_cols() == n);
            for (size_t j = 0; j < n; ++j) {
                if (wi[j] == 0.0f) {
                    for (size_t i = 0; i < n; ++i) {
                        std::complex<float> val = { static_cast<float>(vr[j * n + i]), 0.0f };
                        if (contiguous) out_ptr[i * n + j] = val; else out_cf->set(i, j, val);
                    }
                } else {
                    for (size_t i = 0; i < n; ++i) {
                        float re = static_cast<float>(vr[j * n + i]);
                        float im = static_cast<float>(vr[(j + 1) * n + i]);
                        if (contiguous) { out_ptr[i * n + j] = {re, im}; out_ptr[i * n + (j + 1)] = {re, -im}; }
                        else { out_cf->set(i, j, {re, im}); out_cf->set(i, j + 1, {re, -im}); }
                    }
                    ++j;
                }
            }
        } else {
            throw std::runtime_error("eig requires complex matrix output");
        }
    };

    // Double precision
    if (auto* in_d = dynamic_cast<const DenseMatrix<double>*>(&in)) {
        std::vector<cuDoubleComplex> h_W(n);
        std::vector<double> h_WI(n), h_VR(n * n);
        run_xgeev<double, cuDoubleComplex>(*this, nn, CUDA_R_64F, CUDA_C_64F, in_d->data(), h_W.data(), h_VR.data(), true);

        if (auto* d_dst = dynamic_cast<DenseVector<std::complex<double>>*>(&eigenvalues)) {
            std::complex<double>* ptr = d_dst->data();
            for (size_t i = 0; i < n; ++i) { ptr[i] = { h_W[i].x, h_W[i].y }; h_WI[i] = h_W[i].y; }
        } else {
            throw std::runtime_error("eig requires complex vector output");
        }
        unpack_eigenvectors(h_WI, h_VR.data(), eigenvectors);
        return;
    }

    // Single precision
    if (auto* in_f = dynamic_cast<const DenseMatrix<float>*>(&in)) {
        std::vector<cuComplex> h_W(n);
        std::vector<float> h_WI(n), h_VR(n * n);
        run_xgeev<float, cuComplex>(*this, nn, CUDA_R_32F, CUDA_C_32F, in_f->data(), h_W.data(), h_VR.data(), true);

        if (auto* f_dst = dynamic_cast<DenseVector<std::complex<float>>*>(&eigenvalues)) {
            std::complex<float>* ptr = f_dst->data();
            for (size_t i = 0; i < n; ++i) { ptr[i] = { h_W[i].x, h_W[i].y }; h_WI[i] = h_W[i].y; }
        } else {
            throw std::runtime_error("eig requires complex vector output");
        }
        unpack_eigenvectors(h_WI, h_VR.data(), eigenvectors);
        return;
    }

    throw std::runtime_error("eig not implemented for these types");
}

void CudaDevice::eigvals(const MatrixBase& in, VectorBase& eigenvalues) {
    if (in.rows() != in.cols()) throw std::invalid_argument("eigvals requires square matrix");
    const uint64_t n = in.rows();
    if (eigenvalues.size() != n) throw std::invalid_argument("Eigenvalues vector bad size");

    const int64_t nn = static_cast<int64_t>(n);

    // Double precision
    if (auto* in_d = dynamic_cast<const DenseMatrix<double>*>(&in)) {
        std::vector<cuDoubleComplex> h_W(n);
        run_xgeev<double, cuDoubleComplex>(*this, nn, CUDA_R_64F, CUDA_C_64F, in_d->data(), h_W.data(), nullptr, false);

        if (auto* d_dst = dynamic_cast<DenseVector<std::complex<double>>*>(&eigenvalues)) {
            std::complex<double>* ptr = d_dst->data();
            for (size_t i = 0; i < n; ++i) ptr[i] = { h_W[i].x, h_W[i].y };
        } else {
            throw std::runtime_error("eigvals requires complex vector output");
        }
        return;
    }

    // Single precision
    if (auto* in_f = dynamic_cast<const DenseMatrix<float>*>(&in)) {
        std::vector<cuComplex> h_W(n);
        run_xgeev<float, cuComplex>(*this, nn, CUDA_R_32F, CUDA_C_32F, in_f->data(), h_W.data(), nullptr, false);

        if (auto* f_dst = dynamic_cast<DenseVector<std::complex<float>>*>(&eigenvalues)) {
            std::complex<float>* ptr = f_dst->data();
            for (size_t i = 0; i < n; ++i) ptr[i] = { h_W[i].x, h_W[i].y };
        } else {
            throw std::runtime_error("eigvals requires complex vector output");
        }
        return;
    }

    throw std::runtime_error("eigvals not implemented for these types");
}

void CudaDevice::matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, VectorBase& result) {
    throw std::runtime_error("CudaDevice::matrix_vector_multiply not implemented");
}

void CudaDevice::vector_matrix_multiply(const VectorBase& v, const MatrixBase& m, VectorBase& result) {
    throw std::runtime_error("CudaDevice::vector_matrix_multiply not implemented");
}

void CudaDevice::outer_product(const VectorBase& a, const VectorBase& b, MatrixBase& result) {
    throw std::runtime_error("CudaDevice::outer_product not implemented");
}

void CudaDevice::elementwise_multiply(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    throw std::runtime_error("CudaDevice::elementwise_multiply not implemented");
}

void CudaDevice::elementwise_divide(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    throw std::runtime_error("CudaDevice::elementwise_divide not implemented");
}

void CudaDevice::batch_gemv_streaming(const MatrixBase& A, const double* x_data, double* y_data, size_t b, size_t available_mem) {
    auto* a_double = dynamic_cast<const DenseMatrix<double>*>(&A);
    auto* a_float = dynamic_cast<const DenseMatrix<float>*>(&A);
    
    uint64_t n = A.rows();  // dimension, not element count (A.size() == n*n)
    
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

double CudaDevice::l2_norm(const VectorBase& v) {
    throw std::runtime_error("CudaDevice::l2_norm not implemented");
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

void CudaDevice::cross_product(const VectorBase& a, const VectorBase& b, VectorBase& result) {
    (void)a;
    (void)b;
    (void)result;
    throw std::runtime_error("CudaDevice::cross_product not implemented");
}

std::unique_ptr<TriangularMatrix<double>> CudaDevice::compute_k_matrix(
    const TriangularMatrix<bool>& C,
    double a,
    const std::string& output_path,
    int num_threads
) {
    (void)C;
    (void)a;
    (void)output_path;
    (void)num_threads;
    throw std::runtime_error("CudaDevice::compute_k_matrix not implemented");
}

double CudaDevice::frobenius_norm(const MatrixBase& m) {
    throw std::runtime_error("CudaDevice::frobenius_norm not implemented");
}

} // namespace pycauset

