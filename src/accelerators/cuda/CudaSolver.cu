#include "CudaSolver.hpp"
#include "pycauset/core/StorageUtils.hpp"
#include "pycauset/math/Eigen.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/matrix/SymmetricMatrix.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuComplex.h>

using namespace pycauset;

namespace {
    template <typename T>
    __global__ void k_add(const T* a, const T* b, T* c, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
        
        uint32_t x = row_bits;
        // Swap 16x16 blocks
        // mask = 0x0000FFFF; shift = 16;
        // if (tid & 16) x = (x << 16) | (other >> 16) ...
        // Actually, standard algorithm:
        // 1. Swap 16x16 quadrants.
        //    Target: thread i (0..15) bit j (16..31) <-> thread i+16 bit j-16
        //    Use shfl_xor(16).
        
        uint32_t y = __shfl_xor_sync(0xFFFFFFFF, x, 16);
        // Threads 0..15: x has rows 0..15. y has rows 16..31.
        // We want bits 0..15 from x, and bits 0..15 from y (shifted).
        // Wait, this is getting complicated.
        // Let's use the O(32) ballot loop, it's simple and robust. 32 iterations is fine.
        
        for (int k = 0; k < 32; ++k) {
            uint32_t c = __ballot_sync(0xFFFFFFFF, (row_bits >> k) & 1);
            if (tid == k) col_bits = c;
        }
        
        // Write to dst
        // We are writing row 'tile_x + tid' of dst.
        // This corresponds to column 'tile_x + tid' of src.
        // We have 32 bits: rows 'tile_y' to 'tile_y + 31'.
        // These go into dst[dst_row * stride_words + word_idx]
        
        size_t dst_row = tile_x + tid;
        if (dst_row < n) {
            size_t dst_word_idx = tile_y / 64;
            size_t dst_bit_offset = tile_y % 64; // 0 or 32
            
            if (tile_y < n) {
                // We need to write 'col_bits' into the 64-bit word at dst_word_idx.
                // It goes into bits [dst_bit_offset, dst_bit_offset + 31].
                // Since multiple blocks might write to the same word (if tile_y=0 and tile_y=32),
                // we need atomic operations OR we ensure we write full words?
                // No, we process tiles.
                // If we use atomicOr, it's safe.
                
                unsigned long long* addr = (unsigned long long*)&dst[dst_row * stride_words + dst_word_idx];
                unsigned long long val = (unsigned long long)col_bits << dst_bit_offset;
                
                // We need to clear the bits first?
                // Assuming dst is initialized to 0.
                atomicOr(addr, val);
            }
        }
    }

    // BitMatrix Multiplication Kernel
    // C[i, j] = popcount(A[i] & B_T[j])
    // A, B_T are row-major packed.
    // C is int32.
    __global__ void k_matmul_bits(const uint64_t* A, const uint64_t* B_T, int32_t* C, size_t n, size_t stride_words) {
        size_t i = blockIdx.y * blockDim.y + threadIdx.y;
        size_t j = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (i < n && j < n) {
            int32_t sum = 0;
            const uint64_t* A_row = A + i * stride_words;
            const uint64_t* B_row = B_T + j * stride_words;
            
            for (size_t k = 0; k < stride_words; ++k) {
                sum += __popcll(A_row[k] & B_row[k]);
            }
            
            C[i * n + j] = sum;
        }
    }
}


// Helper for parallel for loop (simple OpenMP-like wrapper)
template<typename Func>
void ParallelFor(size_t start, size_t end, Func f) {
    // For now, serial execution to avoid dependency issues
    for (size_t i = start; i < end; ++i) {
        f(i);
    }
}

CudaSolver::CudaSolver(CudaDevice* device) : device_(device) {}

void CudaSolver::invert(const MatrixBase& in, MatrixBase& out) {
    size_t n = in.size();
    size_t element_size = (in.get_data_type() == DataType::FLOAT64) ? 8 : 4;
    size_t bytes = n * n * element_size;
    size_t available = device_->get_available_memory_bytes();
    
    // Heuristic: If matrix fits in 80% of VRAM, use in-core
    if (bytes < available * 0.8) {
        device_->inverse_incore(in, out);
        return;
    }

    // ---------------------------------------------------------
    // Out-of-Core Blocked LU Decomposition with Global Pivoting
    // ---------------------------------------------------------
    
    if (in.get_data_type() == DataType::FLOAT64) {
        const DenseMatrix<double>* in_dense = static_cast<const DenseMatrix<double>*>(&in);
        DenseMatrix<double>* out_dense = static_cast<DenseMatrix<double>*>(&out);
        
        size_t N = in.size();
        const double* src = in_dense->data();
        double* dst = out_dense->data();
        
        std::string temp_path = make_unique_storage_file("lu_workspace");
        DenseMatrix<double> lu_workspace(N, temp_path);
        double* lu_data = lu_workspace.data();
        
        // Copy Input to LU Workspace
        for(size_t i=0; i<N; ++i) {
            std::copy(src + i * N, src + i * N + N, lu_data + i * N);
        }
        
        // Initialize Output to Identity
        std::fill(dst, dst + N*N, 0.0);
        for(size_t i=0; i<N; ++i) {
            dst[i*N + i] = 1.0;
        }
        
        // Determine Block Size
        size_t block_size = 1024;
        while (block_size * N * sizeof(double) > available * 0.4 && block_size > 32) {
            block_size /= 2;
        }
        
        std::cout << "Out-of-Core Inverse (Double): N=" << N << " BlockSize=" << block_size << std::endl;

        // LU Factorization Loop
        for (size_t k = 0; k < N; k += block_size) {
            size_t b = std::min(block_size, N - k);
            
            std::vector<int> pivots = factor_panel(lu_data, N, k, b);
            apply_pivots(lu_data, N, k, b, pivots, false);
            apply_pivots(dst, N, k, b, pivots, true);
            solve_row_panel(lu_data, N, k, b);
            
            if (k + b < N) {
                double* A_ik = lu_data + (k + b) * N + k;
                double* A_kj = lu_data + k * N + (k + b);
                double* A_ij = lu_data + (k + b) * N + (k + b);
                
                gemm_streaming(
                    N - (k + b), N - (k + b), b,
                    -1.0,
                    A_ik, N,
                    A_kj, N,
                    1.0,
                    A_ij, N
                );
            }
        }
        
        solve_forward(lu_data, dst, N, block_size);
        solve_backward(lu_data, dst, N, block_size);
        
    } else if (in.get_data_type() == DataType::FLOAT32) {
        const DenseMatrix<float>* in_dense = static_cast<const DenseMatrix<float>*>(&in);
        DenseMatrix<float>* out_dense = static_cast<DenseMatrix<float>*>(&out);
        
        size_t N = in.size();
        const float* src = in_dense->data();
        float* dst = out_dense->data();
        
        std::string temp_path = make_unique_storage_file("lu_workspace_float");
        DenseMatrix<float> lu_workspace(N, temp_path);
        float* lu_data = lu_workspace.data();
        
        for(size_t i=0; i<N; ++i) {
            std::copy(src + i * N, src + i * N + N, lu_data + i * N);
        }
        
        std::fill(dst, dst + N*N, 0.0f);
        for(size_t i=0; i<N; ++i) {
            dst[i*N + i] = 1.0f;
        }
        
        size_t block_size = 1024;
        while (block_size * N * sizeof(float) > available * 0.4 && block_size > 32) {
            block_size /= 2;
        }
        
        std::cout << "Out-of-Core Inverse (Float): N=" << N << " BlockSize=" << block_size << std::endl;

        for (size_t k = 0; k < N; k += block_size) {
            size_t b = std::min(block_size, N - k);
            
            std::vector<int> pivots = factor_panel(lu_data, N, k, b);
            apply_pivots(lu_data, N, k, b, pivots, false);
            apply_pivots(dst, N, k, b, pivots, true);
            solve_row_panel(lu_data, N, k, b);
            
            if (k + b < N) {
                float* A_ik = lu_data + (k + b) * N + k;
                float* A_kj = lu_data + k * N + (k + b);
                float* A_ij = lu_data + (k + b) * N + (k + b);
                
                gemm_streaming(
                    N - (k + b), N - (k + b), b,
                    -1.0f,
                    A_ik, N,
                    A_kj, N,
                    1.0f,
                    A_ij, N
                );
            }
        }
        
        solve_forward(lu_data, dst, N, block_size);
        solve_backward(lu_data, dst, N, block_size);
        
    } else {
        throw std::runtime_error("Out-of-Core Solver only supports FLOAT64 and FLOAT32.");
    }
}

std::vector<int> CudaSolver::factor_panel(double* data, size_t N, size_t k, size_t b) {
    // Load A[k:N, k:k+b] into GPU
    size_t height = N - k;
    size_t panel_bytes = height * b * sizeof(double);
    double* d_panel;
    device_->check_cuda_error(cudaMalloc(&d_panel, panel_bytes), "Malloc Panel");
    
    std::vector<double> h_panel(height * b);
    // Gather (Col Major for cuSOLVER)
    for(size_t c=0; c<b; ++c) {
        for(size_t r=0; r<height; ++r) {
            h_panel[c*height + r] = data[(k+r)*N + (k+c)];
        }
    }
    device_->check_cuda_error(cudaMemcpy(d_panel, h_panel.data(), panel_bytes, cudaMemcpyHostToDevice), "Copy Panel");
    
    // Factorize using cusolverDnDgetrf
    int* d_ipiv;
    int* d_info;
    device_->check_cuda_error(cudaMalloc(&d_ipiv, std::min(height, b) * sizeof(int)), "Malloc Ipiv");
    device_->check_cuda_error(cudaMalloc(&d_info, sizeof(int)), "Malloc Info");
    
    int lwork = 0;
    cusolverDnDgetrf_bufferSize(device_->get_cusolver_handle(), height, b, d_panel, height, &lwork);
    double* d_work;
    cudaMalloc(&d_work, lwork * sizeof(double));
    
    cusolverDnDgetrf(device_->get_cusolver_handle(), height, b, d_panel, height, d_work, d_ipiv, d_info);
    
    cudaFree(d_work);
    
    // Copy back panel and pivots
    device_->check_cuda_error(cudaMemcpy(h_panel.data(), d_panel, panel_bytes, cudaMemcpyDeviceToHost), "Copy Panel Back");
    std::vector<int> pivots(std::min(height, b));
    device_->check_cuda_error(cudaMemcpy(pivots.data(), d_ipiv, pivots.size() * sizeof(int), cudaMemcpyDeviceToHost), "Copy Pivots");
    
    cudaFree(d_panel);
    cudaFree(d_ipiv);
    cudaFree(d_info);
    
    // Scatter back
    for(size_t c=0; c<b; ++c) {
        for(size_t r=0; r<height; ++r) {
            data[(k+r)*N + (k+c)] = h_panel[c*height + r];
        }
    }
    
    return pivots;
}

void CudaSolver::apply_pivots(double* data, size_t N, size_t k, size_t b, const std::vector<int>& pivots, bool full_row) {
    for(size_t i=0; i<pivots.size(); ++i) {
        int pivot_idx = pivots[i] - 1; // 0-based relative to k
        if (pivot_idx != i) {
            size_t row1 = k + i;
            size_t row2 = k + pivot_idx;
            
            if (full_row) {
                // Swap entire row
                for(size_t c=0; c<N; ++c) {
                    std::swap(data[row1*N + c], data[row2*N + c]);
                }
            } else {
                // Swap L part (0 to k)
                for(size_t c=0; c<k; ++c) {
                    std::swap(data[row1*N + c], data[row2*N + c]);
                }
                // Swap Trailing part (k+b to N)
                for(size_t c=k+b; c<N; ++c) {
                    std::swap(data[row1*N + c], data[row2*N + c]);
                }
            }
        }
    }
}

void CudaSolver::solve_row_panel(double* data, size_t N, size_t k, size_t b) {
    size_t width = N - (k + b);
    if (width == 0) return;
    
    size_t panel_bytes = b * width * sizeof(double);
    size_t diag_bytes = b * b * sizeof(double);
    
    double* d_panel;
    double* d_L;
    
    device_->check_cuda_error(cudaMalloc(&d_panel, panel_bytes), "Malloc Row Panel");
    device_->check_cuda_error(cudaMalloc(&d_L, diag_bytes), "Malloc L");
    
    // Gather Row Panel (Col Major)
    std::vector<double> h_panel(b * width);
    for(size_t c=0; c<width; ++c) {
        for(size_t r=0; r<b; ++r) {
            h_panel[c*b + r] = data[(k+r)*N + (k+b+c)];
        }
    }
    
    // Gather L (Col Major)
    std::vector<double> h_L(b * b, 0.0);
    for(size_t c=0; c<b; ++c) {
        for(size_t r=0; r<b; ++r) {
            if (r > c) {
                h_L[c*b + r] = data[(k+r)*N + (k+c)];
            } else if (r == c) {
                h_L[c*b + r] = 1.0; // Unit diagonal
            }
        }
    }
    
    device_->check_cuda_error(cudaMemcpy(d_panel, h_panel.data(), panel_bytes, cudaMemcpyHostToDevice), "Copy Row Panel");
    device_->check_cuda_error(cudaMemcpy(d_L, h_L.data(), diag_bytes, cudaMemcpyHostToDevice), "Copy L");
    
    // TRSM: Solve L * X = B
    double alpha = 1.0;
    cublasDtrsm(device_->get_cublas_handle(),
                CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                b, width,
                &alpha,
                d_L, b,
                d_panel, b);
                
    device_->check_cuda_error(cudaMemcpy(h_panel.data(), d_panel, panel_bytes, cudaMemcpyDeviceToHost), "Copy Result Back");
    
    cudaFree(d_panel);
    cudaFree(d_L);
    
    // Scatter back
    for(size_t c=0; c<width; ++c) {
        for(size_t r=0; r<b; ++r) {
            data[(k+r)*N + (k+b+c)] = h_panel[c*b + r];
        }
    }
}

void CudaSolver::solve_forward(const double* lu_data, double* rhs_data, size_t N, size_t block_size) {
    // Solve L * X = B
    // L is unit lower triangular in lu_data
    // B is rhs_data (overwritten with X)
    // We process block by block.
    
    for (size_t k = 0; k < N; k += block_size) {
        size_t b = std::min(block_size, N - k);
        
        // 1. Solve diagonal block L_kk * X_k = B_k
        // L_kk is b x b.
        
        size_t rhs_cols = N;
        size_t tile_cols = 1024; // Tile width for RHS
        
        // Load L_kk
        std::vector<double> h_L(b * b);
        for(size_t r=0; r<b; ++r) {
            for(size_t c=0; c<b; ++c) {
                if (r > c) h_L[c*b + r] = lu_data[(k+r)*N + (k+c)];
                else if (r == c) h_L[c*b + r] = 1.0;
                else h_L[c*b + r] = 0.0;
            }
        }
        double* d_L;
        device_->check_cuda_error(cudaMalloc(&d_L, b*b*sizeof(double)), "Malloc L fwd");
        device_->check_cuda_error(cudaMemcpy(d_L, h_L.data(), b*b*sizeof(double), cudaMemcpyHostToDevice), "Copy L fwd");
        
        for(size_t j=0; j<rhs_cols; j+=tile_cols) {
            size_t tc = std::min(tile_cols, rhs_cols - j);
            
            // Load B_k_tile (b x tc)
            std::vector<double> h_B(b * tc);
            for(size_t r=0; r<b; ++r) {
                for(size_t c=0; c<tc; ++c) {
                    h_B[c*b + r] = rhs_data[(k+r)*N + (j+c)];
                }
            }
            
            double* d_B;
            device_->check_cuda_error(cudaMalloc(&d_B, b*tc*sizeof(double)), "Malloc B fwd");
            device_->check_cuda_error(cudaMemcpy(d_B, h_B.data(), b*tc*sizeof(double), cudaMemcpyHostToDevice), "Copy B fwd");
            
            // TRSM: Solve L * X = B
            // SIDE_LEFT, LOWER, NO_TRANS, UNIT
            double alpha = 1.0;
            cublasDtrsm(device_->get_cublas_handle(),
                        CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                        b, tc,
                        &alpha,
                        d_L, b,
                        d_B, b);
            
            // Copy back X_k
            device_->check_cuda_error(cudaMemcpy(h_B.data(), d_B, b*tc*sizeof(double), cudaMemcpyDeviceToHost), "Copy X fwd");
            cudaFree(d_B);
            
            for(size_t r=0; r<b; ++r) {
                for(size_t c=0; c<tc; ++c) {
                    rhs_data[(k+r)*N + (j+c)] = h_B[c*b + r];
                }
            }
        }
        cudaFree(d_L);
        
        // Step 2: Update trailing B_i -= L_ik * X_k
        // L_ik is (N - k - b) x b
        // X_k is b x N
        // B_i is (N - k - b) x N
        if (k + b < N) {
            double* L_ik = (double*)lu_data + (k + b) * N + k;
            double* X_k = rhs_data + k * N;
            double* B_i = rhs_data + (k + b) * N;
            
            gemm_streaming(
                N - (k + b), N, b,
                -1.0,
                L_ik, N,
                X_k, N,
                1.0,
                B_i, N
            );
        }
    }
}

void CudaSolver::solve_backward(const double* lu_data, double* rhs_data, size_t N, size_t block_size) {
    // Solve U * X = B
    // U is upper triangular in lu_data
    // Process backwards from N to 0
    
    size_t num_blocks = (N + block_size - 1) / block_size;
    
    for (size_t block_idx = num_blocks; block_idx > 0; --block_idx) {
        size_t k = (block_idx - 1) * block_size;
        size_t b = std::min(block_size, N - k);
        
        // 1. Solve diagonal block U_kk * X_k = B_k
        // U_kk is b x b upper triangular.
        
        size_t rhs_cols = N;
        size_t tile_cols = 1024;
        
        // Load U_kk
        std::vector<double> h_U(b * b);
        for(size_t r=0; r<b; ++r) {
            for(size_t c=0; c<b; ++c) {
                if (r <= c) h_U[c*b + r] = lu_data[(k+r)*N + (k+c)];
                else h_U[c*b + r] = 0.0;
            }
        }
        double* d_U;
        device_->check_cuda_error(cudaMalloc(&d_U, b*b*sizeof(double)), "Malloc U bwd");
        device_->check_cuda_error(cudaMemcpy(d_U, h_U.data(), b*b*sizeof(double), cudaMemcpyHostToDevice), "Copy U bwd");
        
        for(size_t j=0; j<rhs_cols; j+=tile_cols) {
            size_t tc = std::min(tile_cols, rhs_cols - j);
            
            // Load B_k_tile (b x tc)
            std::vector<double> h_B(b * tc);
            for(size_t r=0; r<b; ++r) {
                for(size_t c=0; c<tc; ++c) {
                    h_B[c*b + r] = rhs_data[(k+r)*N + (j+c)];
                }
            }
            
            double* d_B;
            device_->check_cuda_error(cudaMalloc(&d_B, b*tc*sizeof(double)), "Malloc B bwd");
            device_->check_cuda_error(cudaMemcpy(d_B, h_B.data(), b*tc*sizeof(double), cudaMemcpyHostToDevice), "Copy B bwd");
            
            // TRSM: Solve U * X = B
            // SIDE_LEFT, UPPER, NO_TRANS, NON_UNIT
            double alpha = 1.0;
            cublasDtrsm(device_->get_cublas_handle(),
                        CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                        b, tc,
                        &alpha,
                        d_U, b,
                        d_B, b);
            
            // Copy back X_k
            device_->check_cuda_error(cudaMemcpy(h_B.data(), d_B, b*tc*sizeof(double), cudaMemcpyDeviceToHost), "Copy X bwd");
            cudaFree(d_B);
            
            for(size_t r=0; r<b; ++r) {
                for(size_t c=0; c<tc; ++c) {
                    rhs_data[(k+r)*N + (j+c)] = h_B[c*b + r];
                }
            }
        }
        cudaFree(d_U);
        
        // 2. Update "upper" B_i -= U_ik * X_k for i < k
        // U_ik is k x b (block above diagonal)
        // X_k is b x N
        // B_i is k x N
        if (k > 0) {
            double* U_ik = (double*)lu_data + 0 * N + k; // Start of column panel k, row 0
            // But U_ik is not contiguous in memory as a block if we take 0..k rows.
            // It is a submatrix A[0:k, k:k+b].
            // gemm_streaming handles stride (lda=N), so it's fine.
            
            double* X_k = rhs_data + k * N;
            double* B_i = rhs_data; // Start of matrix
            
            gemm_streaming(
                k, N, b,
                -1.0,
                U_ik, N,
                X_k, N,
                1.0,
                B_i, N
            );
        }
    }
}

void CudaSolver::gemm_streaming(
    size_t m, size_t n, size_t k,
    double alpha,
    const double* A, size_t lda,
    const double* B, size_t ldb,
    double beta,
    double* C, size_t ldc
) {
    // Tiled GEMM: C = alpha * A * B + beta * C
    // A is m x k, B is k x n, C is m x n
    // We tile C into blocks that fit in GPU memory
    
    size_t available = device_->get_available_memory_bytes();
    size_t tile_size = 1024;
    
    // Adjust tile size to fit 3 blocks (A_tile, B_tile, C_tile)
    while (3 * tile_size * tile_size * sizeof(double) > available * 0.8 && tile_size > 32) {
        tile_size /= 2;
    }
    
    for (size_t i = 0; i < m; i += tile_size) {
        size_t tm = std::min(tile_size, m - i);
        for (size_t j = 0; j < n; j += tile_size) {
            size_t tn = std::min(tile_size, n - j);
            
            // Accumulate C_tile
            // We need to loop over k dimension as well if A/B don't fit?
            // For simplicity, let's assume we can stream k in chunks too, 
            // but standard blocked GEMM usually loops k inside.
            
            // To keep it simple and robust:
            // Load C_tile (tm x tn)
            std::vector<double> h_C(tm * tn);
            for(size_t r=0; r<tm; ++r) {
                for(size_t c=0; c<tn; ++c) {
                    h_C[c*tm + r] = C[(i+r)*ldc + (j+c)]; // Col Major for cuBLAS
                }
            }
            
            double* d_C;
            double* d_A;
            double* d_B;
            cudaMalloc(&d_C, tm * tn * sizeof(double));
            cudaMemcpy(d_C, h_C.data(), tm * tn * sizeof(double), cudaMemcpyHostToDevice);
            
            // Scale C by beta
            // Actually cuBLAS handles beta.
            
            for (size_t l = 0; l < k; l += tile_size) {
                size_t tk = std::min(tile_size, k - l);
                
                cudaMalloc(&d_A, tm * tk * sizeof(double));
                cudaMalloc(&d_B, tk * tn * sizeof(double));
                
                // Load A_tile (tm x tk)
                std::vector<double> h_A(tm * tk);
                for(size_t r=0; r<tm; ++r) {
                    for(size_t c=0; c<tk; ++c) {
                        h_A[c*tm + r] = A[(i+r)*lda + (l+c)];
                    }
                }
                cudaMemcpy(d_A, h_A.data(), tm * tk * sizeof(double), cudaMemcpyHostToDevice);
                
                // Load B_tile (tk x tn)
                std::vector<double> h_B(tk * tn);
                for(size_t r=0; r<tk; ++r) {
                    for(size_t c=0; c<tn; ++c) {
                        h_B[c*tk + r] = B[(l+r)*ldb + (j+c)];
                    }
                }
                cudaMemcpy(d_B, h_B.data(), tk * tn * sizeof(double), cudaMemcpyHostToDevice);
                
                // GEMM
                // C = alpha * A * B + (l==0 ? beta : 1.0) * C
                double current_beta = (l == 0) ? beta : 1.0;
                cublasDgemm(device_->get_cublas_handle(),
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            tm, tn, tk,
                            &alpha,
                            d_A, tm,
                            d_B, tk,
                            &current_beta,
                            d_C, tm);
                            
                cudaFree(d_A);
                cudaFree(d_B);
            }
            
            // Copy C back
            cudaMemcpy(h_C.data(), d_C, tm * tn * sizeof(double), cudaMemcpyDeviceToHost);
            cudaFree(d_C);
            
            for(size_t r=0; r<tm; ++r) {
                for(size_t c=0; c<tn; ++c) {
                    C[(i+r)*ldc + (j+c)] = h_C[c*tm + r];
                }
            }
        }
    }
}
#if 0 // Legacy CUDA eigvals implementation removed (depended on deleted ComplexVector)
        int* d_Info;
        
        size_t bytes_A = n * n * sizeof(double);
        size_t bytes_W = n * sizeof(double);
        
        device_->check_cuda_error(cudaMalloc(&d_A, bytes_A), "Malloc A");
        device_->check_cuda_error(cudaMalloc(&d_W, bytes_W), "Malloc W");
        device_->check_cuda_error(cudaMalloc(&d_Info, sizeof(int)), "Malloc Info");
        
        device_->check_cuda_error(cudaMemcpy(d_A, mat_d->data(), bytes_A, cudaMemcpyHostToDevice), "Copy A");
        
        int lwork = 0;
        cusolverDnDsyevd_bufferSize(device_->get_cusolver_handle(), 
                                   CUSOLVER_EIG_MODE_NOVECTOR, 
                                   CUBLAS_FILL_MODE_LOWER, 
                                   n, d_A, n, d_W, &lwork);
                                   
        device_->check_cuda_error(cudaMalloc(&d_Work, lwork * sizeof(double)), "Malloc Work");
        
        cusolverDnDsyevd(device_->get_cusolver_handle(),
                        CUSOLVER_EIG_MODE_NOVECTOR,
                        CUBLAS_FILL_MODE_LOWER,
                        n, d_A, n, d_W,
                        d_Work, lwork, d_Info);
                        
        int info = 0;
        cudaMemcpy(&info, d_Info, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (info != 0) {
            std::cerr << "[PyCauset] GPU Eigenvalues failed (Info=" << info << "). Falling back to CPU." << std::endl;
            cudaFree(d_A); cudaFree(d_W); cudaFree(d_Work); cudaFree(d_Info);
            pycauset::eigvals_cpu(matrix, result);
            return;
        }
        
        std::vector<double> h_W(n);
        cudaMemcpy(h_W.data(), d_W, bytes_W, cudaMemcpyDeviceToHost);
        
        for(size_t i=0; i<n; ++i) {
            result.set(i, std::complex<double>(h_W[i], 0.0));
        }
        
        cudaFree(d_A); cudaFree(d_W); cudaFree(d_Work); cudaFree(d_Info);
        return;
    }
    
    // Float Precision
    if (auto* mat_f = dynamic_cast<const DenseMatrix<float>*>(&matrix)) {
        // Check Symmetry
        const float* h_data = mat_f->data();
        bool symmetric = true;
        for(size_t i=0; i<n; ++i) {
            for(size_t j=i+1; j<n; ++j) {
                if (std::abs(h_data[i*n+j] - h_data[j*n+i]) > 1e-6f) {
                    symmetric = false;
                    break;
                }
            }
            if (!symmetric) break;
        }

        if (!symmetric) {
             std::cerr << "[PyCauset] GPU Eigenvalues: Matrix is not symmetric. Falling back to CPU." << std::endl;
             pycauset::eigvals_cpu(matrix, result);
             return;
        }

        float* d_A;
        float* d_W;
        float* d_Work;
        int* d_Info;
        
        size_t bytes_A = n * n * sizeof(float);
        size_t bytes_W = n * sizeof(float);
        
        device_->check_cuda_error(cudaMalloc(&d_A, bytes_A), "Malloc A");
        device_->check_cuda_error(cudaMalloc(&d_W, bytes_W), "Malloc W");
        device_->check_cuda_error(cudaMalloc(&d_Info, sizeof(int)), "Malloc Info");
        
        device_->check_cuda_error(cudaMemcpy(d_A, mat_f->data(), bytes_A, cudaMemcpyHostToDevice), "Copy A");
        
        int lwork = 0;
        cusolverDnSsyevd_bufferSize(device_->get_cusolver_handle(), 
                                   CUSOLVER_EIG_MODE_NOVECTOR, 
                                   CUBLAS_FILL_MODE_LOWER, 
                                   n, d_A, n, d_W, &lwork);
                                   
        device_->check_cuda_error(cudaMalloc(&d_Work, lwork * sizeof(float)), "Malloc Work");
        
        cusolverDnSsyevd(device_->get_cusolver_handle(),
                        CUSOLVER_EIG_MODE_NOVECTOR,
                        CUBLAS_FILL_MODE_LOWER,
                        n, d_A, n, d_W,
                        d_Work, lwork, d_Info);
                        
        int info = 0;
        cudaMemcpy(&info, d_Info, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (info != 0) {
            std::cerr << "[PyCauset] GPU Eigenvalues failed (Info=" << info << "). Falling back to CPU." << std::endl;
            cudaFree(d_A); cudaFree(d_W); cudaFree(d_Work); cudaFree(d_Info);
            pycauset::eigvals_cpu(matrix, result);
            return;
        }
        
        std::vector<float> h_W(n);
        cudaMemcpy(h_W.data(), d_W, bytes_W, cudaMemcpyDeviceToHost);
        
        for(size_t i=0; i<n; ++i) {
            result.set(i, std::complex<double>((double)h_W[i], 0.0));
        }
        
        cudaFree(d_A); cudaFree(d_W); cudaFree(d_Work); cudaFree(d_Info);
        return;
    }

    // Symmetric Matrix (Double)
    if (auto* mat_sym = dynamic_cast<const SymmetricMatrix<double>*>(&matrix)) {
        std::complex<double> scalar = mat_sym->get_scalar();
        bool is_anti = mat_sym->is_antisymmetric();
        
        // Case: Antisymmetric + Imaginary Scalar -> Hermitian
        bool is_hermitian_via_anti = is_anti && (std::abs(scalar.real()) < 1e-12);
        
        if (is_hermitian_via_anti) {
            // std::cout << "[PyCauset] Using GPU for Antisymmetric Matrix (Hermitian path)..." << std::endl;
            
            cuDoubleComplex* d_A;
            double* d_W; 
            cuDoubleComplex* d_Work;
            int* d_Info;
            uint64_t* d_Offsets;
            char* d_Packed;
            
            // 1. Copy Offsets
            std::vector<uint64_t> offsets(n);
            for(size_t i=0; i<n; ++i) offsets[i] = mat_sym->get_row_offset(i);
            
            device_->check_cuda_error(cudaMalloc(&d_Offsets, n * sizeof(uint64_t)), "Malloc Offsets");
            device_->check_cuda_error(cudaMemcpy(d_Offsets, offsets.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice), "Copy Offsets");
            
            // 2. Copy Packed Data
            // Calculate size: Offset of last row + size of last row
            // Last row index is n-1.
            // SymmetricMatrix always has diagonal, so last row has 1 element.
            // Element size is 8 bytes. Alignment is 8 bytes.
            size_t total_bytes = offsets[n-1] + 8;
            
            device_->check_cuda_error(cudaMalloc(&d_Packed, total_bytes), "Malloc Packed");
            device_->check_cuda_error(cudaMemcpy(d_Packed, mat_sym->data(), total_bytes, cudaMemcpyHostToDevice), "Copy Packed");
            
            // 3. Allocate Dense Complex Matrix
            device_->check_cuda_error(cudaMalloc(&d_A, n * n * sizeof(cuDoubleComplex)), "Malloc A Complex");
            device_->check_cuda_error(cudaMalloc(&d_W, n * sizeof(double)), "Malloc W Real");
            device_->check_cuda_error(cudaMalloc(&d_Info, sizeof(int)), "Malloc Info");
            
            // 4. Expand
            dim3 block(16, 16);
            dim3 grid((unsigned int)((n + 15) / 16), (unsigned int)((n + 15) / 16));
            cuDoubleComplex c_scalar = make_cuDoubleComplex(scalar.real(), scalar.imag());
            
            k_expand_symmetric_complex<<<grid, block>>>(d_Packed, d_A, n, d_Offsets, c_scalar, is_anti);
            
            // 5. Solve (Zheevd)
            int lwork = 0;
            cusolverDnZheevd_bufferSize(device_->get_cusolver_handle(), 
                                       CUSOLVER_EIG_MODE_NOVECTOR, 
                                       CUBLAS_FILL_MODE_LOWER, 
                                       (int)n, d_A, (int)n, d_W, &lwork);
                                       
            device_->check_cuda_error(cudaMalloc(&d_Work, lwork * sizeof(cuDoubleComplex)), "Malloc Work");
            
            cusolverDnZheevd(device_->get_cusolver_handle(),
                            CUSOLVER_EIG_MODE_NOVECTOR,
                            CUBLAS_FILL_MODE_LOWER,
                            (int)n, d_A, (int)n, d_W,
                            d_Work, lwork, d_Info);
                            
            int info = 0;
            cudaMemcpy(&info, d_Info, sizeof(int), cudaMemcpyDeviceToHost);
            
            if (info != 0) {
                std::cerr << "[PyCauset] GPU Eigenvalues (Zheevd) failed (Info=" << info << "). Falling back to CPU." << std::endl;
                cudaFree(d_A); cudaFree(d_W); cudaFree(d_Work); cudaFree(d_Info); cudaFree(d_Packed); cudaFree(d_Offsets);
                pycauset::eigvals_cpu(matrix, result);
                return;
            }
            
            // 6. Copy Result
            std::vector<double> h_W(n);
            cudaMemcpy(h_W.data(), d_W, n * sizeof(double), cudaMemcpyDeviceToHost);
            
            for(size_t i=0; i<n; ++i) {
                result.set(i, std::complex<double>(h_W[i], 0.0));
            }
            
            cudaFree(d_A); cudaFree(d_W); cudaFree(d_Work); cudaFree(d_Info); cudaFree(d_Packed); cudaFree(d_Offsets);
            return;
        }
    }

    // Fallback
    pycauset::eigvals_cpu(matrix, result);
}

#endif


void CudaSolver::add(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    size_t n = a.size();
    size_t num_elements = n * n;
    
    if (auto* a_d = dynamic_cast<const DenseMatrix<double>*>(&a)) {
        auto* b_d = dynamic_cast<const DenseMatrix<double>*>(&b);
        auto* c_d = dynamic_cast<DenseMatrix<double>*>(&result);
        
        if (a_d && b_d && c_d) {
            device_->ensure_buffers(num_elements);
            double* d_A = device_->d_A_;
            double* d_B = device_->d_B_;
            double* d_C = device_->d_C_;
            
            device_->check_cuda_error(cudaMemcpy(d_A, a_d->data(), num_elements * sizeof(double), cudaMemcpyHostToDevice), "Copy A");
            device_->check_cuda_error(cudaMemcpy(d_B, b_d->data(), num_elements * sizeof(double), cudaMemcpyHostToDevice), "Copy B");
            
            int threads = 256;
            int blocks = (num_elements + threads - 1) / threads;
            k_add<<<blocks, threads>>>(d_A, d_B, d_C, num_elements);
            
            device_->check_cuda_error(cudaMemcpy(c_d->data(), d_C, num_elements * sizeof(double), cudaMemcpyDeviceToHost), "Copy C");
            return;
        }
    }
    
    if (auto* a_f = dynamic_cast<const DenseMatrix<float>*>(&a)) {
        auto* b_f = dynamic_cast<const DenseMatrix<float>*>(&b);
        auto* c_f = dynamic_cast<DenseMatrix<float>*>(&result);
        
        if (a_f && b_f && c_f) {
            device_->ensure_float_buffers(num_elements);
            float* d_A = device_->d_A_float_;
            float* d_B = device_->d_B_float_;
            float* d_C = device_->d_C_float_;
            
            device_->check_cuda_error(cudaMemcpy(d_A, a_f->data(), num_elements * sizeof(float), cudaMemcpyHostToDevice), "Copy A");
            device_->check_cuda_error(cudaMemcpy(d_B, b_f->data(), num_elements * sizeof(float), cudaMemcpyHostToDevice), "Copy B");
            
            int threads = 256;
            int blocks = (num_elements + threads - 1) / threads;
            k_add<<<blocks, threads>>>(d_A, d_B, d_C, num_elements);
            
            device_->check_cuda_error(cudaMemcpy(c_f->data(), d_C, num_elements * sizeof(float), cudaMemcpyDeviceToHost), "Copy C");
            return;
        }
    }
    
    throw std::runtime_error("CudaSolver::add only supports DenseMatrix<double> or DenseMatrix<float>");
}

void CudaSolver::subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    size_t n = a.size();
    size_t num_elements = n * n;
    
    if (auto* a_d = dynamic_cast<const DenseMatrix<double>*>(&a)) {
        auto* b_d = dynamic_cast<const DenseMatrix<double>*>(&b);
        auto* c_d = dynamic_cast<DenseMatrix<double>*>(&result);
        
        if (a_d && b_d && c_d) {
            device_->ensure_buffers(num_elements);
            double* d_A = device_->d_A_;
            double* d_B = device_->d_B_;
            double* d_C = device_->d_C_;
            
            device_->check_cuda_error(cudaMemcpy(d_A, a_d->data(), num_elements * sizeof(double), cudaMemcpyHostToDevice), "Copy A");
            device_->check_cuda_error(cudaMemcpy(d_B, b_d->data(), num_elements * sizeof(double), cudaMemcpyHostToDevice), "Copy B");
            
            int threads = 256;
            int blocks = (num_elements + threads - 1) / threads;
            k_sub<<<blocks, threads>>>(d_A, d_B, d_C, num_elements);
            
            device_->check_cuda_error(cudaMemcpy(c_d->data(), d_C, num_elements * sizeof(double), cudaMemcpyDeviceToHost), "Copy C");
            return;
        }
    }
    
    if (auto* a_f = dynamic_cast<const DenseMatrix<float>*>(&a)) {
        auto* b_f = dynamic_cast<const DenseMatrix<float>*>(&b);
        auto* c_f = dynamic_cast<DenseMatrix<float>*>(&result);
        
        if (a_f && b_f && c_f) {
            device_->ensure_float_buffers(num_elements);
            float* d_A = device_->d_A_float_;
            float* d_B = device_->d_B_float_;
            float* d_C = device_->d_C_float_;
            
            device_->check_cuda_error(cudaMemcpy(d_A, a_f->data(), num_elements * sizeof(float), cudaMemcpyHostToDevice), "Copy A");
            device_->check_cuda_error(cudaMemcpy(d_B, b_f->data(), num_elements * sizeof(float), cudaMemcpyHostToDevice), "Copy B");
            
            int threads = 256;
            int blocks = (num_elements + threads - 1) / threads;
            k_sub<<<blocks, threads>>>(d_A, d_B, d_C, num_elements);
            
            device_->check_cuda_error(cudaMemcpy(c_f->data(), d_C, num_elements * sizeof(float), cudaMemcpyDeviceToHost), "Copy C");
            return;
        }
    }
    
    throw std::runtime_error("CudaSolver::subtract only supports DenseMatrix<double> or DenseMatrix<float>");
}

void CudaSolver::multiply_scalar(const MatrixBase& a, double scalar, MatrixBase& result) {
    size_t n = a.size();
    size_t num_elements = n * n;
    
    if (auto* a_d = dynamic_cast<const DenseMatrix<double>*>(&a)) {
        auto* c_d = dynamic_cast<DenseMatrix<double>*>(&result);
        
        if (a_d && c_d) {
            device_->ensure_buffers(num_elements);
            double* d_A = device_->d_A_;
            double* d_C = device_->d_C_;
            
            device_->check_cuda_error(cudaMemcpy(d_A, a_d->data(), num_elements * sizeof(double), cudaMemcpyHostToDevice), "Copy A");
            
            int threads = 256;
            int blocks = (num_elements + threads - 1) / threads;
            k_mul_scalar<<<blocks, threads>>>(d_A, scalar, d_C, num_elements);
            
            device_->check_cuda_error(cudaMemcpy(c_d->data(), d_C, num_elements * sizeof(double), cudaMemcpyDeviceToHost), "Copy C");
            return;
        }
    }
    
    if (auto* a_f = dynamic_cast<const DenseMatrix<float>*>(&a)) {
        auto* c_f = dynamic_cast<DenseMatrix<float>*>(&result);
        
        if (a_f && c_f) {
            device_->ensure_float_buffers(num_elements);
            float* d_A = device_->d_A_float_;
            float* d_C = device_->d_C_float_;
            
            device_->check_cuda_error(cudaMemcpy(d_A, a_f->data(), num_elements * sizeof(float), cudaMemcpyHostToDevice), "Copy A");
            
            int threads = 256;
            int blocks = (num_elements + threads - 1) / threads;
            k_mul_scalar<<<blocks, threads>>>(d_A, (float)scalar, d_C, num_elements);
            
            device_->check_cuda_error(cudaMemcpy(c_f->data(), d_C, num_elements * sizeof(float), cudaMemcpyDeviceToHost), "Copy C");
            return;
        }
    }
    
    throw std::runtime_error("CudaSolver::multiply_scalar only supports DenseMatrix<double> or DenseMatrix<float>");
}

void CudaSolver::matmul_bit(const DenseMatrix<bool>& a, const DenseMatrix<bool>& b, DenseMatrix<int32_t>& result) {
    size_t n = a.size();
    if (b.size() != n || result.size() != n) {
        throw std::invalid_argument("Dimension mismatch");
    }

    // 1. Calculate sizes
    size_t stride_bytes = a.stride_bytes();
    size_t stride_words = stride_bytes / 8;
    size_t matrix_bytes = n * stride_bytes;
    size_t result_bytes = n * n * sizeof(int32_t);

    // 2. Allocate GPU Memory
    // We need: d_A, d_B, d_B_T, d_C
    // d_B_T is the transpose of B
    
    uint64_t *d_A, *d_B, *d_B_T;
    int32_t *d_C;
    
    device_->check_cuda_error(cudaMalloc(&d_A, matrix_bytes), "Malloc A");
    device_->check_cuda_error(cudaMalloc(&d_B, matrix_bytes), "Malloc B");
    device_->check_cuda_error(cudaMalloc(&d_B_T, matrix_bytes), "Malloc B_T");
    device_->check_cuda_error(cudaMalloc(&d_C, result_bytes), "Malloc C");
    
    // Initialize B_T to 0 (important for atomicOr)
    device_->check_cuda_error(cudaMemset(d_B_T, 0, matrix_bytes), "Memset B_T");

    // 3. Copy Data
    device_->check_cuda_error(cudaMemcpy(d_A, a.data(), matrix_bytes, cudaMemcpyHostToDevice), "Copy A");
    device_->check_cuda_error(cudaMemcpy(d_B, b.data(), matrix_bytes, cudaMemcpyHostToDevice), "Copy B");

    // 4. Transpose B -> B_T
    // std::cout << "[GPU] Launching Transpose Kernel..." << std::endl;
    dim3 block_trans(32, 1);
    dim3 grid_trans((unsigned int)((n + 31) / 32), (unsigned int)((n + 31) / 32));
    
    k_transpose_bits<<<grid_trans, block_trans>>>(d_B, d_B_T, n, stride_words);
    cudaDeviceSynchronize(); // Ensure completion for debugging
    device_->check_cuda_error(cudaGetLastError(), "Transpose Kernel");
    // std::cout << "[GPU] Transpose Complete." << std::endl;

    // 5. Matmul
    // std::cout << "[GPU] Launching Matmul Kernel..." << std::endl;
    dim3 block_mm(16, 16);
    dim3 grid_mm((unsigned int)((n + 15) / 16), (unsigned int)((n + 15) / 16));
    
    k_matmul_bits<<<grid_mm, block_mm>>>(d_A, d_B_T, d_C, n, stride_words);
    cudaDeviceSynchronize(); // Ensure completion for debugging
    device_->check_cuda_error(cudaGetLastError(), "Matmul Kernel");
    // std::cout << "[GPU] Matmul Complete." << std::endl;

    // 6. Copy Result
    // std::cout << "[GPU] Copying Result..." << std::endl;
    device_->check_cuda_error(cudaMemcpy(result.data(), d_C, result_bytes, cudaMemcpyDeviceToHost), "Copy C");
    // std::cout << "[GPU] Done." << std::endl;

    // 7. Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_B_T);
    cudaFree(d_C);
}

// -----------------------------------------------------------------------------
// Single Precision Implementations
// -----------------------------------------------------------------------------

std::vector<int> CudaSolver::factor_panel(float* data, size_t N, size_t k, size_t b) {
    size_t height = N - k;
    size_t panel_bytes = height * b * sizeof(float);
    float* d_panel;
    device_->check_cuda_error(cudaMalloc(&d_panel, panel_bytes), "Malloc Panel Float");
    
    std::vector<float> h_panel(height * b);
    for(size_t c=0; c<b; ++c) {
        for(size_t r=0; r<height; ++r) {
            h_panel[c*height + r] = data[(k+r)*N + (k+c)];
        }
    }
    device_->check_cuda_error(cudaMemcpy(d_panel, h_panel.data(), panel_bytes, cudaMemcpyHostToDevice), "Copy Panel Float");
    
    int* d_ipiv;
    int* d_info;
    device_->check_cuda_error(cudaMalloc(&d_ipiv, std::min(height, b) * sizeof(int)), "Malloc Ipiv");
    device_->check_cuda_error(cudaMalloc(&d_info, sizeof(int)), "Malloc Info");
    
    int lwork = 0;
    cusolverDnSgetrf_bufferSize(device_->get_cusolver_handle(), height, b, d_panel, height, &lwork);
    float* d_work;
    cudaMalloc(&d_work, lwork * sizeof(float));
    
    cusolverDnSgetrf(device_->get_cusolver_handle(), height, b, d_panel, height, d_work, d_ipiv, d_info);
    
    cudaFree(d_work);
    
    device_->check_cuda_error(cudaMemcpy(h_panel.data(), d_panel, panel_bytes, cudaMemcpyDeviceToHost), "Copy Panel Back Float");
    std::vector<int> pivots(std::min(height, b));
    device_->check_cuda_error(cudaMemcpy(pivots.data(), d_ipiv, pivots.size() * sizeof(int), cudaMemcpyDeviceToHost), "Copy Pivots");
    
    cudaFree(d_panel);
    cudaFree(d_ipiv);
    cudaFree(d_info);
    
    for(size_t c=0; c<b; ++c) {
        for(size_t r=0; r<height; ++r) {
            data[(k+r)*N + (k+c)] = h_panel[c*height + r];
        }
    }
    
    return pivots;
}

void CudaSolver::apply_pivots(float* data, size_t N, size_t k, size_t b, const std::vector<int>& pivots, bool full_row) {
    for(size_t i=0; i<pivots.size(); ++i) {
        int pivot_idx = pivots[i] - 1;
        if (pivot_idx != i) {
            size_t row1 = k + i;
            size_t row2 = k + pivot_idx;
            
            if (full_row) {
                for(size_t c=0; c<N; ++c) {
                    std::swap(data[row1*N + c], data[row2*N + c]);
                }
            } else {
                for(size_t c=0; c<k; ++c) {
                    std::swap(data[row1*N + c], data[row2*N + c]);
                }
                for(size_t c=k+b; c<N; ++c) {
                    std::swap(data[row1*N + c], data[row2*N + c]);
                }
            }
        }
    }
}

void CudaSolver::solve_row_panel(float* data, size_t N, size_t k, size_t b) {
    size_t width = N - (k + b);
    if (width == 0) return;
    
    size_t panel_bytes = b * width * sizeof(float);
    size_t diag_bytes = b * b * sizeof(float);
    
    float* d_panel;
    float* d_L;
    
    device_->check_cuda_error(cudaMalloc(&d_panel, panel_bytes), "Malloc Row Panel Float");
    device_->check_cuda_error(cudaMalloc(&d_L, diag_bytes), "Malloc L Float");
    
    std::vector<float> h_panel(b * width);
    for(size_t c=0; c<width; ++c) {
        for(size_t r=0; r<b; ++r) {
            h_panel[c*b + r] = data[(k+r)*N + (k+b+c)];
        }
    }
    
    std::vector<float> h_L(b * b, 0.0f);
    for(size_t c=0; c<b; ++c) {
        for(size_t r=0; r<b; ++r) {
            if (r > c) {
                h_L[c*b + r] = data[(k+r)*N + (k+c)];
            } else if (r == c) {
                h_L[c*b + r] = 1.0f;
            }
        }
    }
    
    device_->check_cuda_error(cudaMemcpy(d_panel, h_panel.data(), panel_bytes, cudaMemcpyHostToDevice), "Copy Row Panel Float");
    device_->check_cuda_error(cudaMemcpy(d_L, h_L.data(), diag_bytes, cudaMemcpyHostToDevice), "Copy L Float");
    
    float alpha = 1.0f;
    cublasStrsm(device_->get_cublas_handle(),
                CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                b, width,
                &alpha,
                d_L, b,
                d_panel, b);
                
    device_->check_cuda_error(cudaMemcpy(h_panel.data(), d_panel, panel_bytes, cudaMemcpyDeviceToHost), "Copy Result Back Float");
    
    cudaFree(d_panel);
    cudaFree(d_L);
    
    for(size_t c=0; c<width; ++c) {
        for(size_t r=0; r<b; ++r) {
            data[(k+r)*N + (k+b+c)] = h_panel[c*b + r];
        }
    }
}

void CudaSolver::solve_forward(const float* lu_data, float* rhs_data, size_t N, size_t block_size) {
    for (size_t k = 0; k < N; k += block_size) {
        size_t b = std::min(block_size, N - k);
        
        size_t rhs_cols = N;
        size_t tile_cols = 1024;
        
        std::vector<float> h_L(b * b);
        for(size_t r=0; r<b; ++r) {
            for(size_t c=0; c<b; ++c) {
                if (r > c) h_L[c*b + r] = lu_data[(k+r)*N + (k+c)];
                else if (r == c) h_L[c*b + r] = 1.0f;
                else h_L[c*b + r] = 0.0f;
            }
        }
        float* d_L;
        device_->check_cuda_error(cudaMalloc(&d_L, b*b*sizeof(float)), "Malloc L fwd Float");
        device_->check_cuda_error(cudaMemcpy(d_L, h_L.data(), b*b*sizeof(float), cudaMemcpyHostToDevice), "Copy L fwd Float");
        
        for(size_t j=0; j<rhs_cols; j+=tile_cols) {
            size_t tc = std::min(tile_cols, rhs_cols - j);
            
            std::vector<float> h_B(b * tc);
            for(size_t r=0; r<b; ++r) {
                for(size_t c=0; c<tc; ++c) {
                    h_B[c*b + r] = rhs_data[(k+r)*N + (j+c)];
                }
            }
            
            float* d_B;
            device_->check_cuda_error(cudaMalloc(&d_B, b*tc*sizeof(float)), "Malloc B fwd Float");
            device_->check_cuda_error(cudaMemcpy(d_B, h_B.data(), b*tc*sizeof(float), cudaMemcpyHostToDevice), "Copy B fwd Float");
            
            float alpha = 1.0f;
            cublasStrsm(device_->get_cublas_handle(),
                        CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                        b, tc,
                        &alpha,
                        d_L, b,
                        d_B, b);
            
            device_->check_cuda_error(cudaMemcpy(h_B.data(), d_B, b*tc*sizeof(float), cudaMemcpyDeviceToHost), "Copy X fwd Float");
            cudaFree(d_B);
            
            for(size_t r=0; r<b; ++r) {
                for(size_t c=0; c<tc; ++c) {
                    rhs_data[(k+r)*N + (j+c)] = h_B[c*b + r];
                }
            }
        }
        cudaFree(d_L);
        
        if (k + b < N) {
            float* L_ik = (float*)lu_data + (k + b) * N + k;
            float* X_k = rhs_data + k * N;
            float* B_i = rhs_data + (k + b) * N;
            
            gemm_streaming(
                N - (k + b), N, b,
                -1.0f,
                L_ik, N,
                X_k, N,
                1.0f,
                B_i, N
            );
        }
    }
}

void CudaSolver::solve_backward(const float* lu_data, float* rhs_data, size_t N, size_t block_size) {
    size_t num_blocks = (N + block_size - 1) / block_size;
    
    for (size_t block_idx = num_blocks; block_idx > 0; --block_idx) {
        size_t k = (block_idx - 1) * block_size;
        size_t b = std::min(block_size, N - k);
        
        size_t rhs_cols = N;
        size_t tile_cols = 1024;
        
        std::vector<float> h_U(b * b);
        for(size_t r=0; r<b; ++r) {
            for(size_t c=0; c<b; ++c) {
                if (r <= c) h_U[c*b + r] = lu_data[(k+r)*N + (k+c)];
                else h_U[c*b + r] = 0.0f;
            }
        }
        float* d_U;
        device_->check_cuda_error(cudaMalloc(&d_U, b*b*sizeof(float)), "Malloc U bwd Float");
        device_->check_cuda_error(cudaMemcpy(d_U, h_U.data(), b*b*sizeof(float), cudaMemcpyHostToDevice), "Copy U bwd Float");
        
        for(size_t j=0; j<rhs_cols; j+=tile_cols) {
            size_t tc = std::min(tile_cols, rhs_cols - j);
            
            std::vector<float> h_B(b * tc);
            for(size_t r=0; r<b; ++r) {
                for(size_t c=0; c<tc; ++c) {
                    h_B[c*b + r] = rhs_data[(k+r)*N + (j+c)];
                }
            }
            
            float* d_B;
            device_->check_cuda_error(cudaMalloc(&d_B, b*tc*sizeof(float)), "Malloc B bwd Float");
            device_->check_cuda_error(cudaMemcpy(d_B, h_B.data(), b*tc*sizeof(float), cudaMemcpyHostToDevice), "Copy B bwd Float");
            
            float alpha = 1.0f;
            cublasStrsm(device_->get_cublas_handle(),
                        CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                        b, tc,
                        &alpha,
                        d_U, b,
                        d_B, b);
            
            device_->check_cuda_error(cudaMemcpy(h_B.data(), d_B, b*tc*sizeof(float), cudaMemcpyDeviceToHost), "Copy X bwd Float");
            cudaFree(d_B);
            
            for(size_t r=0; r<b; ++r) {
                for(size_t c=0; c<tc; ++c) {
                    rhs_data[(k+r)*N + (j+c)] = h_B[c*b + r];
                }
            }
        }
        cudaFree(d_U);
        
        if (k > 0) {
            float* U_ik = (float*)lu_data + 0 * N + k;
            float* X_k = rhs_data + k * N;
            float* B_i = rhs_data;
            
            gemm_streaming(
                k, N, b,
                -1.0f,
                U_ik, N,
                X_k, N,
                1.0f,
                B_i, N
            );
        }
    }
}

void CudaSolver::gemm_streaming(
    size_t m, size_t n, size_t k,
    float alpha,
    const float* A, size_t lda,
    const float* B, size_t ldb,
    float beta,
    float* C, size_t ldc
) {
    size_t available = device_->get_available_memory_bytes();
    size_t tile_size = 1024;
    
    while (3 * tile_size * tile_size * sizeof(float) > available * 0.8 && tile_size > 32) {
        tile_size /= 2;
    }
    
    device_->ensure_float_buffers(tile_size * tile_size);
    
    float* d_A = device_->d_A_float_;
    float* d_B = device_->d_B_float_;
    float* d_C = device_->d_C_float_;

    for (size_t i = 0; i < m; i += tile_size) {
        size_t tm = std::min(tile_size, m - i);
        for (size_t j = 0; j < n; j += tile_size) {
            size_t tn = std::min(tile_size, n - j);
            
            std::vector<float> h_C(tm * tn);
            for(size_t r=0; r<tm; ++r) {
                for(size_t c=0; c<tn; ++c) {
                    h_C[c*tm + r] = C[(i+r)*ldc + (j+c)];
                }
            }
            
            cudaMemcpy(d_C, h_C.data(), tm * tn * sizeof(float), cudaMemcpyHostToDevice);
            
            for (size_t l = 0; l < k; l += tile_size) {
                size_t tk = std::min(tile_size, k - l);
                
                std::vector<float> h_A(tm * tk);
                for(size_t r=0; r<tm; ++r) {
                    for(size_t c=0; c<tk; ++c) {
                        h_A[c*tm + r] = A[(i+r)*lda + (l+c)];
                    }
                }
                cudaMemcpy(d_A, h_A.data(), tm * tk * sizeof(float), cudaMemcpyHostToDevice);
                
                std::vector<float> h_B(tk * tn);
                for(size_t r=0; r<tk; ++r) {
                    for(size_t c=0; c<tn; ++c) {
                        h_B[c*tk + r] = B[(l+r)*ldb + (j+c)];
                    }
                }
                cudaMemcpy(d_B, h_B.data(), tk * tn * sizeof(float), cudaMemcpyHostToDevice);
                
                float current_beta = (l == 0) ? beta : 1.0f;
                cublasSgemm(device_->get_cublas_handle(),
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            tm, tn, tk,
                            &alpha,
                            d_A, tm,
                            d_B, tk,
                            &current_beta,
                            d_C, tm);
            }
            
            cudaMemcpy(h_C.data(), d_C, tm * tn * sizeof(float), cudaMemcpyDeviceToHost);
            
            for(size_t r=0; r<tm; ++r) {
                for(size_t c=0; c<tn; ++c) {
                    C[(i+r)*ldc + (j+c)] = h_C[c*tm + r];
                }
            }
        }
    }
}



