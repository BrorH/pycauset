#include "CudaSolver.hpp"
#include "StoragePaths.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>

using namespace pycauset;

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
    size_t bytes = n * n * sizeof(double);
    size_t available = device_->get_available_memory_bytes();
    
    // Heuristic: If matrix fits in 80% of VRAM, use in-core
    if (bytes < available * 0.8) {
        device_->inverse_incore(in, out);
        return;
    }

    // ---------------------------------------------------------
    // Out-of-Core Blocked LU Decomposition with Global Pivoting
    // ---------------------------------------------------------
    
    const DenseMatrix<double>* in_dense = nullptr;
    DenseMatrix<double>* out_dense = nullptr;

    if (in.get_data_type() == DataType::FLOAT64) {
        in_dense = static_cast<const DenseMatrix<double>*>(&in);
    }
    if (out.get_data_type() == DataType::FLOAT64) {
        out_dense = static_cast<DenseMatrix<double>*>(&out);
    }
    
    if (!in_dense || !out_dense) {
        throw std::runtime_error("Out-of-Core Solver only supports Double Precision (FLOAT64).");
    }
    
    size_t N = in.size();
    const double* src = in_dense->data();
    double* dst = out_dense->data();
    
    // 1. Create Workspace for LU Factorization
    // We need a temporary matrix to hold the LU factors.
    // 'out' will hold the inverse (initialized to Identity).
    
    std::string temp_path = make_unique_storage_file("lu_workspace");
    DenseMatrix<double> lu_workspace(N, temp_path);
    double* lu_data = lu_workspace.data();
    
    // Copy Input to LU Workspace
    for(size_t i=0; i<N; ++i) {
        std::copy(src + i * N, src + i * N + N, lu_data + i * N);
    }
    
    // Initialize Output to Identity
    // We will apply pivots to this identity matrix to get P.
    std::fill(dst, dst + N*N, 0.0);
    for(size_t i=0; i<N; ++i) {
        dst[i*N + i] = 1.0;
    }
    
    // 2. Determine Block Size
    size_t block_size = 1024;
    while (block_size * N * sizeof(double) > available * 0.4 && block_size > 32) {
        block_size /= 2;
    }
    
    std::cout << "Out-of-Core Inverse: N=" << N << " BlockSize=" << block_size << std::endl;

    // 3. LU Factorization Loop
    for (size_t k = 0; k < N; k += block_size) {
        size_t b = std::min(block_size, N - k);
        
        // A. Factor Panel (find pivots for columns k..k+b)
        std::vector<int> pivots = factor_panel(lu_data, N, k, b);
        
        // B. Apply Pivots to Trailing Submatrix of LU Workspace
        apply_pivots(lu_data, N, k, b, pivots, false);
        
        // C. Apply Pivots to Output (Identity -> P)
        // We apply to the WHOLE row of 'out'
        apply_pivots(dst, N, k, b, pivots, true);
        
        // D. Solve Row Panel (Update U part)
        solve_row_panel(lu_data, N, k, b);
        
        // E. Update Trailing Submatrix (Schur Complement)
        if (k + b < N) {
            double* A_ik = lu_data + (k + b) * N + k;       // Col Panel
            double* A_kj = lu_data + k * N + (k + b);       // Row Panel
            double* A_ij = lu_data + (k + b) * N + (k + b); // Trailing
            
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
    
    // 4. Forward Substitution: Solve L * Y = P (P is in 'out')
    // L is unit lower triangular stored in 'lu_workspace'
    solve_forward(lu_data, dst, N, block_size);
    
    // 5. Backward Substitution: Solve U * X = Y (Y is in 'out')
    // U is upper triangular stored in 'lu_workspace'
    solve_backward(lu_data, dst, N, block_size);
    
    // Cleanup is automatic for DenseMatrix destructor
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
