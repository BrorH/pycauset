#include "pycauset/compute/cpu/CpuEigenSolver.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/core/MemoryHints.hpp"
#include "pycauset/matrix/SymmetricMatrix.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <complex>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace pycauset {

// Constants for Jacobi
const int MAX_SWEEPS = 15;
const double EPSILON = 1e-9;

void CpuEigenSolver::eigvals_outofcore(const MatrixBase& matrix, ComplexVector& result) {
    // For now, we only support Symmetric/Hermitian matrices for Block Jacobi
    // General non-symmetric matrices require QR which is harder to do out-of-core efficiently without blocking.
    // But Block Jacobi is excellent for Symmetric.
    
    // Check if symmetric/hermitian
    bool is_symmetric = false;
    bool is_hermitian = false;
    bool is_complex_scalar = std::abs(matrix.get_scalar().imag()) > 1e-14;

    if (auto* sym = dynamic_cast<const SymmetricMatrix<double>*>(&matrix)) {
        if (!sym->is_antisymmetric() && !is_complex_scalar) {
            is_symmetric = true;
        } else if (sym->is_antisymmetric() && is_complex_scalar && std::abs(matrix.get_scalar().real()) < 1e-14) {
            is_hermitian = true;
        }
    }
    
    if (is_symmetric || is_hermitian) {
        block_jacobi(matrix, result);
    } else {
        // Fallback to in-memory (which might crash if too big, but we warned the user)
        // Or implement Block Arnoldi for general matrices later.
        std::cerr << "[Warning] Out-of-core solver currently only supports Symmetric/Hermitian matrices. Falling back to in-memory." << std::endl;
        // We can't call the in-memory one easily from here without circular dependency if not careful.
        // But this function is called BY AutoSolver/Eigen.cpp, so we should throw or handle it.
        throw std::runtime_error("Out-of-core solver only supports Symmetric/Hermitian matrices.");
    }
}

void CpuEigenSolver::block_jacobi(const MatrixBase& matrix, ComplexVector& result) {
    uint64_t n = matrix.size();
    
    // 1. Create Dense Working Copy
    // We use DenseMatrix for efficient row/col access and SIMD.
    // Even if input is Symmetric, we expand to Dense for the solver to allow efficient row/col updates.
    // This doubles storage requirement but significantly speeds up access.
    std::string temp_path = "temp_solver_" + std::to_string(std::rand()) + ".mat";
    DenseMatrix<double> working_copy(n, temp_path);
    
    // Copy data (Parallel Copy)
    ParallelFor(0, n, [&](uint64_t i) {
        for(uint64_t j=0; j<n; ++j) {
            working_copy.set(i, j, matrix.get_element_as_double(i, j));
        }
    });

    // Get raw pointer for high-performance access
    double* data = working_copy.data();
    
    const uint64_t BLOCK_SIZE = 512;
    uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // 2. Block Jacobi Sweeps
    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {
        double max_offdiag = 0.0;
        
        for (uint64_t I = 0; I < num_blocks; ++I) {
            for (uint64_t J = I + 1; J < num_blocks; ++J) {
                uint64_t i_start = I * BLOCK_SIZE;
                uint64_t i_end = std::min(i_start + BLOCK_SIZE, n);
                uint64_t j_start = J * BLOCK_SIZE;
                uint64_t j_end = std::min(j_start + BLOCK_SIZE, n);
                
                uint64_t b_i = i_end - i_start;
                uint64_t b_j = j_end - j_start;
                uint64_t dim = b_i + b_j;

                // Hint Memory
                working_copy.hint(pycauset::core::MemoryHint::sequential(i_start * n * 8, b_i * n * 8));
                working_copy.hint(pycauset::core::MemoryHint::sequential(j_start * n * 8, b_j * n * 8));

                // 2.1 Load 2x2 Block Submatrix into Eigen
                Eigen::MatrixXd M(dim, dim);
                
                // Fill M
                // Top-Left (I, I)
                for(uint64_t r=0; r<b_i; ++r)
                    for(uint64_t c=0; c<b_i; ++c)
                        M(r, c) = data[(i_start + r) * n + (i_start + c)];
                
                // Bottom-Right (J, J)
                for(uint64_t r=0; r<b_j; ++r)
                    for(uint64_t c=0; c<b_j; ++c)
                        M(b_i + r, b_i + c) = data[(j_start + r) * n + (j_start + c)];
                
                // Off-Diagonal (I, J) and (J, I)
                double local_max = 0.0;
                for(uint64_t r=0; r<b_i; ++r) {
                    for(uint64_t c=0; c<b_j; ++c) {
                        double val = data[(i_start + r) * n + (j_start + c)];
                        M(r, b_i + c) = val;
                        M(b_i + c, r) = val; // Symmetric
                        local_max = std::max(local_max, std::abs(val));
                    }
                }
                
                max_offdiag = std::max(max_offdiag, local_max);
                if (local_max < EPSILON) continue;

                // 2.2 Solve Eigenproblem for Block
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M);
                const Eigen::MatrixXd& V = es.eigenvectors(); // Rotation Matrix (dim x dim)
                
                // 2.3 Apply Rotation to Rows I and J (Update A * V)
                // We update the full rows I and J.
                // A_row_I = V(0:bi, :)^T * old_row_I + V(bi:dim, :)^T * old_row_J
                
                // We parallelize this update over columns k
                ParallelFor(0, n, [&](uint64_t k) {
                    // Optimization: Use stack buffers to avoid heap allocation contention
                    double v_buf[1024];
                    double new_v_buf[1024];
                    
                    Eigen::Map<Eigen::VectorXd> v(v_buf, dim);
                    Eigen::Map<Eigen::VectorXd> new_v(new_v_buf, dim);

                    // Load v
                    for(uint64_t r=0; r<b_i; ++r) v(r) = data[(i_start + r) * n + k];
                    for(uint64_t r=0; r<b_j; ++r) v(b_i + r) = data[(j_start + r) * n + k];
                    
                    // Compute (No Allocation)
                    new_v.noalias() = V.transpose() * v;
                    
                    // Store back
                    for(uint64_t r=0; r<b_i; ++r) data[(i_start + r) * n + k] = new_v(r);
                    for(uint64_t r=0; r<b_j; ++r) data[(j_start + r) * n + k] = new_v(b_i + r);
                });
                
                // 2.4 Apply Rotation to Columns I and J (Update V^T * A)
                // Update Columns I and J.
                // This touches A_{k, I} and A_{k, J}.
                // Parallelize over rows k.
                
                ParallelFor(0, n, [&](uint64_t k) {
                    // Optimization: Use stack buffers
                    double v_buf[1024];
                    double new_v_buf[1024];
                    
                    Eigen::Map<Eigen::VectorXd> v(v_buf, dim);
                    Eigen::Map<Eigen::VectorXd> new_v(new_v_buf, dim);

                    // Read row k elements at cols I and J
                    for(uint64_t c=0; c<b_i; ++c) v(c) = data[k * n + (i_start + c)];
                    for(uint64_t c=0; c<b_j; ++c) v(b_i + c) = data[k * n + (j_start + c)];
                    
                    // Compute (No Allocation)
                    new_v.noalias() = v.transpose() * V; // Row vector * Matrix
                    
                    // Store back
                    for(uint64_t c=0; c<b_i; ++c) data[k * n + (i_start + c)] = new_v(c);
                    for(uint64_t c=0; c<b_j; ++c) data[k * n + (j_start + c)] = new_v(b_i + c);
                });
            }
        }
        
        if (max_offdiag < EPSILON) break;
    }
    
    // 3. Extract Diagonal
    for (uint64_t i = 0; i < n; ++i) {
        result.set(i, data[i * n + i]);
    }
}

void CpuEigenSolver::solve_2x2_block(double& aii, double& ajj, double& aij, double& c, double& s) {
    // SymSchur2 decomposition
    if (std::abs(aij) < 1e-15) {
        c = 1.0; s = 0.0;
        return;
    }
    double tau = (ajj - aii) / (2.0 * aij);
    double t;
    if (tau >= 0) t = 1.0 / (tau + std::sqrt(1.0 + tau*tau));
    else          t = -1.0 / (-tau + std::sqrt(1.0 + tau*tau));
    
    c = 1.0 / std::sqrt(1.0 + t*t);
    s = t * c;
}

} // namespace pycauset
