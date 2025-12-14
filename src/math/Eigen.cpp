#include "pycauset/math/Eigen.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/matrix/SymmetricMatrix.hpp"
#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/matrix/IdentityMatrix.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/core/StorageUtils.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/compute/ComputeContext.hpp"
#include "pycauset/compute/ComputeDevice.hpp"
#include "pycauset/compute/cpu/CpuEigenSolver.hpp"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <vector>
#include <random>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace pycauset {

// Helper: Block Matrix Multiply Y = A * X
// A: N x N (MatrixBase)
// X: N x b (Row Major flat vector)
// Y: N x b (Row Major flat vector)
void parallel_block_multiply(const MatrixBase& A, const std::vector<double>& X, std::vector<double>& Y, size_t n, size_t b) {
    ComputeContext::instance().get_device()->batch_gemv(A, X.data(), Y.data(), b);
}

// Helper: Load matrix into memory (FLAT vector for cache locality)
// Used by Arnoldi
std::vector<double> to_memory_flat(const MatrixBase& m) {
    uint64_t n = m.size();
    std::vector<double> mat(n * n);
    
    // Parallel copy if dense
    if (auto* dm = dynamic_cast<const DenseMatrix<double>*>(&m)) {
        const double* src = dm->data();
        std::copy(src, src + n*n, mat.begin());
    } else {
        // Fallback for other types
        pycauset::ParallelFor(0, n, [&](size_t i) {
            for(size_t j=0; j<n; ++j) {
                mat[i*n + j] = m.get_element_as_double(i, j);
            }
        });
    }
    return mat;
}

std::vector<std::complex<double>> to_memory_flat_complex(const MatrixBase& m) {
    uint64_t n = m.size();
    std::vector<std::complex<double>> mat(n * n);
    
    pycauset::ParallelFor(0, n, [&](size_t i) {
        for(size_t j=0; j<n; ++j) {
            mat[i*n + j] = m.get_element_as_complex(i, j);
        }
    });
    return mat;
}

double trace(const MatrixBase& matrix) {
    if (auto cached = matrix.get_cached_trace()) {
        return *cached;
    }
    
    double tr = 0.0;
    uint64_t n = matrix.size();
    auto type = matrix.get_matrix_type();
    
    if (type == MatrixType::IDENTITY) {
        tr = matrix.get_scalar().real() * n;
    } else {
        for(uint64_t i=0; i<n; ++i) {
            tr += matrix.get_element_as_double(i, i);
        }
    }
    
    matrix.set_cached_trace(tr);
    return tr;
}

void eigvals_cpu(const MatrixBase& matrix, ComplexVector& result) {
    uint64_t n = matrix.size();
    auto type = matrix.get_matrix_type();
    auto dtype = matrix.get_data_type();
    
    std::vector<std::complex<double>> vals;

    if (type == MatrixType::IDENTITY) {
        std::complex<double> val = matrix.get_scalar();
        vals.assign(n, val);
    } else if (type == MatrixType::DIAGONAL) {
        for(uint64_t i=0; i<n; ++i) vals.push_back(matrix.get_element_as_complex(i, i));
    } else if (type == MatrixType::TRIANGULAR_FLOAT || type == MatrixType::CAUSAL) {
        bool has_diag = false;
        if (auto* m = dynamic_cast<const TriangularMatrix<double>*>(&matrix)) has_diag = m->has_diagonal();
        else if (auto* m = dynamic_cast<const TriangularMatrix<int32_t>*>(&matrix)) has_diag = m->has_diagonal();
        
        if (has_diag) {
            for(uint64_t i=0; i<n; ++i) vals.push_back(matrix.get_element_as_complex(i, i));
        } else {
            vals.assign(n, 0.0);
        }
    } else {
        // Dense or other
        bool is_complex_scalar = std::abs(matrix.get_scalar().imag()) > 1e-14;
        
        // Check for Symmetric/Hermitian optimization
        bool is_symmetric = false;
        bool is_hermitian = false;
        
        if (auto* sym = dynamic_cast<const SymmetricMatrix<double>*>(&matrix)) {
            if (!sym->is_antisymmetric() && !is_complex_scalar) {
                is_symmetric = true;
            } else if (sym->is_antisymmetric() && is_complex_scalar && std::abs(matrix.get_scalar().real()) < 1e-14) {
                // Anti-symmetric * i = Hermitian
                is_hermitian = true;
            }
        }

        if (is_symmetric) {
             // Optimization: Use Out-of-Core Block Jacobi for large matrices
             // This avoids loading the entire matrix into RAM if it's huge, and uses parallel updates.
             // Threshold: 512 (Small enough for L2 cache blocks, large enough to matter)
             if (n >= 512) {
                 try {
                     CpuEigenSolver::eigvals_outofcore(matrix, result);
                     return; // Result is already set in result vector
                 } catch (const std::exception& e) {
                     // Fallback to in-memory if out-of-core fails (e.g. disk error)
                     std::cerr << "[PyCauset] Warning: Out-of-core solver failed (" << e.what() << "). Falling back to in-memory." << std::endl;
                 }
             }

             std::vector<double> data = to_memory_flat(matrix);
             Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                mat_eigen(data.data(), n, n);
             Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> es(mat_eigen, false);
             auto evals = es.eigenvalues();
             vals.resize(n);
             for(uint64_t i=0; i<n; ++i) vals[i] = evals[i];
        } else if (is_hermitian) {
             std::vector<std::complex<double>> data = to_memory_flat_complex(matrix);
             Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                mat_eigen(data.data(), n, n);
             Eigen::SelfAdjointEigenSolver<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> es(mat_eigen, false);
             auto evals = es.eigenvalues();
             vals.resize(n);
             for(uint64_t i=0; i<n; ++i) vals[i] = evals[i];
        } else if (is_complex_scalar) {
             std::vector<std::complex<double>> data = to_memory_flat_complex(matrix);
             Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                mat_eigen(data.data(), n, n);
             Eigen::ComplexEigenSolver<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> es(mat_eigen, false);
             auto evals = es.eigenvalues();
             vals.resize(n);
             for(uint64_t i=0; i<n; ++i) vals[i] = evals[i];
        } else if (dtype == DataType::FLOAT64 && type == MatrixType::DENSE_FLOAT) {
            const auto* dm = dynamic_cast<const DenseMatrix<double>*>(&matrix);
            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                mat_eigen(dm->data(), n, n);
            Eigen::EigenSolver<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> es(mat_eigen, false);
            auto evals = es.eigenvalues();
            vals.resize(n);
            for(uint64_t i=0; i<n; ++i) vals[i] = evals[i];
        } else if (dtype == DataType::FLOAT32 && type == MatrixType::DENSE_FLOAT) {
            const auto* dm = dynamic_cast<const DenseMatrix<float>*>(&matrix);
            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                mat_eigen(dm->data(), n, n);
            Eigen::EigenSolver<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> es(mat_eigen, false);
            auto evals = es.eigenvalues();
            vals.resize(n);
            for(uint64_t i=0; i<n; ++i) vals[i] = std::complex<double>(evals[i].real(), evals[i].imag());
        } else {
            // Fallback: Convert to double dense
            std::vector<double> data = to_memory_flat(matrix);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                mat_eigen(data.data(), n, n);
            Eigen::EigenSolver<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> es(mat_eigen, false);
            auto evals = es.eigenvalues();
            vals.resize(n);
            for(uint64_t i=0; i<n; ++i) vals[i] = evals[i];
        }
    }

    for(uint64_t i=0; i<n; ++i) {
        result.set(i, vals[i]);
    }
}

std::unique_ptr<ComplexVector> eigvals(const MatrixBase& matrix, const std::string& saveas_real, const std::string& saveas_imag) {
    /*
    if (auto cached = matrix.get_cached_eigenvalues()) {
        auto res = std::make_unique<ComplexVector>(matrix.size(), saveas_real, saveas_imag);
        for(uint64_t i=0; i<matrix.size(); ++i) {
            res->set(i, (*cached)[i]);
        }
        return res;
    }
    */

    auto res = std::make_unique<ComplexVector>(matrix.size(), saveas_real, saveas_imag);
    ComputeContext::instance().get_device()->eigvals(matrix, *res);
    
    /*
    // Cache the result
    std::vector<std::complex<double>> vals(matrix.size());
    for(uint64_t i=0; i<matrix.size(); ++i) {
        vals[i] = res->get(i);
    }
    matrix.set_cached_eigenvalues(vals);
    */
    
    return res;
}

std::unique_ptr<ComplexVector> eigvals(const ComplexMatrix& matrix, const std::string& saveas_real, const std::string& saveas_imag) {
    uint64_t n = matrix.size();
    auto res = std::make_unique<ComplexVector>(n, saveas_real, saveas_imag);

    // Load complex matrix into Eigen
    // We need to interleave real and imag parts or use Eigen::Map with stride?
    // Eigen::MatrixXcd stores complex numbers contiguously (real, imag, real, imag...)
    // Our ComplexMatrix stores real part in one file, imag part in another.
    // So we must copy.
    
    std::vector<std::complex<double>> data(n * n);
    
    // Parallel copy
    const auto* real_mat = matrix.real();
    const auto* imag_mat = matrix.imag();
    
    // Assuming dense float64 for now
    // TODO: Handle Float32
    
    if (real_mat->get_data_type() == DataType::FLOAT64) {
        const double* r_ptr = dynamic_cast<const DenseMatrix<double>*>(real_mat)->data();
        const double* i_ptr = dynamic_cast<const DenseMatrix<double>*>(imag_mat)->data();
        
        pycauset::ParallelFor(0, n * n, [&](size_t i) {
            data[i] = std::complex<double>(r_ptr[i], i_ptr[i]);
        });
    } else {
        // Fallback
        pycauset::ParallelFor(0, n, [&](size_t i) {
            for(size_t j=0; j<n; ++j) {
                data[i*n + j] = matrix.get(i, j);
            }
        });
    }
    
    // Use Eigen::ComplexEigenSolver
    // Eigen default is Column Major.
    // If we fill data as Row Major (i*n + j), we should tell Eigen it's Row Major.
    
    Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        mat_eigen_row(data.data(), n, n);
        
    Eigen::ComplexEigenSolver<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> ces;
    ces.compute(mat_eigen_row, false); // false = no eigenvectors
    
    auto evals = ces.eigenvalues();
    for(uint64_t i=0; i<n; ++i) {
        res->set(i, evals[i]);
    }
    
    return res;
}

double determinant(const MatrixBase& matrix) {
    if (auto cached = matrix.get_cached_determinant()) {
        return *cached;
    }
    
    double det = 0.0;
    uint64_t n = matrix.size();
    auto type = matrix.get_matrix_type();
    
    if (type == MatrixType::IDENTITY) {
        det = std::pow(matrix.get_scalar(), n).real();
    } else if (type == MatrixType::DIAGONAL) {
        det = 1.0;
        for(uint64_t i=0; i<n; ++i) det *= matrix.get_element_as_double(i, i);
    } else if (type == MatrixType::TRIANGULAR_FLOAT || type == MatrixType::CAUSAL) {
        // Determinant of triangular matrix is product of diagonal
        bool has_diag = false;
        if (auto* m = dynamic_cast<const TriangularMatrix<double>*>(&matrix)) has_diag = m->has_diagonal();
        else if (auto* m = dynamic_cast<const TriangularMatrix<int32_t>*>(&matrix)) has_diag = m->has_diagonal();
        
        if (has_diag) {
            det = 1.0;
            for(uint64_t i=0; i<n; ++i) det *= matrix.get_element_as_double(i, i);
        } else {
            det = 0.0; // Strictly triangular -> 0 diagonal -> det 0
        }
    } else {
        // Use eigenvalues product
        // This might be recursive if eigvals calls determinant, but eigvals uses QR which doesn't use determinant.
        // However, eigvals calls qr_algorithm which computes eigenvalues.
        // We need to be careful about infinite recursion if I implemented eigvals using determinant (I didn't).
        
        // We can call eigvals directly.
        // But eigvals returns ComplexVector.
        // We need the raw vector to avoid creating ComplexVector object if possible, but eigvals caches it.
        
        // Let's just call eigvals.
        // Note: eigvals returns unique_ptr<ComplexVector>.
        auto ev = eigvals(matrix);
        std::complex<double> prod(1.0, 0.0);
        for(uint64_t i=0; i<n; ++i) prod *= ev->get(i);
        det = prod.real();
    }
    
    matrix.set_cached_determinant(det);
    return det;
}

std::pair<std::unique_ptr<ComplexVector>, std::unique_ptr<ComplexMatrix>> eig(const MatrixBase& matrix, 
                                                                              const std::string& saveas_vals_real, 
                                                                              const std::string& saveas_vals_imag,
                                                                              const std::string& saveas_vecs_real,
                                                                              const std::string& saveas_vecs_imag) {
    uint64_t n = matrix.size();
    auto vals_vec = std::make_unique<ComplexVector>(n, saveas_vals_real, saveas_vals_imag);
    auto vecs_mat = std::make_unique<ComplexMatrix>(n, saveas_vecs_real, saveas_vecs_imag);
    
    auto type = matrix.get_matrix_type();
    auto dtype = matrix.get_data_type();
    
    if (type == MatrixType::IDENTITY) {
        double val = matrix.get_scalar().real();
        for(uint64_t i=0; i<n; ++i) {
            vals_vec->set(i, std::complex<double>(val, 0));
            vecs_mat->set(i, i, 1.0);
        }
    } else if (type == MatrixType::DIAGONAL) {
        for(uint64_t i=0; i<n; ++i) {
            vals_vec->set(i, matrix.get_element_as_double(i, i));
            vecs_mat->set(i, i, 1.0);
        }
    } else {
        // Dense or other
        if (dtype == DataType::FLOAT64 && type == MatrixType::DENSE_FLOAT) {
            const auto* dm = dynamic_cast<const DenseMatrix<double>*>(&matrix);
            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                mat_eigen(dm->data(), n, n);
            Eigen::EigenSolver<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> es(mat_eigen, true);
            
            auto evals = es.eigenvalues();
            auto evecs = es.eigenvectors();
            
            for(uint64_t i=0; i<n; ++i) {
                vals_vec->set(i, evals[i]);
                for(uint64_t j=0; j<n; ++j) {
                    vecs_mat->set(j, i, evecs(j, i));
                }
            }
        } else if (dtype == DataType::FLOAT32 && type == MatrixType::DENSE_FLOAT) {
            const auto* dm = dynamic_cast<const DenseMatrix<float>*>(&matrix);
            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                mat_eigen(dm->data(), n, n);
            Eigen::EigenSolver<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> es(mat_eigen, true);
            
            auto evals = es.eigenvalues();
            auto evecs = es.eigenvectors();
            
            for(uint64_t i=0; i<n; ++i) {
                vals_vec->set(i, std::complex<double>(evals[i].real(), evals[i].imag()));
                for(uint64_t j=0; j<n; ++j) {
                    vecs_mat->set(j, i, std::complex<double>(evecs(j, i).real(), evecs(j, i).imag()));
                }
            }
        } else {
            // Fallback: Convert to double dense
            std::vector<double> data = to_memory_flat(matrix);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                mat_eigen(data.data(), n, n);
            Eigen::EigenSolver<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> es(mat_eigen, true);
            
            auto evals = es.eigenvalues();
            auto evecs = es.eigenvectors();
            
            for(uint64_t i=0; i<n; ++i) {
                vals_vec->set(i, evals[i]);
                for(uint64_t j=0; j<n; ++j) {
                    vecs_mat->set(j, i, evecs(j, i));
                }
            }
        }
    }
    
    return {std::move(vals_vec), std::move(vecs_mat)};
}

std::pair<std::unique_ptr<ComplexVector>, std::unique_ptr<ComplexMatrix>> eig(const ComplexMatrix& matrix, 
                                                                              const std::string& saveas_vals_real, 
                                                                              const std::string& saveas_vals_imag,
                                                                              const std::string& saveas_vecs_real,
                                                                              const std::string& saveas_vecs_imag) {
    uint64_t n = matrix.size();
    auto vals_vec = std::make_unique<ComplexVector>(n, saveas_vals_real, saveas_vals_imag);
    auto vecs_mat = std::make_unique<ComplexMatrix>(n, saveas_vecs_real, saveas_vecs_imag);
    
    std::vector<std::complex<double>> data(n * n);
    
    const auto* real_mat = matrix.real();
    const auto* imag_mat = matrix.imag();
    
    const double* r_data = real_mat->data();
    const double* i_data = imag_mat->data();
    
    double r_scalar = real_mat->get_scalar().real();
    double i_scalar = imag_mat->get_scalar().real();
    
    bool r_trans = real_mat->is_transposed();
    bool i_trans = imag_mat->is_transposed();
    
    if (!r_trans && !i_trans) {
        // Fast path: both row-major
        if (r_scalar == 1.0 && i_scalar == 1.0) {
            for(uint64_t k=0; k<n*n; ++k) {
                data[k] = std::complex<double>(r_data[k], i_data[k]);
            }
        } else {
            for(uint64_t k=0; k<n*n; ++k) {
                data[k] = std::complex<double>(r_data[k] * r_scalar, i_data[k] * i_scalar);
            }
        }
    } else {
        // Slow path (one or both transposed)
        for(uint64_t i=0; i<n; ++i) {
            for(uint64_t j=0; j<n; ++j) {
                double r_val = r_trans ? r_data[j*n + i] : r_data[i*n + j];
                double i_val = i_trans ? i_data[j*n + i] : i_data[i*n + j];
                data[i*n + j] = std::complex<double>(r_val * r_scalar, i_val * i_scalar);
            }
        }
    }
    
    Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        mat_eigen(data.data(), n, n);
        
    Eigen::ComplexEigenSolver<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> ces(mat_eigen, true);
    
    auto evals = ces.eigenvalues();
    auto evecs = ces.eigenvectors();
    
    // Optimize output copy
    auto* res_real = vals_vec->real();
    auto* res_imag = vals_vec->imag();
    double* rr_data = res_real->data();
    double* ri_data = res_imag->data();
    
    for(uint64_t i=0; i<n; ++i) {
        rr_data[i] = evals[i].real();
        ri_data[i] = evals[i].imag();
    }
    
    auto* v_real = vecs_mat->real();
    auto* v_imag = vecs_mat->imag();
    double* vr_data = v_real->data();
    double* vi_data = v_imag->data();
    
    for(uint64_t i=0; i<n; ++i) {
        for(uint64_t j=0; j<n; ++j) {
            std::complex<double> val = evecs(j, i);
            uint64_t idx = j*n + i;
            vr_data[idx] = val.real();
            vi_data[idx] = val.imag();
        }
    }
    
    return {std::move(vals_vec), std::move(vecs_mat)};
}

std::unique_ptr<ComplexVector> eigvals_arnoldi(const MatrixBase& matrix, int k, int max_iter, double tol, const std::string& saveas_real, const std::string& saveas_imag) {
    size_t n = matrix.size();
    if (k > n) k = n;
    if (max_iter > n) max_iter = n;
    if (max_iter < k) max_iter = k + 20;
    if (max_iter > n) max_iter = n;

    // Block size for I/O optimization
    // For N=10^6, b=16 reduces I/O by 16x.
    size_t b = 16;
    if (b > k) b = k;
    if (b > max_iter) b = max_iter;
    if (b == 0) b = 1;

    // Adjust max_iter to be multiple of b for simplicity
    size_t steps = (max_iter + b - 1) / b;
    max_iter = steps * b;
    if (max_iter > n) {
        max_iter = n;
        steps = (max_iter + b - 1) / b;
    }

    // Krylov subspace basis Q (N x (max_iter + b))
    // Stored as flat vector (Row Major: N rows, (max_iter+b) cols)
    // Actually, storing as Column Major (max_iter+b columns of size N) is better for Gram-Schmidt.
    // But parallel_block_multiply expects X as Row Major (N x b).
    // Let's store Q as Column Major (vector of vectors) for GS, 
    // and transpose the active block to Row Major for multiply.
    std::vector<std::vector<double>> Q;
    Q.reserve(max_iter + b);

    // Hessenberg matrix H (max_iter x max_iter)
    // We store as flat vector
    std::vector<double> H(max_iter * max_iter, 0.0);

    // Initial random block
    for (size_t j = 0; j < b; ++j) {
        std::vector<double> q(n);
        for (size_t i = 0; i < n; ++i) q[i] = (double)rand() / RAND_MAX;
        Q.push_back(std::move(q));
    }

    // Orthonormalize initial block (QR of Q[0:b])
    for (size_t j = 0; j < b; ++j) {
        double norm = 0.0;
        for (double val : Q[j]) norm += val * val;
        norm = std::sqrt(norm);
        if (norm < 1e-14) norm = 1.0; // Safety
        for (size_t i = 0; i < n; ++i) Q[j][i] /= norm;

        for (size_t l = j + 1; l < b; ++l) {
            double dot = 0.0;
            for (size_t i = 0; i < n; ++i) dot += Q[j][i] * Q[l][i];
            for (size_t i = 0; i < n; ++i) Q[l][i] -= dot * Q[j][i];
        }
    }

    // Arnoldi Loop
    size_t m = 0; // Current column in H
    bool breakdown = false;
    for (size_t step = 0; step < steps; ++step) {
        if (m >= max_iter) break;
        
        size_t current_b = b;
        if (m + current_b > max_iter) current_b = max_iter - m;

        // Prepare input block X (N x current_b) in Row Major for multiply
        std::vector<double> X(n * current_b);
        pycauset::ParallelFor(0, n, [&](size_t i) {
            for (size_t j = 0; j < current_b; ++j) {
                X[i * current_b + j] = Q[m + j][i];
            }
        });

        // Compute W = A * X (N x current_b)
        std::vector<double> W_flat(n * current_b);
        parallel_block_multiply(matrix, X, W_flat, n, current_b);

        // Convert W back to Column Major vectors
        std::vector<std::vector<double>> W(current_b, std::vector<double>(n));
        pycauset::ParallelFor(0, n, [&](size_t i) {
            for (size_t j = 0; j < current_b; ++j) {
                W[j][i] = W_flat[i * current_b + j];
            }
        });

        // Block Gram-Schmidt
        // Orthogonalize W against all previous Q
        for (size_t j = 0; j < m + current_b; ++j) {
            // Compute H[j, m:m+b] = Q[j]^T * W
            // This is (1 x N) * (N x b) -> 1 x b
            // We can parallelize this dot product
            
            std::vector<double> h_row(current_b);
            for (size_t l = 0; l < current_b; ++l) {
                double dot = 0.0;
                // Simple sequential dot for now (vectorized by compiler hopefully)
                // Or use OpenMP reduction if available, but ParallelFor is custom.
                // Let's just do sequential dot, it's O(N).
                const auto& q_vec = Q[j];
                const auto& w_vec = W[l];
                for(size_t i=0; i<n; ++i) dot += q_vec[i] * w_vec[i];
                h_row[l] = dot;
            }

            // Store in H
            if (j < max_iter) {
                for (size_t l = 0; l < current_b; ++l) {
                    if (m + l < max_iter) {
                        H[j * max_iter + (m + l)] = h_row[l];
                    }
                }
            }

            // Subtract: W = W - Q[j] * h_row
            for (size_t l = 0; l < current_b; ++l) {
                double h_val = h_row[l];
                auto& w_vec = W[l];
                const auto& q_vec = Q[j];
                for(size_t i=0; i<n; ++i) w_vec[i] -= h_val * q_vec[i];
            }
        }

        // QR of W to get next basis vectors
        // We just do GS on W itself
        for (size_t j = 0; j < current_b; ++j) {
            double norm = 0.0;
            for (double val : W[j]) norm += val * val;
            norm = std::sqrt(norm);
            
            // Re-orthogonalize W[j] against W[0..j-1]
            // Note: W[0..j-1] have been moved to Q[m..m+j-1]
            // Actually, they are in Q[m+b..m+b+j-1] because we pushed W[0]..W[j-1] to Q
            for (size_t l = 0; l < j; ++l) {
                double dot = 0.0;
                const auto& w_prev = Q[m + b + l];
                for(size_t i=0; i<n; ++i) dot += w_prev[i] * W[j][i];
                for(size_t i=0; i<n; ++i) W[j][i] -= dot * w_prev[i];
                
                if (m + b + l < max_iter && m + j < max_iter) {
                    H[(m + b + l) * max_iter + (m + j)] += dot;
                }
            }
            
            norm = 0.0;
            for (double val : W[j]) norm += val * val;
            norm = std::sqrt(norm);
            
            if (m + b + j < max_iter && m + j < max_iter) {
                H[(m + b + j) * max_iter + (m + j)] = norm;
            }
            
            if (norm < tol) {
                breakdown = true;
                break;
            }
            for(size_t i=0; i<n; ++i) W[j][i] /= norm;
            
            Q.push_back(std::move(W[j]));
        }
        
        m += current_b;
        if (breakdown) break;
    }

    // Solve eigenvalues of H (m x m)
    size_t h_size = m;
    if (h_size > max_iter) h_size = max_iter;

    auto H_mat = std::make_unique<DenseMatrix<double>>(h_size, "");
    for(size_t i=0; i<h_size; ++i) {
        for(size_t j=0; j<h_size; ++j) {
            H_mat->set(i, j, H[i*max_iter + j]);
        }
    }
    
    return eigvals(*H_mat, saveas_real, saveas_imag);
}

std::unique_ptr<ComplexVector> eigvals_skew(const MatrixBase& matrix, int k, int max_iter, double tol, const std::string& saveas_real, const std::string& saveas_imag) {
    // Block Skew-Lanczos for Real Skew-Symmetric Matrices
    // A^T = -A
    // Eigenvalues are purely imaginary pairs +/- i*lambda
    
    size_t n = matrix.size();
    if (max_iter > n) max_iter = n;
    if (k > max_iter) k = max_iter;

    // Block size
    size_t b = 16;
    if (b > k) b = k;
    if (b > max_iter) b = max_iter;
    if (b == 0) b = 1;

    // Adjust max_iter
    size_t steps = (max_iter + b - 1) / b;
    max_iter = steps * b;
    if (max_iter > n) {
        max_iter = n;
        steps = (max_iter + b - 1) / b;
    }

    // We store the tridiagonal blocks
    // D_blocks: Diagonal blocks (should be 0)
    // C_blocks: Super-diagonal blocks (should be -B^T)
    // B_blocks: Sub-diagonal blocks
    
    std::vector<std::vector<double>> B_blocks; 
    std::vector<std::vector<double>> C_blocks;
    std::vector<std::vector<double>> D_blocks;
    
    B_blocks.reserve(steps);
    C_blocks.reserve(steps);
    D_blocks.reserve(steps);

    // Initial random block V_curr
    std::vector<std::vector<double>> V_curr(b, std::vector<double>(n));
    for (size_t j = 0; j < b; ++j) {
        for (size_t i = 0; i < n; ++i) V_curr[j][i] = (double)rand() / RAND_MAX;
    }

    // Orthonormalize V_curr (QR)
    for (size_t j = 0; j < b; ++j) {
        double norm = 0.0;
        for (double val : V_curr[j]) norm += val * val;
        norm = std::sqrt(norm);
        if (norm < 1e-14) norm = 1.0;
        for (size_t i = 0; i < n; ++i) V_curr[j][i] /= norm;

        for (size_t l = j + 1; l < b; ++l) {
            double dot = 0.0;
            for (size_t i = 0; i < n; ++i) dot += V_curr[j][i] * V_curr[l][i];
            for (size_t i = 0; i < n; ++i) V_curr[l][i] -= dot * V_curr[j][i];
        }
    }

    std::vector<std::vector<double>> V_prev; // Empty initially

    // Helper: Parallel Dot Product
    auto parallel_dot = [&](const std::vector<double>& v1, const std::vector<double>& v2) {
        size_t num_threads = ThreadPool::get_num_threads();
        if (n < 10000 || num_threads <= 1) { 
            double s = 0.0;
            for(size_t i=0; i<n; ++i) s += v1[i] * v2[i];
            return s;
        }
        
        std::vector<std::future<double>> futures;
        size_t chunk = (n + num_threads - 1) / num_threads;
        
        for(size_t t=0; t<num_threads; ++t) {
            size_t start = t * chunk;
            size_t end = std::min(start + chunk, n);
            if(start >= end) break;
            futures.emplace_back(ThreadPool::instance().enqueue([&, start, end]() {
                double s = 0.0;
                for(size_t i=start; i<end; ++i) s += v1[i] * v2[i];
                return s;
            }));
        }
        double sum = 0.0;
        for(auto& f : futures) sum += f.get();
        return sum;
    };

    bool breakdown = false;
    for (size_t step = 0; step < steps; ++step) {
        // 1. Compute W = A * V_curr
        // Prepare input X (Row Major)
        std::vector<double> X(n * b);
        pycauset::ParallelFor(0, n, [&](size_t i) {
            for (size_t j = 0; j < b; ++j) X[i * b + j] = V_curr[j][i];
        });

        std::vector<double> W_flat(n * b);
        parallel_block_multiply(matrix, X, W_flat, n, b);

        // Convert W back to Column Major
        std::vector<std::vector<double>> W(b, std::vector<double>(n));
        pycauset::ParallelFor(0, n, [&](size_t i) {
            for (size_t j = 0; j < b; ++j) W[j][i] = W_flat[i * b + j];
        });

        // Storage for coefficients
        std::vector<double> C_flat(b * b, 0.0); // Super-diagonal (V_prev^T * A * V_curr)
        std::vector<double> D_flat(b * b, 0.0); // Diagonal (V_curr^T * A * V_curr)

        // 2. Orthogonalize against V_prev (if exists) and V_curr
        // We do this twice (Re-orthogonalization) for stability
        for (int pass = 0; pass < 2; ++pass) {
            std::vector<double> C_pass(b * b, 0.0);
            std::vector<double> D_pass(b * b, 0.0);

            // Compute projections (CGS)
            // Parallelize over the b*b entries
            if (!V_prev.empty()) {
                pycauset::ParallelFor(0, b * b, [&](size_t idx) {
                    size_t l = idx / b;
                    size_t j = idx % b;
                    // Compute dot product V_prev[l] . W[j]
                    // Sequential dot here because we have b*b tasks
                    double dot = 0.0;
                    const auto& v = V_prev[l];
                    const auto& w = W[j];
                    for(size_t i=0; i<n; ++i) dot += v[i] * w[i];
                    C_pass[idx] = dot;
                });
            }

            pycauset::ParallelFor(0, b * b, [&](size_t idx) {
                size_t l = idx / b;
                size_t j = idx % b;
                double dot = 0.0;
                const auto& v = V_curr[l];
                const auto& w = W[j];
                for(size_t i=0; i<n; ++i) dot += v[i] * w[i];
                D_pass[idx] = dot;
            });

            // Accumulate coefficients
            for(size_t i=0; i<b*b; ++i) {
                C_flat[i] += C_pass[i];
                D_flat[i] += D_pass[i];
            }

            // Update W (Subtract projections)
            // Parallelize over N
            pycauset::ParallelFor(0, n, [&](size_t i) {
                for (size_t j = 0; j < b; ++j) {
                    double correction = 0.0;
                    if (!V_prev.empty()) {
                        for (size_t l = 0; l < b; ++l) {
                            correction += V_prev[l][i] * C_pass[l * b + j];
                        }
                    }
                    for (size_t l = 0; l < b; ++l) {
                        correction += V_curr[l][i] * D_pass[l * b + j];
                    }
                    W[j][i] -= correction;
                }
            });
        }
        
        if (!V_prev.empty()) C_blocks.push_back(C_flat);
        D_blocks.push_back(D_flat);

        // 3. QR of W to get V_next and B_next
        // W = V_next * B_next
        // We store B_next (b x b upper triangular)
        std::vector<double> B_flat(b * b, 0.0); // Row major b x b
        
        for (size_t j = 0; j < b; ++j) {
            // Norm
            double norm = parallel_dot(W[j], W[j]);
            norm = std::sqrt(norm);
            
            // Re-orthogonalize W[j] against W[0..j-1]
            // We can compute all dot products in parallel
            if (j > 0) {
                std::vector<double> dots(j);
                // Compute dots[l] = W[l] . W[j]
                // Since j is small, we can't parallelize over l effectively.
                // But we can parallelize the dot product computation itself if we fuse them?
                // Or just use parallel_dot for each l.
                for(size_t l=0; l<j; ++l) {
                    dots[l] = parallel_dot(W[l], W[j]);
                }
                
                // Update W[j]
                pycauset::ParallelFor(0, n, [&](size_t i) {
                    double correction = 0.0;
                    for(size_t l=0; l<j; ++l) {
                        correction += W[l][i] * dots[l];
                    }
                    W[j][i] -= correction;
                });
                
                // Store in B
                for(size_t l=0; l<j; ++l) B_flat[l * b + j] += dots[l];
            }
            
            // Re-compute norm
            norm = parallel_dot(W[j], W[j]);
            norm = std::sqrt(norm);
            
            B_flat[j * b + j] = norm;
            
            if (norm < tol) {
                breakdown = true;
                break;
            }
            
            // Normalize
            double inv_norm = 1.0 / norm;
            pycauset::ParallelFor(0, n, [&](size_t i) {
                W[j][i] *= inv_norm;
            });
        }
        
        B_blocks.push_back(B_flat);
        
        if (breakdown) break;
        
        // Shift
        V_prev = std::move(V_curr);
        V_curr = std::move(W);
    }

    // Construct Block Tridiagonal Matrix T (m x m)
    size_t m = B_blocks.size() * b;
    auto T_mat = std::make_unique<DenseMatrix<double>>(m, "");
    
    for (size_t step = 0; step < B_blocks.size(); ++step) {
        size_t row_start = step * b;
        size_t col_start = step * b;
        
        // Diagonal Block D
        if (step < D_blocks.size()) {
            const auto& D = D_blocks[step];
            for (size_t r = 0; r < b; ++r) {
                for (size_t c = 0; c < b; ++c) {
                    T_mat->set(row_start + r, col_start + c, D[r * b + c]);
                }
            }
        }
        
        // Sub-diagonal Block B (at step+1, step)
        if (step + 1 < B_blocks.size() + 1 && row_start + b < m) { // B_blocks has size steps
             // Wait, B_blocks[step] is B_{step+1}
             const auto& B = B_blocks[step];
             for (size_t r = 0; r < b; ++r) {
                for (size_t c = 0; c < b; ++c) {
                    T_mat->set(row_start + b + r, col_start + c, B[r * b + c]);
                }
            }
        }
        
        // Super-diagonal Block C (at step-1, step) -> No, C is at (step-1, step) in recurrence?
        // A V_{step} = V_{step-1} C + ...
        // So C is at (step-1, step).
        // Wait, C_blocks[step] was computed in step `step`.
        // It is V_{step-1}^T A V_{step}.
        // So it is the block at (step-1, step).
        // So T[ (step-1)*b : ..., step*b : ... ] = C_blocks[step]
        
        if (step > 0 && step - 1 < C_blocks.size()) {
             // C_blocks has size steps-1 (first step has no prev)
             // Actually, I push to C_blocks only if !V_prev.empty().
             // So C_blocks[0] corresponds to step 1.
             // So C_blocks[step-1] corresponds to step `step`.
             
             const auto& C = C_blocks[step-1];
             for (size_t r = 0; r < b; ++r) {
                for (size_t c = 0; c < b; ++c) {
                    T_mat->set(row_start - b + r, col_start + c, C[r * b + c]);
                }
            }
        }
    }
    
    auto full_evals = eigvals(*T_mat, "", "");
    
    // Sort and truncate to k
    std::vector<std::complex<double>> vals;
    vals.reserve(full_evals->size());
    for(size_t i=0; i<full_evals->size(); ++i) vals.push_back(full_evals->get(i));
    
    std::sort(vals.begin(), vals.end(), [](const std::complex<double>& a, const std::complex<double>& b) {
        return std::abs(a) > std::abs(b);
    });
    
    if (vals.size() > (size_t)k) vals.resize(k);
    
    auto res = std::make_unique<ComplexVector>(vals.size(), saveas_real, saveas_imag);
    for(size_t i=0; i<vals.size(); ++i) res->set(i, vals[i]);
    
    return res;
}

}
