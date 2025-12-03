#pragma once

#include "MatrixBase.hpp"
#include "ComplexVector.hpp"
#include "ComplexMatrix.hpp"
#include <vector>
#include <complex>

namespace pycauset {

// Returns eigenvalues as a ComplexVector
std::unique_ptr<ComplexVector> eigvals(const MatrixBase& matrix, const std::string& saveas_real = "", const std::string& saveas_imag = "");

// Computes k largest magnitude eigenvalues using Arnoldi iteration
// Suitable for large sparse matrices where O(N^3) is infeasible.
std::unique_ptr<ComplexVector> eigvals_arnoldi(const MatrixBase& matrix, int k, int max_iter = 100, double tol = 1e-10, const std::string& saveas_real = "", const std::string& saveas_imag = "");

// Computes k largest magnitude eigenvalues for a Real Skew-Symmetric matrix
// Uses Block Skew-Lanczos iteration (optimized for memory and speed).
// Assumes matrix is skew-symmetric (A^T = -A).
std::unique_ptr<ComplexVector> eigvals_skew(const MatrixBase& matrix, int k, int max_iter = 100, double tol = 1e-10, const std::string& saveas_real = "", const std::string& saveas_imag = "");

// Returns trace of the matrix
double trace(const MatrixBase& matrix);

// Returns determinant of the matrix
double determinant(const MatrixBase& matrix);

// Returns pair of (eigenvalues, eigenvectors)
// Eigenvectors are returned as a ComplexMatrix where columns are eigenvectors.
std::pair<std::unique_ptr<ComplexVector>, std::unique_ptr<ComplexMatrix>> eig(const MatrixBase& matrix, 
                                                                              const std::string& saveas_vals_real = "", 
                                                                              const std::string& saveas_vals_imag = "",
                                                                              const std::string& saveas_vecs_real = "",
                                                                              const std::string& saveas_vecs_imag = "");

}
