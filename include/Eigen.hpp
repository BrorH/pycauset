#pragma once

#include "MatrixBase.hpp"
#include "ComplexVector.hpp"
#include "ComplexMatrix.hpp"
#include <vector>
#include <complex>

namespace pycauset {

// Returns eigenvalues as a ComplexVector
std::unique_ptr<ComplexVector> eigvals(const MatrixBase& matrix, const std::string& saveas_real = "", const std::string& saveas_imag = "");

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
