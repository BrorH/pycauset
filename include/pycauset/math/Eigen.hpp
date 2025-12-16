#pragma once

#include "pycauset/matrix/MatrixBase.hpp"

namespace pycauset {

// Returns trace of the matrix
double trace(const MatrixBase& matrix);

// Returns determinant of the matrix
double determinant(const MatrixBase& matrix);

}
