#pragma once

#include <complex>

#include "pycauset/matrix/MatrixBase.hpp"

namespace pycauset {

// Determinant: complex for complex inputs (or a complex view factor), otherwise the real
// value in the real part. (trace lives with the other reductions in math/Reductions.hpp.)
std::complex<double> determinant(const MatrixBase& matrix);

}
