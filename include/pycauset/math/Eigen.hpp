#pragma once

#include <complex>
#include <utility>

#include "pycauset/matrix/MatrixBase.hpp"

namespace pycauset {

// Determinant: complex for complex inputs (or a complex view factor), otherwise the real
// value in the real part. (trace lives with the other reductions in math/Reductions.hpp.)
std::complex<double> determinant(const MatrixBase& matrix);

// (sign, log|det|) as NumPy's slogdet: sign is complex (unit modulus) for complex inputs,
// 0 with -inf for a singular matrix. Computed from the LU pivots, so it does not overflow.
std::pair<std::complex<double>, double> slogdet(const MatrixBase& matrix);

}
