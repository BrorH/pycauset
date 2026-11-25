#pragma once

#include <complex>
#include <memory>

#include "pycauset/matrix/MatrixBase.hpp"

namespace pycauset {

// The one definition of view semantics (review PLAN Phase N.2). The logical value of element
// (i, j) of any matrix is
//
//     conj?(raw(i, j) * factor)
//
// where raw is the stored value at the (transposed, windowed) storage position, factor is
// the scalar view factor and conj the pending conjugation. Integer storage with an integer
// factor wraps like NumPy (F-042); real storage with a real factor multiplies in real
// arithmetic (a factor stored as (-1, -0) gives 0.0 * -1 = -0.0, as NumPy scaling a real
// array; a complex product would give +0.0); the diagonal of an antisymmetric matrix is +0.0
// whatever the factor (F-078).
enum class WideKind { Int64, Float64, Complex128 };

// The operand as a plain DenseMatrix<int64_t | double | std::complex<double>> holding its
// logical values (view applied, no structure kept), read through each class's typed `get`
// with one class dispatch per call: no per-element virtual call. Returns nullptr when the
// class has no typed reader for this operand (windows of structured classes, lazy/closure
// classes); callers then read through the virtual accessor.
std::unique_ptr<MatrixBase> read_wide(const MatrixBase& a, WideKind kind);

}  // namespace pycauset
