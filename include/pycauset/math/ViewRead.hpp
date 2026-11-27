#pragma once

#include <complex>
#include <cstdint>
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

// Row reader (review N.7): logical rows [i0, i1) of a into out (row-major, a.cols() wide),
// typed and view-applied, O(rows x cols) memory instead of the whole matrix. The diagonal
// reader fills min(rows, cols) entries. Same element rules and the same `false` cases as
// read_wide (windows of structured classes, lazy closures): callers then read virtually.
bool read_rows(const MatrixBase& a, uint64_t i0, uint64_t i1, int64_t* out);
bool read_rows(const MatrixBase& a, uint64_t i0, uint64_t i1, double* out);
bool read_rows(const MatrixBase& a, uint64_t i0, uint64_t i1, std::complex<double>* out);
bool read_diagonal(const MatrixBase& a, int64_t* out);
bool read_diagonal(const MatrixBase& a, double* out);
bool read_diagonal(const MatrixBase& a, std::complex<double>* out);

}  // namespace pycauset
