#pragma once

// Matrix ⊕ scalar operations with NumPy result-dtype semantics (F-001, F-002).
//
// One entry point for `A + s`, `A - s`, `s - A`, `A * s`, `A / s`, `s / A` and `-A`.
// Rules (NumPy 2 / NEP 50, Python scalars are "weak"):
//   integer A  op Python int   -> A's dtype, two's-complement wraparound; a Python int
//                                 that does not fit A's dtype raises std::overflow_error
//   integer A  op Python float -> float64          integer A / anything -> float64
//   integer A  op complex      -> complex128
//   bit A      op Python int   -> int32 (PyCauset counting policy, F-013)
//   float/complex A op real    -> A's dtype (existing O(1) scalar-view paths)
//   real float A op complex    -> complex of matching precision
// Integer and bit inputs are computed exactly (typed element reads, no round trip
// through double). Triangular inputs keep their structure when the op maps 0 -> 0.
// Integer and bit matrices never receive a lazy `scalar_` (see FINDINGS F-001).

#include <complex>
#include <cstdint>
#include <memory>

#include "pycauset/core/Types.hpp"

namespace pycauset {

class MatrixBase;

enum class ScalarOpKind { Add, Sub, RSub, Mul, TrueDiv, RTrueDiv };

struct ScalarArg {
    enum class Kind { Int, Float, Complex };
    Kind kind = Kind::Int;
    // Int: value in [INT64_MIN, UINT64_MAX]. `negative` selects which field holds it.
    int64_t i = 0;
    uint64_t u = 0;
    bool negative = false;
    double f = 0.0;
    std::complex<double> c{0.0, 0.0};
    // NumPy scalars are "strong": their dtype takes part in promotion.
    bool strong = false;
    DataType strong_dtype = DataType::UNKNOWN;

    static ScalarArg from_int64(int64_t v);
    static ScalarArg from_uint64(uint64_t v);
    static ScalarArg from_double(double v);
    static ScalarArg from_complex(std::complex<double> v);

    std::complex<double> as_complex() const;
};

// True for every integer dtype and for bit/bool.
bool is_integer_or_bit(DataType dt);

// The dtype NumPy would produce for `a <op> s` (with PyCauset's bit->int32 rule).
DataType scalar_result_dtype(DataType a, ScalarOpKind op, const ScalarArg& s);

std::unique_ptr<MatrixBase> scalar_op(const MatrixBase& a, ScalarOpKind op, const ScalarArg& s);

// -A. Raises std::invalid_argument for bit/bool (NumPy forbids boolean negation).
// Plain copy of a view's values (factor applied exactly); used before consumers that read
// raw storage. Integer and float dtypes supported by the exact scalar kernels.
std::unique_ptr<MatrixBase> materialize(const MatrixBase& a);
std::unique_ptr<MatrixBase> negate(const MatrixBase& a);

// A converted to a dense matrix of dtype `dt` (exact where the target can represent the
// values). Used to promote operands of mixed-dtype ops to the resolved result dtype.
std::unique_ptr<MatrixBase> cast_to(const MatrixBase& a, DataType dt);

// No view metadata a kernel could ignore: not transposed, not a window, factor 1, and not a
// conjugated complex matrix. (Structure may be anything.)
bool is_plain(const MatrixBase& a);

// Dense copy of a's logical values in a's dtype (factor, conjugation, transpose and window
// applied; bit stays bit). For kernels that read storage directly (F-061, F-062).
std::unique_ptr<MatrixBase> write_out(const MatrixBase& a);

// Dense, plain copy of a's logical values converted to dt (any structure, any view). For
// kernels that need one storage layout and dtype (solve's LAPACK path).
std::unique_ptr<MatrixBase> to_dense(const MatrixBase& a, DataType dt);

// a's stored region (view applied) as a new matrix of structure s in a's dtype; entries
// outside the region are not read. Exact for every dtype. Raises for pairs not offered.
std::unique_ptr<MatrixBase> pack_as(const MatrixBase& a, Structure s, bool has_diagonal = false);

}  // namespace pycauset
