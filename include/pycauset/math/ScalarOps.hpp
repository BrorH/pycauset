#pragma once

// Matrix (op) scalar operations: one entry point for `A + s`, `A - s`, `s - A`, `A * s`,
// `A / s`, `s / A` and `-A`.
//
// Result dtype (DType System §9, review 3.3 C-3): the scalar is an operand of one dtype and
// promotion::resolve decides, as for two matrices. A NumPy scalar has its own dtype; a Python
// bool is bool; a Python int takes the matrix's integer dtype (int32 with a bool matrix); a
// Python float the matrix's float dtype (float64 with an integer or bool matrix); a Python
// complex the matrix's complex precision. A Python int that does not fit the result dtype
// raises std::overflow_error; integer arithmetic wraps, as in NumPy.
//
// Kernels: a float / complex result of the matrix's dtype is an O(1) scale view (`A * s`,
// `A / s` when 1/s is finite); integer factors are O(1) integer views where the factor stays
// exact; everything else computes through the typed kernels, which keep a triangular or
// symmetric structure when the op maps 0 to 0.

#include <complex>
#include <cstdint>
#include <memory>
#include <utility>

#include "pycauset/core/Types.hpp"

namespace pycauset {

class MatrixBase;
class VectorBase;

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
    // A Python int beyond 64 bits, held as a Float: NumPy 2 computes it with a float or complex
    // matrix and raises OverflowError for an integer or bool one (review 3.7 C-9).
    bool beyond_64_bits = false;

    static ScalarArg from_int64(int64_t v);
    static ScalarArg from_uint64(uint64_t v);
    static ScalarArg from_double(double v);
    static ScalarArg from_complex(std::complex<double> v);

    std::complex<double> as_complex() const;
};

// The dtype NumPy would produce for `a <op> s` (with PyCauset's bit->int32 rule).
DataType scalar_result_dtype(DataType a, ScalarOpKind op, const ScalarArg& s);

std::unique_ptr<MatrixBase> scalar_op(const MatrixBase& a, ScalarOpKind op, const ScalarArg& s);

// -A. Raises std::invalid_argument for bit/bool (NumPy forbids boolean negation).
std::unique_ptr<MatrixBase> negate(const MatrixBase& a);

// The one normaliser (review 3.6 C-16): a's logical values (factor, conjugation, transpose and
// window applied) in fresh storage of dtype dt, exact where dt can represent them. Keep::Dense
// gives a plain dense matrix (any structure, any view in); Keep::Structure keeps a's structure
// class where the dtype offers it (promoting operands to a result dtype; pc.materialize).
// Half-precision targets are always dense. A bool target only from bool input (raises otherwise).
enum class Keep { Dense, Structure };
std::unique_ptr<MatrixBase> convert(const MatrixBase& a, DataType dt, Keep keep = Keep::Dense);

// No view metadata a kernel could ignore: not transposed, not a window, factor 1, and not a
// conjugated complex matrix. (Structure may be anything.)
bool is_plain(const MatrixBase& a);

// convert(a, a's dtype, Keep::Dense): same dtype (bit stays bit), view folded into storage. For
// kernels that read storage directly (F-061, F-062).
std::unique_ptr<MatrixBase> write_out(const MatrixBase& a);


// a's stored region (view applied) as a new matrix of structure s in a's dtype; entries
// outside the region are not read. Exact for every dtype. Raises for pairs not offered.
std::unique_ptr<MatrixBase> pack_as(const MatrixBase& a, Structure s, bool has_diagonal = false);

// Whether a (view applied) has structure s (SYMMETRIC: A == A.T, ANTISYMMETRIC: A == -A.T with a
// zero diagonal, HERMITIAN: A == conj(A).T), read a block of rows and columns at a time. Exact for
// integer and bool dtypes (in the dtype's wrapping arithmetic, as NumPy's -A); NumPy's
// isclose(rtol, atol) element by element for float and complex. Raises for a non-square matrix
// and for antisymmetric with an unsigned or bool dtype.
bool has_structure(const MatrixBase& a, Structure s, double rtol, double atol);

// For pc.causal_matrix(data): (some value is neither 0 nor 1, some non-zero value is on or below
// the diagonal), read a block of rows at a time.
std::pair<bool, bool> causal_input_check(const MatrixBase& a);

// A DiagonalMatrix of v's dtype with v (view applied) on its diagonal.
std::unique_ptr<MatrixBase> diagonal_from_vector(const VectorBase& v);

}  // namespace pycauset
