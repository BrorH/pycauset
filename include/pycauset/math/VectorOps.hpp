#pragma once
// Vector arithmetic through the matrix machinery.
//
// A dense vector of length n has the same memory layout as a 1 x n dense matrix of the same
// dtype (bit vectors are padded like a bit-matrix row; complex float16 keeps its two planes).
// So vectors reuse the matrix kernels and their NumPy result-dtype rules (ScalarOps.hpp,
// PromotionResolver) instead of a second, weaker implementation: one source of truth for
// promotion, integer wrapping, exactness and views. Orientation (row/column) is vector
// metadata and is carried over from the left operand.

#include <memory>

#include "pycauset/math/ScalarOps.hpp"

namespace pycauset {

class MatrixBase;
class VectorBase;

// 1 x n matrix over v's storage (no copy for dense vectors; UnitVector is written out).
// The vector's factor and conjugation flag carry over; its orientation does not.
std::unique_ptr<MatrixBase> as_row_matrix(const VectorBase& v);

// Vector over the storage of a 1 x n (or n x 1) result; views are written out first.
std::unique_ptr<VectorBase> as_vector(std::unique_ptr<MatrixBase> m, bool transposed);

std::unique_ptr<VectorBase> scalar_op(const VectorBase& v, ScalarOpKind op, const ScalarArg& s);
std::unique_ptr<VectorBase> negate(const VectorBase& v);

enum class VectorBinaryOp { Add, Sub, Mul, Div };
// Element-wise a op b (sizes must match), NumPy result dtype.
std::unique_ptr<VectorBase> binary_op(const VectorBase& a, const VectorBase& b, VectorBinaryOp op);

}  // namespace pycauset
