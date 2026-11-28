#pragma once

#include <memory>

#include "pycauset/matrix/MatrixBase.hpp"

namespace pycauset {

enum class BitwiseOp { And, Or, Xor };

// Element-wise a & b, a | b, a ^ b for bool and integer matrices of any structure and view, as
// NumPy computes them: bool op bool is bool; otherwise NumPy's integer promotion (bool counts
// as the smallest integer). Equal shapes. Float/complex operands: pycauset::dtype_error (the
// binding raises TypeError, as NumPy does). The result is dense.
std::unique_ptr<MatrixBase> bitwise(const MatrixBase& a, const MatrixBase& b, BitwiseOp op);

// ~a: logical not for bool, two's-complement not for integers (NumPy). Dense result.
std::unique_ptr<MatrixBase> bitwise_not(const MatrixBase& a);

// a + b (sign +1) / a - b (sign -1) of two bool matrices of equal shape as int32 counts, from the
// packed words. Dense result.
std::unique_ptr<MatrixBase> bit_counts(const MatrixBase& a, const MatrixBase& b, int sign);

}  // namespace pycauset
