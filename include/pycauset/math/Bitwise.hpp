#pragma once

#include <memory>

#include "pycauset/matrix/MatrixBase.hpp"

namespace pycauset {

enum class BitwiseOp { And, Or, Xor };

// Element-wise a & b, a | b, a ^ b for bool and integer matrices of any structure and view, as
// NumPy computes them: bool op bool is bool; otherwise NumPy's integer promotion (bool counts
// as the smallest integer). Equal shapes. Float/complex operands: std::domain_error (the
// binding raises TypeError, as NumPy does). The result is dense.
std::unique_ptr<MatrixBase> bitwise(const MatrixBase& a, const MatrixBase& b, BitwiseOp op);

// ~a: logical not for bool, two's-complement not for integers (NumPy). Dense result.
std::unique_ptr<MatrixBase> bitwise_not(const MatrixBase& a);

}  // namespace pycauset
