#pragma once

#include "pycauset/matrix/MatrixBase.hpp"

namespace pycauset::matrix_promotion {

// Resolves the result matrix type (structure) for binary operations.
// Kept separate from ObjectFactory so "construction" and "policy" are decoupled.
inline MatrixType resolve(MatrixType a, MatrixType b) {
    if (a == MatrixType::DENSE_FLOAT || b == MatrixType::DENSE_FLOAT) return MatrixType::DENSE_FLOAT;

    if (a == MatrixType::SYMMETRIC && b == MatrixType::SYMMETRIC) return MatrixType::SYMMETRIC;
    if (a == MatrixType::ANTISYMMETRIC && b == MatrixType::ANTISYMMETRIC) return MatrixType::ANTISYMMETRIC;

    if (a == MatrixType::TRIANGULAR_FLOAT && b == MatrixType::TRIANGULAR_FLOAT) return MatrixType::TRIANGULAR_FLOAT;
    if (a == MatrixType::CAUSAL && b == MatrixType::CAUSAL) return MatrixType::CAUSAL;

    return MatrixType::DENSE_FLOAT;
}

} // namespace pycauset::matrix_promotion
