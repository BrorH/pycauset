#pragma once

#include "pycauset/core/Types.hpp"
#include "pycauset/core/PromotionResolver.hpp"

namespace pycauset {
class MatrixBase;
}

namespace pycauset::structure {

// Result structure for an element-wise binary op on concrete operands (F-039). Unlike
// the type-only overload below, it knows orientation (a transposed triangle is a LOWER
// triangle), strictness (diagonal included or not) and which structured classes exist
// for `result_dtype` (F-032), and falls back to DENSE_FLOAT whenever it can't prove the
// structure holds. Used by the eager add / subtract / elementwise_multiply paths.
MatrixType resolve_result_structure(promotion::BinaryOp op, const MatrixBase& a, const MatrixBase& b,
                                    DataType result_dtype);

inline pycauset::MatrixType resolve_structure(pycauset::promotion::BinaryOp op, pycauset::MatrixType a, pycauset::MatrixType b) {
    using namespace pycauset;
    using pycauset::promotion::BinaryOp;

    // Helper predicates
    auto is_diagonal = [](MatrixType t) {
        return t == MatrixType::DIAGONAL || t == MatrixType::IDENTITY;
    };

    auto is_symmetric = [&](MatrixType t) {
        return t == MatrixType::SYMMETRIC || is_diagonal(t);
    };

    if (op == BinaryOp::ElementwiseMultiply) {
        // Hadamard product: Result is non-zero only where BOTH are non-zero.
        // Sparsity is the intersection.
        
        // If either is diagonal, the result is diagonal.
        if (is_diagonal(a) || is_diagonal(b)) return MatrixType::DIAGONAL;
        
        // If both are symmetric, result is symmetric (S_ij * S_ij = S_ji * S_ji).
        if (is_symmetric(a) && is_symmetric(b)) return MatrixType::SYMMETRIC;

        // Triangular * Triangular -> Triangular (at least).
        // Note: If orientations differ, it's Diagonal, but Triangular is a safe superset.
        if (a == MatrixType::TRIANGULAR_FLOAT && b == MatrixType::TRIANGULAR_FLOAT) return MatrixType::TRIANGULAR_FLOAT;

        return MatrixType::DENSE_FLOAT;
    } 
    else if (op == BinaryOp::Add || op == BinaryOp::Subtract) {
        // Elementwise Add/Sub: Result is non-zero where EITHER is non-zero.
        // Sparsity is the union.

        // Diag + Diag -> Diag
        if (is_diagonal(a) && is_diagonal(b)) return MatrixType::DIAGONAL;

        // Sym + Sym -> Sym
        if (is_symmetric(a) && is_symmetric(b)) return MatrixType::SYMMETRIC;

        // Tri + Tri -> Dense (unless we know orientations match, which we don't here)
        // So we fall back to Dense.
        
        return MatrixType::DENSE_FLOAT;
    }

    // Default for other ops (Matmul, etc. - though BinaryExpression is usually elementwise)
    return MatrixType::DENSE_FLOAT;
}

} // namespace pycauset::structure
