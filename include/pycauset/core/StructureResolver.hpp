#pragma once

#include "pycauset/core/Types.hpp"
#include "pycauset/core/PromotionResolver.hpp"

namespace pycauset::structure {

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
