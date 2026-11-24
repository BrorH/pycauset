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
Structure resolve_result_structure(promotion::BinaryOp op, const MatrixBase& a, const MatrixBase& b,
                                    DataType result_dtype);

inline pycauset::Structure resolve_structure(pycauset::promotion::BinaryOp op, pycauset::Structure a, pycauset::Structure b) {
    using namespace pycauset;
    using pycauset::promotion::BinaryOp;

    // Helper predicates
    auto is_diagonal = [](Structure t) {
        return t == Structure::DIAGONAL || t == Structure::IDENTITY;
    };

    auto is_symmetric = [&](Structure t) {
        return t == Structure::SYMMETRIC || is_diagonal(t);
    };

    if (op == BinaryOp::ElementwiseMultiply) {
        // Hadamard product: Result is non-zero only where BOTH are non-zero.
        // Sparsity is the intersection.
        
        // If either is diagonal, the result is diagonal.
        if (is_diagonal(a) || is_diagonal(b)) return Structure::DIAGONAL;
        
        // If both are symmetric, result is symmetric (S_ij * S_ij = S_ji * S_ji).
        if (is_symmetric(a) && is_symmetric(b)) return Structure::SYMMETRIC;

        // Triangular * Triangular -> Triangular (at least).
        // Note: If orientations differ, it's Diagonal, but Triangular is a safe superset.
        if (a == Structure::TRIANGULAR && b == Structure::TRIANGULAR) return Structure::TRIANGULAR;

        return Structure::DENSE;
    } 
    else if (op == BinaryOp::Add || op == BinaryOp::Subtract) {
        // Elementwise Add/Sub: Result is non-zero where EITHER is non-zero.
        // Sparsity is the union.

        // Diag + Diag -> Diag
        if (is_diagonal(a) && is_diagonal(b)) return Structure::DIAGONAL;

        // Sym + Sym -> Sym
        if (is_symmetric(a) && is_symmetric(b)) return Structure::SYMMETRIC;

        // Tri + Tri -> Dense (unless we know orientations match, which we don't here)
        // So we fall back to Dense.
        
        return Structure::DENSE;
    }

    // Default for other ops (Matmul, etc. - though BinaryExpression is usually elementwise)
    return Structure::DENSE;
}

} // namespace pycauset::structure
