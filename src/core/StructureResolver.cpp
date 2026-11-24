// Result-structure rules for element-wise binary ops. See StructureResolver.hpp.
#include "pycauset/core/StructureResolver.hpp"

#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"

namespace pycauset::structure {

namespace {

bool is_diagonal_like(const MatrixBase& m) {
    const MatrixType t = m.get_matrix_type();
    return t == MatrixType::DIAGONAL || t == MatrixType::IDENTITY;
}
bool is_symmetric_like(const MatrixBase& m) {
    return m.get_matrix_type() == MatrixType::SYMMETRIC || is_diagonal_like(m);
}
bool is_antisymmetric(const MatrixBase& m) { return m.get_matrix_type() == MatrixType::ANTISYMMETRIC; }

// Strictly upper triangular storage that is not viewed transposed (i.e. really upper).
bool is_strict_upper(const MatrixBase& m) {
    const MatrixType t = m.get_matrix_type();
    if (t != MatrixType::TRIANGULAR_FLOAT && t != MatrixType::CAUSAL) return false;
    if (m.is_transposed()) return false;
    const auto* tri = dynamic_cast<const TriangularMatrixBase*>(&m);
    return tri != nullptr && !tri->has_diagonal();
}

// Structured result classes that exist (and are registered with Python) per dtype.
bool can_make(MatrixType t, DataType dt) {
    switch (t) {
        case MatrixType::TRIANGULAR_FLOAT:
            // No BIT: the element-wise kernels cannot write a TriangularBitMatrix result.
            return dt == DataType::INT32 || dt == DataType::FLOAT64;
        case MatrixType::DIAGONAL:
        case MatrixType::SYMMETRIC:
        case MatrixType::ANTISYMMETRIC:
            return dt == DataType::FLOAT64;
        default:
            return true;
    }
}

MatrixType or_dense(MatrixType t, DataType dt) { return can_make(t, dt) ? t : MatrixType::DENSE_FLOAT; }

}  // namespace

MatrixType resolve_result_structure(promotion::BinaryOp op, const MatrixBase& a, const MatrixBase& b,
                                    DataType dt) {
    using promotion::BinaryOp;
    if (a.rows() != a.cols() || b.rows() != b.cols()) return MatrixType::DENSE_FLOAT;

    if (op == BinaryOp::Add || op == BinaryOp::Subtract) {
        // Non-zero pattern is the union of the operands' patterns.
        if (is_diagonal_like(a) && is_diagonal_like(b)) return or_dense(MatrixType::DIAGONAL, dt);
        if (is_symmetric_like(a) && is_symmetric_like(b)) return or_dense(MatrixType::SYMMETRIC, dt);
        if (is_antisymmetric(a) && is_antisymmetric(b)) return or_dense(MatrixType::ANTISYMMETRIC, dt);
        // Upper + upper stays upper. Upper + lower (e.g. C + C.T) is a full matrix.
        if (is_strict_upper(a) && is_strict_upper(b)) return or_dense(MatrixType::TRIANGULAR_FLOAT, dt);
        return MatrixType::DENSE_FLOAT;
    }
    if (op == BinaryOp::ElementwiseMultiply) {
        // Non-zero pattern is the intersection.
        if (is_diagonal_like(a) || is_diagonal_like(b)) return or_dense(MatrixType::DIAGONAL, dt);
        const bool upper_a = is_strict_upper(a), upper_b = is_strict_upper(b);
        const auto lower = [](const MatrixBase& m) {
            const MatrixType t = m.get_matrix_type();
            return (t == MatrixType::TRIANGULAR_FLOAT || t == MatrixType::CAUSAL) && m.is_transposed();
        };
        if ((upper_a && !lower(b)) || (upper_b && !lower(a))) return or_dense(MatrixType::TRIANGULAR_FLOAT, dt);
        if ((is_symmetric_like(a) && is_symmetric_like(b)) || (is_antisymmetric(a) && is_antisymmetric(b))) {
            return or_dense(MatrixType::SYMMETRIC, dt);
        }
        if ((is_symmetric_like(a) && is_antisymmetric(b)) || (is_antisymmetric(a) && is_symmetric_like(b))) {
            return or_dense(MatrixType::ANTISYMMETRIC, dt);
        }
        return MatrixType::DENSE_FLOAT;
    }
    // Division (0/0 = NaN breaks every zero pattern) and anything else: dense.
    return MatrixType::DENSE_FLOAT;
}

}  // namespace pycauset::structure
