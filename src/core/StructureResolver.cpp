// Result-structure rules for element-wise binary ops. See StructureResolver.hpp.
#include "pycauset/core/StructureResolver.hpp"

#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"

namespace pycauset::structure {

namespace {

bool is_diagonal_like(const MatrixBase& m) {
    const Structure t = m.get_structure();
    return t == Structure::DIAGONAL || t == Structure::IDENTITY;
}
bool is_symmetric_like(const MatrixBase& m) {
    return m.get_structure() == Structure::SYMMETRIC || is_diagonal_like(m);
}
bool is_antisymmetric(const MatrixBase& m) { return m.get_structure() == Structure::ANTISYMMETRIC; }

// Strictly upper triangular storage that is not viewed transposed (i.e. really upper).
bool is_strict_upper(const MatrixBase& m) {
    const Structure t = m.get_structure();
    if (t != Structure::TRIANGULAR) return false;
    if (m.is_transposed()) return false;
    const auto* tri = dynamic_cast<const TriangularMatrixBase*>(&m);
    return tri != nullptr && !tri->has_diagonal();
}

// Structured result classes that exist (and are registered with Python) per dtype.
bool can_make(Structure t, DataType dt) {
    // Combinations the factory instantiates and the element-wise kernels can write. Every one
    // has a Python class since the structure classes (F-032), so results keep their structure.
    // No BIT results (the kernels can't write packed bits; bit sums promote anyway), no complex
    // structured storage.
    const bool real_non_bit = dt != DataType::BIT && dt != DataType::COMPLEX_FLOAT16 &&
                              dt != DataType::COMPLEX_FLOAT32 && dt != DataType::COMPLEX_FLOAT64 &&
                              dt != DataType::UNKNOWN;
    switch (t) {
        case Structure::TRIANGULAR:
        case Structure::SYMMETRIC:
        case Structure::ANTISYMMETRIC:
            return real_non_bit;
        case Structure::DIAGONAL:
            return real_non_bit && dt != DataType::FLOAT16 && dt != DataType::FLOAT32;
        default:
            return true;
    }
}

Structure or_dense(Structure t, DataType dt) { return can_make(t, dt) ? t : Structure::DENSE; }

}  // namespace

Structure resolve_result_structure(promotion::BinaryOp op, const MatrixBase& a, const MatrixBase& b,
                                    DataType dt) {
    using promotion::BinaryOp;
    if (a.rows() != a.cols() || b.rows() != b.cols()) return Structure::DENSE;

    if (op == BinaryOp::Add || op == BinaryOp::Subtract) {
        // Non-zero pattern is the union of the operands' patterns.
        if (is_diagonal_like(a) && is_diagonal_like(b)) return or_dense(Structure::DIAGONAL, dt);
        if (is_symmetric_like(a) && is_symmetric_like(b)) return or_dense(Structure::SYMMETRIC, dt);
        if (is_antisymmetric(a) && is_antisymmetric(b)) return or_dense(Structure::ANTISYMMETRIC, dt);
        // Upper + upper stays upper. Upper + lower (e.g. C + C.T) is a full matrix.
        if (is_strict_upper(a) && is_strict_upper(b)) return or_dense(Structure::TRIANGULAR, dt);
        return Structure::DENSE;
    }
    if (op == BinaryOp::ElementwiseMultiply) {
        // Non-zero pattern is the intersection.
        if (is_diagonal_like(a) || is_diagonal_like(b)) return or_dense(Structure::DIAGONAL, dt);
        const bool upper_a = is_strict_upper(a), upper_b = is_strict_upper(b);
        const auto lower = [](const MatrixBase& m) {
            const Structure t = m.get_structure();
            return (t == Structure::TRIANGULAR) && m.is_transposed();
        };
        if ((upper_a && !lower(b)) || (upper_b && !lower(a))) return or_dense(Structure::TRIANGULAR, dt);
        if ((is_symmetric_like(a) && is_symmetric_like(b)) || (is_antisymmetric(a) && is_antisymmetric(b))) {
            return or_dense(Structure::SYMMETRIC, dt);
        }
        if ((is_symmetric_like(a) && is_antisymmetric(b)) || (is_antisymmetric(a) && is_symmetric_like(b))) {
            return or_dense(Structure::ANTISYMMETRIC, dt);
        }
        return Structure::DENSE;
    }
    // Division (0/0 = NaN breaks every zero pattern) and anything else: dense.
    return Structure::DENSE;
}

}  // namespace pycauset::structure
