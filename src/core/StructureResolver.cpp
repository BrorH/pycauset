// Result-structure rules for element-wise binary ops. See StructureResolver.hpp.
#include "pycauset/core/StructureResolver.hpp"

#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/core/DTypeDispatch.hpp"

#include <cmath>

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

// B is the transpose view of A: same storage, opposite transpose flag, same factor and
// conjugation, full square span. Then A + B is exactly symmetric and A - B exactly
// antisymmetric (LAZY.md L2a).
bool is_transpose_pair(const MatrixBase& a, const MatrixBase& b) {
    if (a.get_structure() != b.get_structure()) return false;
    if (!a.shared_mapper() || a.shared_mapper() != b.shared_mapper()) return false;
    if (a.is_transposed() == b.is_transposed()) return false;
    if (a.get_scalar() != b.get_scalar() || a.is_conjugated() != b.is_conjugated()) return false;
    if (a.is_subview() || b.is_subview()) return false;
    return a.rows() == a.cols() && a.rows() == a.base_rows() && a.cols() == a.base_cols();
}

// A - A^T has an exactly zero diagonal unless A's diagonal holds inf/NaN (inf - inf = NaN),
// which packed antisymmetric storage can't represent. Integers: always zero. O(n) check.
bool diagonal_is_finite(const MatrixBase& a) {
    const DataType dt = a.get_data_type();
    if (dt != DataType::FLOAT16 && dt != DataType::FLOAT32 && dt != DataType::FLOAT64) return true;
    for (uint64_t i = 0; i < a.rows(); ++i) {
        if (!std::isfinite(a.get_element_as_double(i, i))) return false;
    }
    return true;
}

// Can an element-wise result keep structure t in dtype dt? The pair must exist (the catalogue,
// DTypeDispatch.hpp) and the element-wise kernels must be able to write it: they write real,
// non-bit structured results (complex structured results: review N.7).
bool can_make(Structure t, DataType dt) {
    if (t == Structure::DENSE) return true;
    const bool real_non_bit = dt != DataType::BIT && dt != DataType::COMPLEX_FLOAT16 &&
                              dt != DataType::COMPLEX_FLOAT32 && dt != DataType::COMPLEX_FLOAT64 &&
                              dt != DataType::UNKNOWN;
    return real_non_bit && is_offered(t, dt);
}

Structure or_dense(Structure t, DataType dt) { return can_make(t, dt) ? t : Structure::DENSE; }

}  // namespace

Structure resolve_result_structure(promotion::BinaryOp op, const MatrixBase& a, const MatrixBase& b,
                                    DataType dt) {
    using promotion::BinaryOp;
    if (a.rows() != a.cols() || b.rows() != b.cols()) return Structure::DENSE;

    if (op == BinaryOp::Add || op == BinaryOp::Subtract) {
        // A + A^T / A - A^T (e.g. the comparability matrix C + C^T, Pauli-Jordan-like C - C^T).
        if (is_transpose_pair(a, b)) {
            if (op == BinaryOp::Add) return or_dense(Structure::SYMMETRIC, dt);
            if (diagonal_is_finite(a)) return or_dense(Structure::ANTISYMMETRIC, dt);
        }
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
