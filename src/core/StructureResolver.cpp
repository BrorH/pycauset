// Result-structure rules for element-wise binary ops. See StructureResolver.hpp.
#include "pycauset/core/StructureResolver.hpp"

#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/core/DTypeDispatch.hpp"
#include "pycauset/math/ScalarOps.hpp"

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

// An upper triangle as stored (not viewed transposed), with or without its diagonal.
bool is_upper(const MatrixBase& m) {
    return m.get_structure() == Structure::TRIANGULAR && !m.is_transposed() &&
           dynamic_cast<const TriangularMatrixBase*>(&m) != nullptr;
}
bool stores_diagonal(const MatrixBase& m) {
    const auto* tri = dynamic_cast<const TriangularMatrixBase*>(&m);
    return tri == nullptr || tri->has_diagonal();  // non-triangular operands may be non-zero there
}
bool is_lower(const MatrixBase& m) { return m.get_structure() == Structure::TRIANGULAR && m.is_transposed(); }

ResultStructure triangular_or_dense(bool has_diagonal, DataType dt) {
    if (!can_make(Structure::TRIANGULAR, dt)) return {};
    return {Structure::TRIANGULAR, has_diagonal};
}
ResultStructure of(Structure t, DataType dt) { return {or_dense(t, dt), false}; }

// Both operands are stored as structure t in the result dtype, with no view: the packed
// element-wise kernel computes the result over the stored elements. That is the only way a
// complex structured result is written (the per-element structured paths are real-only), so
// the Hermitian rules below require it.
bool packed_same(const MatrixBase& a, const MatrixBase& b, Structure t, DataType dt) {
    return dt != DataType::COMPLEX_FLOAT16 &&  // its own storage class and kernel
           a.get_structure() == t && b.get_structure() == t && a.get_data_type() == dt &&
           b.get_data_type() == dt && is_plain(a) && is_plain(b) && is_offered(t, dt);
}

}  // namespace

ResultStructure resolve_result_structure(promotion::BinaryOp op, const MatrixBase& a, const MatrixBase& b,
                                          DataType dt) {
    using promotion::BinaryOp;
    // Structured results only for equal square shapes: a 1 x 1 operand is square too but
    // broadcasts, and the n x n operand's structure does not survive that (Diagonal(3) +
    // Diagonal(1) lost its off-diagonal values, review 3.3 C-1).
    if (a.rows() != a.cols() || b.rows() != b.cols() || a.rows() != b.rows()) return {};

    if (op == BinaryOp::Add || op == BinaryOp::Subtract) {
        // A + A^T / A - A^T (e.g. the comparability matrix C + C^T, Pauli-Jordan-like C - C^T).
        if (is_transpose_pair(a, b)) {
            if (op == BinaryOp::Add) return of(Structure::SYMMETRIC, dt);
            if (diagonal_is_finite(a)) return of(Structure::ANTISYMMETRIC, dt);
        }
        // Non-zero pattern is the union of the operands' patterns.
        if (is_diagonal_like(a) && is_diagonal_like(b)) return of(Structure::DIAGONAL, dt);
        if (is_symmetric_like(a) && is_symmetric_like(b)) return of(Structure::SYMMETRIC, dt);
        if (is_antisymmetric(a) && is_antisymmetric(b)) return of(Structure::ANTISYMMETRIC, dt);
        // Upper + upper stays upper (with a diagonal if either has one); upper + diagonal is
        // upper with a diagonal. Upper + lower (e.g. C + C.T) is a full matrix.
        if (is_upper(a) && is_upper(b)) return triangular_or_dense(stores_diagonal(a) || stores_diagonal(b), dt);
        if ((is_upper(a) && is_diagonal_like(b)) || (is_diagonal_like(a) && is_upper(b))) {
            return triangular_or_dense(true, dt);
        }
        // Hermitian +- Hermitian is Hermitian; skew-Hermitian +- skew-Hermitian is skew-Hermitian.
        for (Structure t : {Structure::HERMITIAN, Structure::SKEW_HERMITIAN}) {
            if (packed_same(a, b, t, dt)) return {t, false};
        }
        return {};
    }
    if (op == BinaryOp::ElementwiseMultiply) {
        // Non-zero pattern is the intersection.
        if (is_diagonal_like(a) || is_diagonal_like(b)) return of(Structure::DIAGONAL, dt);
        // Upper * (anything but lower) is upper; the diagonal survives where both can be
        // non-zero on it.
        if ((is_upper(a) && !is_lower(b)) || (is_upper(b) && !is_lower(a))) {
            return triangular_or_dense(stores_diagonal(a) && stores_diagonal(b), dt);
        }
        if ((is_symmetric_like(a) && is_symmetric_like(b)) || (is_antisymmetric(a) && is_antisymmetric(b))) {
            return of(Structure::SYMMETRIC, dt);
        }
        if ((is_symmetric_like(a) && is_antisymmetric(b)) || (is_antisymmetric(a) && is_symmetric_like(b))) {
            return of(Structure::ANTISYMMETRIC, dt);
        }
        // The element-wise product of Hermitian matrices is Hermitian (conj(x) conj(y) = conj(xy)).
        if (packed_same(a, b, Structure::HERMITIAN, dt)) return {Structure::HERMITIAN, false};
        return {};
    }
    // Division (0/0 = NaN breaks every zero pattern) and anything else: dense.
    return {};
}

}  // namespace pycauset::structure
