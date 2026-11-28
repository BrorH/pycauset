#include "pycauset/math/VectorOps.hpp"

#include <stdexcept>
#include <string>

#include "pycauset/core/DTypeDispatch.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/matrix/ComplexFloat16Matrix.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/vector/ComplexFloat16Vector.hpp"
#include "pycauset/vector/DenseVector.hpp"
#include "pycauset/vector/UnitVector.hpp"
#include "pycauset/core/ObjectFactory.hpp"

namespace pycauset {

std::unique_ptr<MatrixBase> as_row_matrix(const VectorBase& v) {
    const uint64_t n = v.size();
    if (v.get_structure() == Structure::UNIT_VECTOR) {
        // One non-zero, already through the view rule; zeros stay zero under any view.
        const auto& u = static_cast<const UnitVector&>(v);
        auto out = ObjectFactory::create_matrix(1, n, v.get_data_type(), Structure::DENSE, "");
        const std::complex<double> z = u.active_value();
        visit_matrix(*out, [&](auto& typed) {
            using E = std::decay_t<decltype(typed.get(0, 0))>;
            const uint64_t k = u.get_active_index();
            if constexpr (std::is_same_v<E, bool>) typed.set(0, k, z.real() != 0.0);
            else if constexpr (std::is_same_v<E, std::complex<double>>) typed.set(0, k, z);  // incl. complex_float16
            else if constexpr (std::is_same_v<E, std::complex<float>>) typed.set(0, k, E(z));
            else if constexpr (std::is_integral_v<E>) typed.set(0, k, static_cast<E>(static_cast<int64_t>(z.real())));
            else typed.set(0, k, static_cast<E>(z.real()));
        });
        return out;
    }
    return visit_dtype(v.get_data_type(), [&](auto tag) -> std::unique_ptr<MatrixBase> {
        using T = typename decltype(tag)::type;
        using M = typename DenseMatrixOf<T>::type;
        const auto& mapper = static_cast<const typename DenseVectorOf<T>::type&>(v).shared_mapper();
        std::unique_ptr<MatrixBase> out;
        if constexpr (std::is_same_v<T, bool>) {
            // A bit-matrix row is padded to 64 bytes; vectors are too (DenseVector<bool>), but
            // check, so a short buffer is copied instead of over-read.
            const uint64_t row_bytes = (((n + 63) / 64) * 8 + 63) & ~63ULL;
            if (!mapper || mapper->get_data_size() < row_bytes) {
                auto copy = std::make_unique<DenseMatrix<bool>>(1, n, std::string());
                for (uint64_t i = 0; i < n; ++i) copy->set(0, i, v.get_element_as_double(i) != 0.0);
                return copy;
            }
        }
        out = std::make_unique<M>(1, n, mapper);
        out->set_scalar(v.get_scalar());
        out->set_conjugated(v.is_conjugated());
        return out;
    });
}

std::unique_ptr<VectorBase> as_vector(std::unique_ptr<MatrixBase> m, bool transposed) {
    const uint64_t n = m->rows() * m->cols();
    if (m->rows() != 1 && m->cols() != 1) throw std::logic_error("as_vector: result is not 1 x n");
    // Row storage maps onto a vector. A float / complex view (factor, conjugation: -v, 2.5 * v,
    // v.conj()) stays a view, as for matrices; an integer view is written out (vectors read
    // integer views per element in double, without NumPy's wrap).
    const DataType dt = m->get_data_type();
    const bool float_like = is_real_float_dtype(dt) ||
                            dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64;
    const bool row = m->get_structure() == Structure::DENSE && m->rows() == 1 && !m->is_transposed() && !m->is_subview();
    const bool unit_view = m->get_scalar() == std::complex<double>(1.0, 0.0) && !m->is_conjugated();
    const bool plain = row && (unit_view || float_like);
    if (!plain) {
        std::unique_ptr<MatrixBase> row = cast_to(*m, m->get_data_type());  // dense, factor 1
        if (row->rows() != 1) row = cast_to(*row->transpose(), row->get_data_type());
        m = std::move(row);
    }
    std::unique_ptr<VectorBase> out = visit_dtype(m->get_data_type(), [&](auto tag) -> std::unique_ptr<VectorBase> {
        using T = typename decltype(tag)::type;
        return std::make_unique<typename DenseVectorOf<T>::type>(n, m->shared_mapper());
    });
    out->set_scalar(m->get_scalar());
    out->set_conjugated(m->is_conjugated());
    out->set_transposed(transposed);
    return out;
}

std::unique_ptr<VectorBase> scalar_op(const VectorBase& v, ScalarOpKind op, const ScalarArg& s) {
    auto row = as_row_matrix(v);
    return as_vector(scalar_op(*row, op, s), v.is_transposed());
}

std::unique_ptr<VectorBase> negate(const VectorBase& v) {
    auto row = as_row_matrix(v);
    return as_vector(negate(*row), v.is_transposed());
}

std::unique_ptr<VectorBase> binary_op(const VectorBase& a, const VectorBase& b, VectorBinaryOp op) {
    // The 1 x n rows broadcast as matrices do: a length-1 vector with any (NumPy; it raised,
    // review 3.6 C-7); other differing lengths raise there.
    auto ra = as_row_matrix(a);
    auto rb = as_row_matrix(b);
    std::unique_ptr<MatrixBase> r;
    switch (op) {
        case VectorBinaryOp::Add: r = add(*ra, *rb); break;
        case VectorBinaryOp::Sub: r = subtract(*ra, *rb); break;
        case VectorBinaryOp::Mul: r = elementwise_multiply(*ra, *rb); break;
        case VectorBinaryOp::Div: r = elementwise_divide(*ra, *rb); break;
    }
    return as_vector(std::move(r), a.is_transposed());
}

}  // namespace pycauset
