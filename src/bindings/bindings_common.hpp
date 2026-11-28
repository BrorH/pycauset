#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

namespace py = pybind11;

void bind_core_classes(py::module_& m);
void bind_matrix_classes(py::module_& m);
void bind_vector_classes(py::module_& m);
void bind_math_ops(py::module_& m);
void bind_causet_classes(py::module_& m);

// ---------------------------------------------------------------------------
// One Python class per structure (review/RESTRUCTURE.md 3.1 item 4).
//
// pybind11 normally exposes an object under its most-derived C++ type
// (DenseMatrix<int8_t>, TriangularMatrix<bool>, ...). Python instead registers one class per
// *structure* (DenseMatrix, TriangularMatrix, ...; see bind_matrix.cpp), bound to the
// structure bases in MatrixBase.hpp / SymmetricMatrix.hpp / VectorBase.hpp. This hook maps
// every object, whatever static type a binding returns, to its structure class using its
// Structure tag. That makes "a result arrives as a bare MatrixBase" (F-032) impossible:
// every instantiation of every dtype has a class.
//
// It must be visible in every translation unit that converts matrices or vectors to Python,
// which is why it lives in this header.
// ---------------------------------------------------------------------------
#include <typeinfo>
#include <type_traits>

#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/matrix/SymmetricMatrix.hpp"
#include "pycauset/vector/VectorBase.hpp"
#include "pycauset/vector/UnitVector.hpp"
#include "pycauset/core/DTypeDispatch.hpp"

namespace pycauset_bindings {

template <typename Base, typename Obj>
inline const void* as_structure(const Obj* src, const std::type_info*& type) {
    if (const auto* b = dynamic_cast<const Base*>(src)) {
        type = &typeid(Base);
        return static_cast<const void*>(b);
    }
    type = &typeid(*src);  // not expected: fall back to the dynamic type
    return dynamic_cast<const void*>(src);
}

inline const void* structure_class_of(const pycauset::MatrixBase* src, const std::type_info*& type) {
    using S = pycauset::Structure;
    if (src == nullptr) {
        type = nullptr;
        return nullptr;
    }
    switch (src->get_structure()) {
        case S::DENSE: return as_structure<pycauset::DenseMatrixBase>(src, type);
        case S::TRIANGULAR: return as_structure<pycauset::TriangularMatrixBase>(src, type);
        case S::DIAGONAL: return as_structure<pycauset::DiagonalMatrixBase>(src, type);
        case S::IDENTITY: return as_structure<pycauset::IdentityMatrixBase>(src, type);
        case S::SYMMETRIC: return as_structure<pycauset::SymmetricMatrixBase>(src, type);
        case S::ANTISYMMETRIC: return as_structure<pycauset::AntiSymmetricMatrixBase>(src, type);
        case S::HERMITIAN: return as_structure<pycauset::HermitianMatrixBase>(src, type);
        case S::SKEW_HERMITIAN: return as_structure<pycauset::SkewHermitianMatrixBase>(src, type);
        default:
            type = &typeid(*src);
            return dynamic_cast<const void*>(src);
    }
}

inline const void* structure_class_of(const pycauset::VectorBase* src, const std::type_info*& type) {
    using S = pycauset::Structure;
    if (src == nullptr) {
        type = nullptr;
        return nullptr;
    }
    if (src->get_structure() == S::UNIT_VECTOR) return as_structure<pycauset::UnitVector>(src, type);
    return as_structure<pycauset::DenseVectorBase>(src, type);
}

// NumPy export of any matrix (every class, dtype, view): bind_matrix.cpp.
py::object export_matrix_numpy(const pycauset::MatrixBase& m);

}  // namespace pycauset_bindings

namespace pybind11 {

template <typename itype>
struct polymorphic_type_hook<itype, detail::enable_if_t<std::is_base_of<pycauset::MatrixBase, itype>::value>> {
    static const void* get(const itype* src, const std::type_info*& type) {
        return pycauset_bindings::structure_class_of(static_cast<const pycauset::MatrixBase*>(src), type);
    }
};

template <typename itype>
struct polymorphic_type_hook<itype, detail::enable_if_t<std::is_base_of<pycauset::VectorBase, itype>::value>> {
    static const void* get(const itype* src, const std::type_info*& type) {
        return pycauset_bindings::structure_class_of(static_cast<const pycauset::VectorBase*>(src), type);
    }
};

}  // namespace pybind11

// A dimension argument: NumPy's ValueError for a negative one (uint64_t parameters gave pybind's
// "incompatible function arguments" TypeError, review 3.7 C-6).
inline uint64_t dim_arg(int64_t n) {
    if (n < 0) throw py::value_error("negative dimensions are not allowed");
    return static_cast<uint64_t>(n);
}

// An element index along an axis of length `dim`: negative counts from the end, as A[i, j].
inline uint64_t index_arg(int64_t i, uint64_t dim) {
    if (i < 0) i += static_cast<int64_t>(dim);
    if (i < 0 || static_cast<uint64_t>(i) >= dim) throw py::index_error("index out of bounds");
    return static_cast<uint64_t>(i);
}

// A user's dtype argument (a name, a NumPy code, np.float32, np.dtype(...), int, ...) through
// the one parser, pycauset._internal.dtypes.normalize_dtype (review 3.3 C-9: the native
// constructors had their own table: pc.DenseMatrix(2, dtype="bit") raised, pc.zeros accepted
// it). Leaked statics: no Python object is destroyed after the interpreter finalizes.
inline pycauset::DataType dtype_arg(const py::handle& dtype) {
    static py::object* normalize = new py::object(py::module_::import("pycauset._internal.dtypes").attr("normalize_dtype"));
    static py::object* np = new py::object(py::module_::import("numpy"));
    py::object name = (*normalize)(dtype, py::arg("np_module") = *np);
    if (name.is_none()) throw py::type_error("unsupported dtype: " + py::repr(dtype).cast<std::string>());
    return pycauset::parse_dtype_name(name.cast<std::string>());
}
