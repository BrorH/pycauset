#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

namespace py = pybind11;

void bind_core_classes(py::module_& m);
void bind_op_registry(py::module_& m);
void bind_matrix_classes(py::module_& m);
void bind_vector_classes(py::module_& m);
void bind_causet_classes(py::module_& m);
void bind_math_ops(py::module_& m);

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
