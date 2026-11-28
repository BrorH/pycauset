#include "pycauset/math/Compare.hpp"
#include "bindings_common.hpp"
#include "element_conversion.hpp"
#include "scalar_args.hpp"
#include "pycauset/math/VectorOps.hpp"
#include "pycauset/core/DTypeDispatch.hpp"
#include "pycauset/core/ScalarUtils.hpp"
#include "pycauset/core/ObjectFactory.hpp"

#include "binding_warnings.hpp"

#include "pycauset/core/PromotionResolver.hpp"
#include "pycauset/core/Float16.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/vector/VectorBase.hpp"
#include "pycauset/vector/DenseVector.hpp"
#include "pycauset/vector/UnitVector.hpp"
#include "pycauset/math/Bitwise.hpp"
#include "pycauset/math/ViewValue.hpp"
#include "pycauset/vector/ComplexFloat16Vector.hpp"

#include "pycauset/matrix/MatrixBase.hpp"

#include "numpy_export_guard.hpp"
#include "matrix_indexing.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <limits>

#include "numpy_import.hpp"

using namespace pycauset;
using namespace pycauset_bindings::numpy_import;

namespace {

struct DenseVectorBufferState {
    Py_ssize_t shape[2];
    Py_ssize_t strides[2];
};

inline bool u64_to_py_ssize(uint64_t v, Py_ssize_t* out) {
    if (v > static_cast<uint64_t>(PY_SSIZE_T_MAX)) {
        return false;
    }
    *out = static_cast<Py_ssize_t>(v);
    return true;
}


// Inner product of two vectors (orientation ignored, no conjugation, like np.dot).
// The engine's one dot (math/LinearAlgebra.cpp `dot`: matmul's rule), its 1 x 1 result read as
// a Python scalar.
inline py::object vector_dot(const VectorBase& a, const VectorBase& b) {
    const auto r = translate_invalid_argument([&] { return pycauset::dot(a, b); });
    return pycauset_bindings::indexing::element_get(*r, 0, 0);
}

// ---------------------------------------------------------------------------------------
// Generic (all dtypes) element access for DenseVector / UnitVector. Mirrors the matrix
// helpers in bind_matrix.cpp: floats and complex read through the view-aware readers,
// integers and bits exactly; writes fold a scale/conjugation view first (F-045).
// ---------------------------------------------------------------------------------------
template <typename V>
struct vec_element { using type = std::decay_t<decltype(std::declval<const V&>().get(uint64_t{0}))>; };
template <>
struct vec_element<UnitVector> { using type = double; };  // read via get_element_as_double
template <typename V>
using vec_element_t = typename vec_element<V>::type;

inline py::object vector_element_get(const VectorBase& v, uint64_t i) {
    const DataType dt = v.get_data_type();
    if (is_complex_dtype(dt)) {
        return py::cast(v.get_element_as_complex(i));
    }
    if (is_real_float_dtype(dt)) {
        return py::float_(v.get_element_as_double(i));
    }
    if (v.get_scalar() != std::complex<double>(1.0, 0.0)) {
        throw std::logic_error("integer vector carries a scale factor");
    }
    return pycauset::visit_vector(v, [&](const auto& typed) -> py::object {
        using V = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<V, UnitVector>) {
            const double x = typed.get_element_as_double(i);  // integer / bit dtypes: exact small values
            if (dt == DataType::BIT) return py::bool_(x != 0.0);
            return py::int_(static_cast<int64_t>(x));
        } else {
        using E = vec_element_t<V>;
        const E x = typed.get(i);
        if constexpr (std::is_same_v<E, bool>) return py::bool_(x);
        else if constexpr (std::is_integral_v<E>) return py::int_(x);
        else throw std::logic_error("unreachable: non-integer dtype in the exact read path");
        }
    });
}

inline void vector_fold_value_view(VectorBase& v) {
    const std::complex<double> s = v.get_scalar();
    const DataType dt = v.get_data_type();
    const bool conj = v.is_conjugated() && is_complex_dtype(dt);
    if (s == std::complex<double>(1.0, 0.0) && !conj) {
        v.set_conjugated(false);  // conj of real values is a no-op flag (review 3.2 C-13)
        return;
    }
    pycauset::visit_vector(v, [&](auto& typed) {
        using V = std::decay_t<decltype(typed)>;
        using E = vec_element_t<V>;
        if constexpr (std::is_same_v<V, UnitVector>) {
            throw py::type_error("UnitVector is read-only");
        } else if constexpr (std::is_integral_v<E>) {
            throw std::logic_error("integer vector carries a scale factor");
        } else {
            for (uint64_t i = 0; i < typed.size(); ++i) {
                const std::complex<double> x = pycauset::view_value(typed.get(i), s, conj);  // the one view rule
                if constexpr (std::is_same_v<E, std::complex<double>> || std::is_same_v<E, std::complex<float>>) {
                    typed.set(i, static_cast<E>(x));
                } else if constexpr (std::is_same_v<V, ComplexFloat16Vector>) {
                    typed.set(i, x);
                } else {
                    typed.set(i, static_cast<E>(x.real()));
                }
            }
        }
    });
    v.set_scalar(1.0);
    v.set_conjugated(false);
}

inline void vector_element_set(VectorBase& v, uint64_t i, const py::object& value) {
    vector_fold_value_view(v);
    pycauset::visit_vector(v, [&](auto& typed) {
        using V = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<V, UnitVector>) {
            throw py::type_error("UnitVector is read-only");
        } else {
            typed.set(i, pycauset_bindings::py_to_element<vec_element_t<V>>(value));
        }
    });
}

inline uint64_t vector_index(const VectorBase& v, py::ssize_t i) {
    const auto n = static_cast<py::ssize_t>(v.size());
    if (i < 0) i += n;
    if (i < 0 || i >= n) throw py::index_error("vector index out of range");
    return static_cast<uint64_t>(i);
}

// Vector arithmetic: `a op other` (or `other op a` when reflected) for another vector, a 1-D
// NumPy array or a scalar. Everything runs through the matrix machinery (math/VectorOps.hpp),
// so result dtypes follow NumPy and integers are exact. Anything else: NotImplemented.
// b == nullptr: ~a.
inline std::shared_ptr<VectorBase> vector_bitwise(const VectorBase& a, const VectorBase* b, pycauset::BitwiseOp op) {
    try {
        auto ra = pycauset::as_row_matrix(a);
        std::unique_ptr<MatrixBase> out;
        if (b == nullptr) {
            out = pycauset::bitwise_not(*ra);
        } else {
            auto rb = pycauset::as_row_matrix(*b);
            out = pycauset::bitwise(*ra, *rb, op);
        }
        return std::shared_ptr<VectorBase>(pycauset::as_vector(std::move(out), a.is_transposed()).release());
    } catch (const std::invalid_argument& e) {
        throw py::value_error(e.what());
    }  // pycauset::dtype_error is a TypeError (translated globally)
}

inline py::object vector_arith(const VectorBase& a, const py::object& other, Arith op, bool reflected) {
    std::shared_ptr<VectorBase> b;
    if (py::isinstance<VectorBase>(other)) {
        b = other.cast<std::shared_ptr<VectorBase>>();
    } else if (py::isinstance<py::array>(other) && other.cast<py::array>().ndim() == 1) {
        b = vector_from_numpy(other.cast<py::array>());
    }
    if (b) {
        const VectorBase& x = reflected ? *b : a;
        const VectorBase& y = reflected ? a : *b;
        if (op == Arith::Add || op == Arith::Sub) {
            maybe_warn_bit_promotes_to_int32(op == Arith::Add ? "add" : "subtract",
                                             op == Arith::Add ? promotion::BinaryOp::Add : promotion::BinaryOp::Subtract,
                                             x.get_data_type(), y.get_data_type());
        }
        const VectorBinaryOp vop = op == Arith::Add ? VectorBinaryOp::Add
                                 : op == Arith::Sub ? VectorBinaryOp::Sub
                                 : op == Arith::Mul ? VectorBinaryOp::Mul : VectorBinaryOp::Div;
        return py::cast(std::shared_ptr<VectorBase>(pycauset::binary_op(x, y, vop).release()));
    }
    ScalarArg s;
    if (!to_scalar_arg(other, s)) return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    return py::cast(std::shared_ptr<VectorBase>(pycauset::scalar_op(a, scalar_kind(op, reflected), s).release()));
}

// v <op> other for a vector: element-wise, a bool vector (a bool matrix when broadcasting
// against a matrix makes more than one row). NotImplemented for operands that are not
// matrix-like.
inline py::object vector_compare(const VectorBase& v, const py::object& other, pycauset::CompareOp op) {
    const auto b = compare_operand(other);
    if (!b) return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    auto row = pycauset::as_row_matrix(v);
    auto out = translate_invalid_argument([&] { return pycauset::compare(*row, *b, op); });
    if (out->rows() != 1) return py::cast(std::shared_ptr<MatrixBase>(out.release()));
    return py::cast(std::shared_ptr<VectorBase>(pycauset::as_vector(std::move(out), v.is_transposed()).release()));
}

} // namespace

void bind_vector_classes(py::module_& m) {
    py::class_<VectorBase, std::shared_ptr<VectorBase>>(m, "VectorBase")
        .def_property_readonly("backing_file", &VectorBase::get_backing_file)
        .def("get_backing_file", &VectorBase::get_backing_file)
        .def_property_readonly("is_temporary", &VectorBase::is_temporary)
        .def("set_temporary", &VectorBase::set_temporary)
        .def("close", &VectorBase::close)
        .def_property_readonly("size", &VectorBase::size, "Number of elements, as NumPy's ndarray.size.")
        .def_property_readonly("ndim", [](const VectorBase& v) { return v.is_transposed() ? 2 : 1; },
                               "Number of dimensions (len(shape): 1, or 2 for a transposed (1, n) vector).")
        .def("__len__", &VectorBase::size)
        .def("__bool__", [](const VectorBase& v) {  // NumPy's truth value (see MatrixBase)
            if (v.size() != 1) {
                throw py::value_error("The truth value of a vector with more than one element is ambiguous");
            }
            return vector_element_get(v, 0).cast<bool>();
        })
        .def("__eq__", [](const VectorBase& v, const py::object& o) { return vector_compare(v, o, pycauset::CompareOp::Eq); }, py::is_operator())
        .def("__ne__", [](const VectorBase& v, const py::object& o) { return vector_compare(v, o, pycauset::CompareOp::Ne); }, py::is_operator())
        .def("__lt__", [](const VectorBase& v, const py::object& o) { return vector_compare(v, o, pycauset::CompareOp::Lt); }, py::is_operator())
        .def("__le__", [](const VectorBase& v, const py::object& o) { return vector_compare(v, o, pycauset::CompareOp::Le); }, py::is_operator())
        .def("__gt__", [](const VectorBase& v, const py::object& o) { return vector_compare(v, o, pycauset::CompareOp::Gt); }, py::is_operator())
        .def("__ge__", [](const VectorBase& v, const py::object& o) { return vector_compare(v, o, pycauset::CompareOp::Ge); }, py::is_operator())
        .def_property("seed", &VectorBase::get_seed, &VectorBase::set_seed)
        .def_property("scalar", &VectorBase::get_scalar, &VectorBase::set_scalar)
        .def("get_scalar", &VectorBase::get_scalar)
        .def("set_scalar", &VectorBase::set_scalar)
        .def("is_transposed", &VectorBase::is_transposed)
        .def("set_transposed", &VectorBase::set_transposed)
        .def("is_conjugated", &VectorBase::is_conjugated)
        .def("set_conjugated", &VectorBase::set_conjugated)
        .def("get_element_as_double", &VectorBase::get_element_as_double)
        .def("get_element_as_complex", &VectorBase::get_element_as_complex)
        .def_property_readonly(
            "dtype",
            [](const VectorBase& v) { return std::string(pycauset::dtype_name(v.get_data_type())); },
            "Element type, e.g. 'float64', 'int32', 'bool' (bit), 'complex_float64'.")
        .def("get", [](const VectorBase& v, uint64_t i) { return vector_element_get(v, i); }, py::arg("i"))
        .def("set", [](VectorBase& v, uint64_t i, const py::object& value) { vector_element_set(v, i, value); },
             py::arg("i"), py::arg("value"))
        .def("__getitem__", [](const VectorBase& v, py::ssize_t i) { return vector_element_get(v, vector_index(v, i)); })
        .def("__setitem__", [](VectorBase& v, py::ssize_t i, const py::object& value) {
            vector_element_set(v, vector_index(v, i), value);
        })
        .def("fill", [](VectorBase& v, const py::object& value) {
            for (uint64_t i = 0; i < v.size(); ++i) vector_element_set(v, i, value);
        }, py::arg("value"))
        .def(
            "__repr__",
            [](py::object self) {
                const auto& v = self.cast<const VectorBase&>();
                std::string type_name = py::str(py::type::of(self).attr("__name__"));

                std::ostringstream oss;
                if (v.is_transposed()) {
                    oss << "<" << type_name << " shape=(1, " << v.size() << ") dtype=" << pycauset::dtype_name(v.get_data_type()) << " transposed=True>";
                } else {
                    oss << "<" << type_name << " shape=(" << v.size() << ",) dtype=" << pycauset::dtype_name(v.get_data_type()) << " transposed=False>";
                }
                return oss.str();
            })
        .def(
            "__str__",
            [](py::object self) {
                return py::cast<std::string>(self.attr("__repr__")());
            })
        .def_property_readonly("shape", [](const VectorBase& v) {
            if (v.is_transposed()) {
                return py::make_tuple(1, v.size());
            }
            return py::make_tuple(v.size());
        })
        .def(
            "_to_numpy_fast",
            [](const VectorBase& v, bool allow_huge) -> py::object {
                // pycauset.to_numpy: the same exporter as __array__, with the allow_huge policy.
                if (!allow_huge) pycauset::bindings::ensure_numpy_export_allowed(v);
                py::object out = pycauset_bindings::export_matrix_numpy(*pycauset::as_row_matrix(v));
                if (!v.is_transposed()) out = out.attr("reshape")(static_cast<py::ssize_t>(v.size()));
                return out;
            },
            py::arg("allow_huge") = false)
        .def_property_readonly(
            "T",
            [](const VectorBase& v) {
                auto out = v.transpose("");
                return std::shared_ptr<VectorBase>(out.release());
            })
        .def_property_readonly(
            "H",
            [](const VectorBase& v) {
                auto out = v.transpose("");
                out->set_conjugated(!v.is_conjugated());
                return std::shared_ptr<VectorBase>(out.release());
            })
        .def(
            "transpose",
            [](const VectorBase& v) {
                auto out = v.transpose("");
                return std::shared_ptr<VectorBase>(out.release());
            })
        .def(
            "conjugate",
            [](const VectorBase& v) {
                auto out = v.clone();
                auto* vec = dynamic_cast<VectorBase*>(out.get());
                if (!vec) {
                    throw std::runtime_error("Internal error: clone() did not return a VectorBase");
                }
                vec->set_conjugated(!v.is_conjugated());
                return std::shared_ptr<VectorBase>(static_cast<VectorBase*>(out.release()));
            })
        .def(
            "conj",
            [](const VectorBase& v) {
                auto out = v.clone();
                auto* vec = dynamic_cast<VectorBase*>(out.get());
                if (!vec) {
                    throw std::runtime_error("Internal error: clone() did not return a VectorBase");
                }
                vec->set_conjugated(!v.is_conjugated());
                return std::shared_ptr<VectorBase>(static_cast<VectorBase*>(out.release()));
            })
        .def(
            "dot",
            [](const VectorBase& a, const std::shared_ptr<VectorBase>& b) { return vector_dot(a, *b); },
            py::arg("other"))
        // Bitwise ops through the matrix machinery (1 x n bridge, math/Bitwise.cpp).
        .def("__and__", [](const VectorBase& a, const VectorBase& b) { return vector_bitwise(a, &b, pycauset::BitwiseOp::And); }, py::is_operator())
        .def("__or__", [](const VectorBase& a, const VectorBase& b) { return vector_bitwise(a, &b, pycauset::BitwiseOp::Or); }, py::is_operator())
        .def("__xor__", [](const VectorBase& a, const VectorBase& b) { return vector_bitwise(a, &b, pycauset::BitwiseOp::Xor); }, py::is_operator())
        .def("__invert__", [](const VectorBase& a) { return vector_bitwise(a, nullptr, pycauset::BitwiseOp::And); }, py::is_operator())
        .def("__add__", [](const VectorBase& a, const py::object& b) { return vector_arith(a, b, Arith::Add, false); }, py::is_operator())
        .def("__radd__", [](const VectorBase& a, const py::object& b) { return vector_arith(a, b, Arith::Add, true); }, py::is_operator())
        .def("__sub__", [](const VectorBase& a, const py::object& b) { return vector_arith(a, b, Arith::Sub, false); }, py::is_operator())
        .def("__rsub__", [](const VectorBase& a, const py::object& b) { return vector_arith(a, b, Arith::Sub, true); }, py::is_operator())
        .def("__mul__", [](const VectorBase& a, const py::object& b) { return vector_arith(a, b, Arith::Mul, false); }, py::is_operator())
        .def("__rmul__", [](const VectorBase& a, const py::object& b) { return vector_arith(a, b, Arith::Mul, true); }, py::is_operator())
        .def("__truediv__", [](const VectorBase& a, const py::object& b) { return vector_arith(a, b, Arith::Div, false); }, py::is_operator())
        .def("__rtruediv__", [](const VectorBase& a, const py::object& b) { return vector_arith(a, b, Arith::Div, true); }, py::is_operator())
        .def("__neg__", [](const VectorBase& a) { return std::shared_ptr<VectorBase>(pycauset::negate(a).release()); })
        .def(
            "__matmul__",
            [](const VectorBase& a, const std::shared_ptr<VectorBase>& b) -> py::object {
                // Col @ Row -> Outer product; otherwise dot.
                if (!a.is_transposed() && b->is_transposed()) {
                    auto out = pycauset::outer_product(a, *b, "");
                    return py::cast(std::shared_ptr<MatrixBase>(out.release()));
                }
                return vector_dot(a, *b);
            },
            py::is_operator())
        .def(
            "__matmul__",
            [](const VectorBase& a, const std::shared_ptr<MatrixBase>& mtx) {
                return translate_invalid_argument([&]() {
                    maybe_warn_bit_promotes_to_int32(
                        "matmul",
                        promotion::BinaryOp::VectorMatrixMultiply,
                        a.get_data_type(),
                        mtx->get_data_type());
                    auto out = pycauset::vector_matrix_multiply(a, *mtx, "");
                    return std::shared_ptr<VectorBase>(out.release());
                });
            },
            py::is_operator())
        // NumPy operands, as matrices take them (review 3.6 C-7): v @ arr and arr @ v (NumPy
        // defers to us, __array_ufunc__ = None). 1-D: the dot; 2-D: the vector-matrix product.
        .def(
            "__matmul__",
            [](const VectorBase& a, const py::array& b) -> py::object {
                if (b.ndim() == 1) return vector_dot(a, *vector_from_numpy(b));
                if (b.ndim() == 2) {
                    auto m = matrix_from_numpy(b);
                    return py::cast(std::shared_ptr<VectorBase>(
                        translate_invalid_argument([&] { return pycauset::vector_matrix_multiply(a, *m, ""); }).release()));
                }
                throw py::value_error("matmul: a NumPy operand must be 1-D or 2-D");
            },
            py::is_operator())
        .def(
            "__rmatmul__",
            [](const VectorBase& a, const py::array& b) -> py::object {
                if (b.ndim() == 1) return vector_dot(*vector_from_numpy(b), a);
                if (b.ndim() == 2) {
                    auto m = matrix_from_numpy(b);
                    return py::cast(std::shared_ptr<VectorBase>(
                        translate_invalid_argument([&] { return pycauset::matrix_vector_multiply(*m, a, ""); }).release()));
                }
                throw py::value_error("matmul: a NumPy operand must be 1-D or 2-D");
            },
            py::is_operator());

    // One Python class per structure; dtype is an attribute (RESTRUCTURE 3.1 item 4).
    py::class_<pycauset::DenseVectorBase, VectorBase, std::shared_ptr<pycauset::DenseVectorBase>>(
        m, "DenseVector", "Dense vector of any dtype. Prefer pycauset.vector / zeros to construct.")
        .def(py::init([](uint64_t n, const py::object& dtype) {
                 auto obj = pycauset::ObjectFactory::create_vector(n, dtype_arg(dtype), pycauset::Structure::VECTOR, "");
                 auto* typed = dynamic_cast<pycauset::DenseVectorBase*>(obj.get());
                 if (typed == nullptr) throw std::logic_error("factory returned the wrong vector structure");
                 obj.release();
                 return std::shared_ptr<pycauset::DenseVectorBase>(typed);
             }),
             py::arg("n"), py::arg("dtype") = "float64");

    py::class_<UnitVector, VectorBase, std::shared_ptr<UnitVector>>(m, "UnitVector", "Standard basis vector e_k (read-only).")
        .def(py::init([](uint64_t n, uint64_t active_index, const py::object& dtype) {
                 return std::make_shared<UnitVector>(n, active_index, "", dtype_arg(dtype));
             }),
             py::arg("n"), py::arg("active_index"), py::arg("dtype") = "float64")
        .def_property_readonly("active_index", &UnitVector::get_active_index);

    // Encourage NumPy to prefer VectorBase's reverse ops over coercion.
    py::object vb_cls = m.attr("VectorBase");
    vb_cls.attr("__array_priority__") = py::float_(1000.0);
}
