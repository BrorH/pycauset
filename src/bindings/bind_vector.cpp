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
#include "pycauset/math/ViewValue.hpp"
#include "pycauset/vector/ComplexFloat16Vector.hpp"

#include "pycauset/matrix/MatrixBase.hpp"

#include "numpy_export_guard.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <limits>

using namespace pycauset;

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


inline bool is_numpy_float16(const py::array& array) {
    try {
        return py::str(array.dtype()).cast<std::string>() == "float16";
    } catch (...) {
        return false;
    }
}

// Exact integer dot product (DType System.md 5.1: reductions never wrap silently). Both
// operands integer or bit, dense: logical elements (integer view factor applied with NumPy's
// element wrap), accumulated in 128 bits, returned as a Python int; OverflowError past 127 bits.
// Returns an empty object when the operands are not both integer (callers use the float path).
inline py::object exact_int_dot(const VectorBase& a, const VectorBase& b) {
    auto is_int = [](const VectorBase& v) {
        const DataType dt = v.get_data_type();
        return v.get_structure() != Structure::UNIT_VECTOR && dt != DataType::FLOAT16 &&
               dt != DataType::FLOAT32 && dt != DataType::FLOAT64 && dt != DataType::COMPLEX_FLOAT16 &&
               dt != DataType::COMPLEX_FLOAT32 && dt != DataType::COMPLEX_FLOAT64 &&
               scalar::is_exact_int_factor(v.get_scalar());
    };
    if (!is_int(a) || !is_int(b)) return py::object();
    if (a.get_data_type() == DataType::BIT && b.get_data_type() == DataType::BIT) {
        // Word-level popcount kernel; a count < 2^53 is exact in double.
        return py::int_(static_cast<int64_t>(pycauset::dot_product(a, b)));
    }
    if (a.size() != b.size()) throw py::value_error("dot: vector sizes differ");
    const int64_t ka = static_cast<int64_t>(a.get_scalar().real());
    const int64_t kb = static_cast<int64_t>(b.get_scalar().real());
    __int128 sum = 0;
    visit_dtype(a.get_data_type(), [&](auto ta) {
        visit_dtype(b.get_data_type(), [&](auto tb) {
            using A = typename decltype(ta)::type;
            using B = typename decltype(tb)::type;
            if constexpr (std::is_integral_v<A> && std::is_integral_v<B>) {
                const auto& va = static_cast<const typename DenseVectorOf<A>::type&>(a);
                const auto& vb = static_cast<const typename DenseVectorOf<B>::type&>(b);
                const uint64_t n = a.size();
                for (uint64_t i = 0; i < n; ++i) {
                    const __int128 x = scalar::wrap_scale<A>(va.get(i), ka);
                    const __int128 y = scalar::wrap_scale<B>(vb.get(i), kb);
                    __int128 term;
                    if (__builtin_mul_overflow(x, y, &term) || __builtin_add_overflow(sum, term, &sum)) {
                        throw std::overflow_error("dot: integer result does not fit the 128-bit accumulator");
                    }
                }
            }
        });
    });
    return pycauset_bindings::int128_to_py(sum);
}

// Inner product of two vectors (orientation ignored, no conjugation, like np.dot).
inline py::object vector_dot(const VectorBase& a, const VectorBase& b) {
    if (py::object exact = exact_int_dot(a, b)) return exact;
    const auto is_complex = [](DataType dt) {
        return dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64;
    };
    if (is_complex(a.get_data_type()) || is_complex(b.get_data_type())) {
        return py::cast(pycauset::dot_product_complex(a, b));
    }
    return py::float_(pycauset::dot_product(a, b));
}

inline void require_1d(const py::buffer_info& buf) {
    if (buf.ndim != 1) {
        throw py::value_error("Expected a 1D NumPy array for vector conversion");
    }
}

inline void maybe_warn_bit_promotes_to_int32(const char* op_name, promotion::BinaryOp op, DataType da, DataType db) {
    if (!(da == DataType::BIT && db == DataType::BIT)) {
        return;
    }

    const auto decision = promotion::resolve(op, da, db);
    if (decision.result_dtype != DataType::INT32) {
        return;
    }

    std::string message;
    message.reserve(256);
    message += "pycauset ";
    message += op_name;
    message += ": bit operands promote to int32 because the result can exceed {0,1} (mathematically unavoidable). ";
    message += "Even though int16 exists, bit-bit reductions commonly exceed int16; result is int32.";

    bindings_warn::warn_with_category(message, "PyCausetDTypeWarning", /*stacklevel=*/3);
}

template <typename T>
std::shared_ptr<DenseVector<T>> dense_vector_from_numpy_1d(const py::array_t<T>& array) {
    auto buf = array.request();
    require_1d(buf);

    uint64_t n = static_cast<uint64_t>(buf.shape[0]);
    auto result = std::make_shared<DenseVector<T>>(n);

    const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(T)));
    const T* src_ptr = static_cast<const T*>(buf.ptr);
    T* dst_ptr = result->data();

    if (stride0 == 1) {
        std::memcpy(dst_ptr, src_ptr, n * sizeof(T));
    } else {
        for (uint64_t i = 0; i < n; ++i) {
            dst_ptr[i] = src_ptr[i * stride0];
        }
    }
    return result;
}

std::shared_ptr<VectorBase> vector_from_numpy(const py::array& array) {
    if (is_numpy_float16(array)) {
        auto buf = array.request();
        require_1d(buf);
        uint64_t n = static_cast<uint64_t>(buf.shape[0]);
        auto result = std::make_shared<DenseVector<pycauset::float16_t>>(n);
        const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(uint16_t)));
        const uint16_t* src_ptr = static_cast<const uint16_t*>(buf.ptr);
        pycauset::float16_t* dst_ptr = result->data();
        for (uint64_t i = 0; i < n; ++i) {
            dst_ptr[i] = pycauset::float16_t(static_cast<uint16_t>(src_ptr[i * stride0]));
        }
        return result;
    }
    if (py::isinstance<py::array_t<double>>(array)) {
        return dense_vector_from_numpy_1d<double>(array.cast<py::array_t<double>>());
    }
    if (py::isinstance<py::array_t<float>>(array)) {
        // Promote float32 numpy vectors to float64 vectors for now.
        // This keeps the public API stable and avoids needing a Float32Vector class.
        auto tmp = array.cast<py::array_t<float>>();
        auto buf = tmp.request();
        require_1d(buf);
        uint64_t n = static_cast<uint64_t>(buf.shape[0]);
        auto result = std::make_shared<DenseVector<double>>(n);
        const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(float)));
        const float* src_ptr = static_cast<const float*>(buf.ptr);
        double* dst_ptr = result->data();
        for (uint64_t i = 0; i < n; ++i) {
            dst_ptr[i] = static_cast<double>(src_ptr[i * stride0]);
        }
        return result;
    }
    if (py::isinstance<py::array_t<std::complex<float>>>(array)) {
        return dense_vector_from_numpy_1d<std::complex<float>>(array.cast<py::array_t<std::complex<float>>>());
    }
    if (py::isinstance<py::array_t<std::complex<double>>>(array)) {
        return dense_vector_from_numpy_1d<std::complex<double>>(array.cast<py::array_t<std::complex<double>>>());
    }
    if (py::isinstance<py::array_t<int32_t>>(array)) {
        return dense_vector_from_numpy_1d<int32_t>(array.cast<py::array_t<int32_t>>());
    }
    if (py::isinstance<py::array_t<int64_t>>(array)) {
        return dense_vector_from_numpy_1d<int64_t>(array.cast<py::array_t<int64_t>>());
    }
    if (py::isinstance<py::array_t<int8_t>>(array)) {
        return dense_vector_from_numpy_1d<int8_t>(array.cast<py::array_t<int8_t>>());
    }
    if (py::isinstance<py::array_t<int16_t>>(array)) {
        return dense_vector_from_numpy_1d<int16_t>(array.cast<py::array_t<int16_t>>());
    }
    if (py::isinstance<py::array_t<uint8_t>>(array)) {
        return dense_vector_from_numpy_1d<uint8_t>(array.cast<py::array_t<uint8_t>>());
    }
    if (py::isinstance<py::array_t<uint16_t>>(array)) {
        return dense_vector_from_numpy_1d<uint16_t>(array.cast<py::array_t<uint16_t>>());
    }
    if (py::isinstance<py::array_t<uint32_t>>(array)) {
        return dense_vector_from_numpy_1d<uint32_t>(array.cast<py::array_t<uint32_t>>());
    }
    if (py::isinstance<py::array_t<uint64_t>>(array)) {
        return dense_vector_from_numpy_1d<uint64_t>(array.cast<py::array_t<uint64_t>>());
    }
    if (py::isinstance<py::array_t<bool>>(array)) {
        auto tmp = array.cast<py::array_t<bool>>();
        auto buf = tmp.request();
        require_1d(buf);
        uint64_t n = static_cast<uint64_t>(buf.shape[0]);
        auto result = std::make_shared<DenseVector<bool>>(n);
        const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(bool)));
        const bool* src_ptr = static_cast<const bool*>(buf.ptr);
        for (uint64_t i = 0; i < n; ++i) {
            result->set(i, src_ptr[i * stride0]);
        }
        return result;
    }
    throw py::type_error("Unsupported NumPy dtype for vector conversion");
}

template <typename Fn>
inline auto translate_invalid_argument(Fn&& fn) {
    try {
        return fn();
    } catch (const std::invalid_argument& e) {
        throw py::value_error(e.what());
    }
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
    if (dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64) {
        return py::cast(v.get_element_as_complex(i));
    }
    if (dt == DataType::FLOAT16 || dt == DataType::FLOAT32 || dt == DataType::FLOAT64) {
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
    const bool conj = v.is_conjugated();
    if (s == std::complex<double>(1.0, 0.0) && !conj) return;
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

} // namespace

void bind_vector_classes(py::module_& m) {
    py::class_<VectorBase, std::shared_ptr<VectorBase>>(m, "VectorBase", py::dynamic_attr())
        .def_property_readonly("backing_file", &VectorBase::get_backing_file)
        .def("get_backing_file", &VectorBase::get_backing_file)
        .def_property_readonly("is_temporary", &VectorBase::is_temporary)
        .def("set_temporary", &VectorBase::set_temporary)
        .def("close", &VectorBase::close)
        .def("copy_storage", &VectorBase::copy_storage, py::arg("result_file_hint") = "")
        .def("size", &VectorBase::size)
        .def("__len__", &VectorBase::size)
        .def_property("seed", &VectorBase::get_seed, &VectorBase::set_seed)
        .def_property("scalar", &VectorBase::get_scalar, &VectorBase::set_scalar)
        .def("get_scalar", &VectorBase::get_scalar)
        .def("set_scalar", &VectorBase::set_scalar)
        .def("is_transposed", &VectorBase::is_transposed)
        .def("set_transposed", &VectorBase::set_transposed)
        .def("is_conjugated", &VectorBase::is_conjugated)
        .def("set_conjugated", &VectorBase::set_conjugated)
        .def("_get_properties_flags", &VectorBase::get_properties_flags)
        .def("_set_properties_flags", &VectorBase::set_properties_flags)
        .def("_set_property_flag", &VectorBase::set_property_flag, py::arg("flag"), py::arg("value"))
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
            "__array__",
            [](const VectorBase& v, py::object /*dtype*/, py::object /*copy*/) -> py::array {
                pycauset::bindings::ensure_numpy_export_allowed(v);
                uint64_t n_u = v.size();
                py::ssize_t n = static_cast<py::ssize_t>(n_u);
                bool as_row = v.is_transposed();

                const DataType dt = v.get_data_type();
                if (dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64) {
                    if (dt == DataType::COMPLEX_FLOAT64) {
                        if (as_row) {
                            py::array out(py::dtype("complex128"), py::array::ShapeContainer{static_cast<py::ssize_t>(1), n});
                            auto buf = out.request();
                            auto* dst_ptr = static_cast<std::complex<double>*>(buf.ptr);
                            const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(std::complex<double>)));
                            const auto stride1 = static_cast<uint64_t>(buf.strides[1] / static_cast<py::ssize_t>(sizeof(std::complex<double>)));
                            for (uint64_t i = 0; i < n_u; ++i) dst_ptr[0 * stride0 + i * stride1] = v.get_element_as_complex(i);
                            return out;
                        }
                        py::array out(py::dtype("complex128"), py::array::ShapeContainer{n});
                        auto buf = out.request();
                        auto* dst_ptr = static_cast<std::complex<double>*>(buf.ptr);
                        const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(std::complex<double>)));
                        for (uint64_t i = 0; i < n_u; ++i) dst_ptr[i * stride0] = v.get_element_as_complex(i);
                        return out;
                    }

                    // complex64 for both complex_float16 and complex_float32.
                    if (as_row) {
                        py::array out(py::dtype("complex64"), py::array::ShapeContainer{static_cast<py::ssize_t>(1), n});
                        auto buf = out.request();
                        auto* dst_ptr = static_cast<std::complex<float>*>(buf.ptr);
                        const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(std::complex<float>)));
                        const auto stride1 = static_cast<uint64_t>(buf.strides[1] / static_cast<py::ssize_t>(sizeof(std::complex<float>)));
                        for (uint64_t i = 0; i < n_u; ++i) {
                            const std::complex<double> z = v.get_element_as_complex(i);
                            dst_ptr[0 * stride0 + i * stride1] = std::complex<float>(static_cast<float>(z.real()), static_cast<float>(z.imag()));
                        }
                        return out;
                    }
                    py::array out(py::dtype("complex64"), py::array::ShapeContainer{n});
                    auto buf = out.request();
                    auto* dst_ptr = static_cast<std::complex<float>*>(buf.ptr);
                    const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(std::complex<float>)));
                    for (uint64_t i = 0; i < n_u; ++i) {
                        const std::complex<double> z = v.get_element_as_complex(i);
                        dst_ptr[i * stride0] = std::complex<float>(static_cast<float>(z.real()), static_cast<float>(z.imag()));
                    }
                    return out;
                }

                if (auto* vf16 = dynamic_cast<const DenseVector<pycauset::float16_t>*>(&v)) {
                    if (as_row) {
                        py::array out(
                            py::dtype("float16"),
                            py::array::ShapeContainer{static_cast<py::ssize_t>(1), n});
                        auto buf = out.request();
                        auto* dst_ptr = static_cast<uint16_t*>(buf.ptr);
                        const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(uint16_t)));
                        const auto stride1 = static_cast<uint64_t>(buf.strides[1] / static_cast<py::ssize_t>(sizeof(uint16_t)));
                        for (uint64_t i = 0; i < n_u; ++i) dst_ptr[0 * stride0 + i * stride1] = vf16->get(i).bits;
                        return out;
                    }
                    py::array out(py::dtype("float16"), py::array::ShapeContainer{n});
                    auto buf = out.request();
                    auto* dst_ptr = static_cast<uint16_t*>(buf.ptr);
                    const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(uint16_t)));
                    for (uint64_t i = 0; i < n_u; ++i) dst_ptr[i * stride0] = vf16->get(i).bits;
                    return out;
                }
                if (auto* vf32 = dynamic_cast<const DenseVector<float>*>(&v)) {
                    if (as_row) {
                        py::array_t<float> out({static_cast<py::ssize_t>(1), n});
                        auto r = out.template mutable_unchecked<2>();
                        for (uint64_t i = 0; i < n_u; ++i) r(0, i) = vf32->get(i);
                        return out;
                    }
                    py::array_t<float> out({n});
                    auto r = out.template mutable_unchecked<1>();
                    for (uint64_t i = 0; i < n_u; ++i) r(i) = vf32->get(i);
                    return out;
                }
                if (auto* vi = dynamic_cast<const DenseVector<int32_t>*>(&v)) {
                    if (as_row) {
                        py::array_t<int32_t> out({static_cast<py::ssize_t>(1), n});
                        auto r = out.template mutable_unchecked<2>();
                        for (uint64_t i = 0; i < n_u; ++i) r(0, i) = vi->get(i);
                        return out;
                    }
                    py::array_t<int32_t> out({n});
                    auto r = out.template mutable_unchecked<1>();
                    for (uint64_t i = 0; i < n_u; ++i) r(i) = vi->get(i);
                    return out;
                }
                if (auto* vi64 = dynamic_cast<const DenseVector<int64_t>*>(&v)) {
                    if (as_row) {
                        py::array_t<int64_t> out({static_cast<py::ssize_t>(1), n});
                        auto r = out.template mutable_unchecked<2>();
                        for (uint64_t i = 0; i < n_u; ++i) r(0, i) = vi64->get(i);
                        return out;
                    }
                    py::array_t<int64_t> out({n});
                    auto r = out.template mutable_unchecked<1>();
                    for (uint64_t i = 0; i < n_u; ++i) r(i) = vi64->get(i);
                    return out;
                }
                if (auto* vi8 = dynamic_cast<const DenseVector<int8_t>*>(&v)) {
                    if (as_row) {
                        py::array_t<int8_t> out({static_cast<py::ssize_t>(1), n});
                        auto r = out.template mutable_unchecked<2>();
                        for (uint64_t i = 0; i < n_u; ++i) r(0, i) = vi8->get(i);
                        return out;
                    }
                    py::array_t<int8_t> out({n});
                    auto r = out.template mutable_unchecked<1>();
                    for (uint64_t i = 0; i < n_u; ++i) r(i) = vi8->get(i);
                    return out;
                }
                if (auto* vi16 = dynamic_cast<const DenseVector<int16_t>*>(&v)) {
                    if (as_row) {
                        py::array_t<int16_t> out({static_cast<py::ssize_t>(1), n});
                        auto r = out.template mutable_unchecked<2>();
                        for (uint64_t i = 0; i < n_u; ++i) r(0, i) = vi16->get(i);
                        return out;
                    }
                    py::array_t<int16_t> out({n});
                    auto r = out.template mutable_unchecked<1>();
                    for (uint64_t i = 0; i < n_u; ++i) r(i) = vi16->get(i);
                    return out;
                }
                if (auto* vu8 = dynamic_cast<const DenseVector<uint8_t>*>(&v)) {
                    if (as_row) {
                        py::array_t<uint8_t> out({static_cast<py::ssize_t>(1), n});
                        auto r = out.template mutable_unchecked<2>();
                        for (uint64_t i = 0; i < n_u; ++i) r(0, i) = vu8->get(i);
                        return out;
                    }
                    py::array_t<uint8_t> out({n});
                    auto r = out.template mutable_unchecked<1>();
                    for (uint64_t i = 0; i < n_u; ++i) r(i) = vu8->get(i);
                    return out;
                }
                if (auto* vu16 = dynamic_cast<const DenseVector<uint16_t>*>(&v)) {
                    if (as_row) {
                        py::array_t<uint16_t> out({static_cast<py::ssize_t>(1), n});
                        auto r = out.template mutable_unchecked<2>();
                        for (uint64_t i = 0; i < n_u; ++i) r(0, i) = vu16->get(i);
                        return out;
                    }
                    py::array_t<uint16_t> out({n});
                    auto r = out.template mutable_unchecked<1>();
                    for (uint64_t i = 0; i < n_u; ++i) r(i) = vu16->get(i);
                    return out;
                }
                if (auto* vu32 = dynamic_cast<const DenseVector<uint32_t>*>(&v)) {
                    if (as_row) {
                        py::array_t<uint32_t> out({static_cast<py::ssize_t>(1), n});
                        auto r = out.template mutable_unchecked<2>();
                        for (uint64_t i = 0; i < n_u; ++i) r(0, i) = vu32->get(i);
                        return out;
                    }
                    py::array_t<uint32_t> out({n});
                    auto r = out.template mutable_unchecked<1>();
                    for (uint64_t i = 0; i < n_u; ++i) r(i) = vu32->get(i);
                    return out;
                }
                if (auto* vu64 = dynamic_cast<const DenseVector<uint64_t>*>(&v)) {
                    if (as_row) {
                        py::array_t<uint64_t> out({static_cast<py::ssize_t>(1), n});
                        auto r = out.template mutable_unchecked<2>();
                        for (uint64_t i = 0; i < n_u; ++i) r(0, i) = vu64->get(i);
                        return out;
                    }
                    py::array_t<uint64_t> out({n});
                    auto r = out.template mutable_unchecked<1>();
                    for (uint64_t i = 0; i < n_u; ++i) r(i) = vu64->get(i);
                    return out;
                }
                if (auto* vb = dynamic_cast<const DenseVector<bool>*>(&v)) {
                    if (as_row) {
                        py::array_t<bool> out({static_cast<py::ssize_t>(1), n});
                        auto r = out.template mutable_unchecked<2>();
                        for (uint64_t i = 0; i < n_u; ++i) r(0, i) = vb->get(i);
                        return out;
                    }
                    py::array_t<bool> out({n});
                    auto r = out.template mutable_unchecked<1>();
                    for (uint64_t i = 0; i < n_u; ++i) r(i) = vb->get(i);
                    return out;
                }

                // Default: float64
                if (as_row) {
                    py::array_t<double> out({static_cast<py::ssize_t>(1), n});
                    auto r = out.template mutable_unchecked<2>();
                    for (uint64_t i = 0; i < n_u; ++i) r(0, i) = v.get_element_as_double(i);
                    return out;
                }
                py::array_t<double> out({n});
                auto r = out.template mutable_unchecked<1>();
                for (uint64_t i = 0; i < n_u; ++i) r(i) = v.get_element_as_double(i);
                return out;
            },
            py::arg("dtype") = py::none(),
            py::arg("copy") = py::none())
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
            py::is_operator());

    // One Python class per structure; dtype is an attribute (RESTRUCTURE 3.1 item 4).
    py::class_<pycauset::DenseVectorBase, VectorBase, std::shared_ptr<pycauset::DenseVectorBase>>(
        m, "DenseVector", "Dense vector of any dtype. Prefer pycauset.vector / zeros to construct.")
        .def(py::init([](uint64_t n, const std::string& dtype) {
                 auto obj = pycauset::ObjectFactory::create_vector(n, pycauset::parse_dtype_name(dtype), pycauset::Structure::VECTOR, "");
                 auto* typed = dynamic_cast<pycauset::DenseVectorBase*>(obj.get());
                 if (typed == nullptr) throw std::logic_error("factory returned the wrong vector structure");
                 obj.release();
                 return std::shared_ptr<pycauset::DenseVectorBase>(typed);
             }),
             py::arg("n"), py::arg("dtype") = "float64");

    py::class_<UnitVector, VectorBase, std::shared_ptr<UnitVector>>(m, "UnitVector", "Standard basis vector e_k (read-only).")
        .def(py::init([](uint64_t n, uint64_t active_index, const std::string& dtype) {
                 return std::make_shared<UnitVector>(n, active_index, "", pycauset::parse_dtype_name(dtype));
             }),
             py::arg("n"), py::arg("active_index"), py::arg("dtype") = "float64")
        .def_property_readonly("active_index", &UnitVector::get_active_index);

    // Encourage NumPy to prefer VectorBase's reverse ops over coercion.
    py::object vb_cls = m.attr("VectorBase");
    vb_cls.attr("__array_priority__") = py::float_(1000.0);
}
