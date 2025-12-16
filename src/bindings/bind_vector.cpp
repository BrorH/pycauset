#include "bindings_common.hpp"

#include "binding_warnings.hpp"

#include "pycauset/core/PromotionResolver.hpp"
#include "pycauset/core/Float16.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/vector/VectorBase.hpp"
#include "pycauset/vector/DenseVector.hpp"
#include "pycauset/vector/UnitVector.hpp"
#include "pycauset/vector/ComplexFloat16Vector.hpp"

#include "pycauset/matrix/MatrixBase.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <limits>

using namespace pycauset;

namespace {

inline bool is_numpy_float16(const py::array& array) {
    try {
        return py::str(array.dtype()).cast<std::string>() == "float16";
    } catch (...) {
        return false;
    }
}

inline int64_t checked_add_int64(int64_t a, int64_t b) {
    if ((b > 0 && a > (std::numeric_limits<int64_t>::max)() - b) ||
        (b < 0 && a < (std::numeric_limits<int64_t>::min)() - b)) {
        throw std::overflow_error("Integer dot overflow: accumulator overflow");
    }
    return a + b;
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

    std::string key;
    key.reserve(96);
    key += "pycauset.";
    key += op_name;
    key += ".bit_promotes_to_int32";

    std::string message;
    message.reserve(256);
    message += "pycauset ";
    message += op_name;
    message += ": bit operands promote to int32 because the result can exceed {0,1} (mathematically unavoidable). ";
    message += "Even though int16 exists, bit-bit reductions commonly exceed int16; result is int32.";

    bindings_warn::warn_once_with_category(key, message, "PyCausetDTypeWarning", /*stacklevel=*/3);
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
        auto result = std::make_shared<DenseVector<float16_t>>(n);
        const auto stride0 = static_cast<uint64_t>(buf.strides[0] / static_cast<py::ssize_t>(sizeof(uint16_t)));
        const uint16_t* src_ptr = static_cast<const uint16_t*>(buf.ptr);
        float16_t* dst_ptr = result->data();
        for (uint64_t i = 0; i < n; ++i) {
            dst_ptr[i] = float16_t(static_cast<uint16_t>(src_ptr[i * stride0]));
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
        .def("get_element_as_double", &VectorBase::get_element_as_double)
        .def("get_element_as_complex", &VectorBase::get_element_as_complex)
        .def(
            "__getitem__",
            [](const VectorBase& v, uint64_t i) -> py::object {
                const DataType dt = v.get_data_type();
                if (dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64) {
                    return py::cast(v.get_element_as_complex(i));
                }
                return py::float_(v.get_element_as_double(i));
            })
        .def(
            "__repr__",
            [](py::object self) {
                const auto& v = self.cast<const VectorBase&>();
                std::string type_name = py::str(self.get_type().attr("__name__"));

                std::ostringstream oss;
                if (v.is_transposed()) {
                    oss << "<" << type_name << " shape=(1, " << v.size() << ") transposed=True>";
                } else {
                    oss << "<" << type_name << " shape=(" << v.size() << ",) transposed=False>";
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

                if (auto* vf16 = dynamic_cast<const DenseVector<float16_t>*>(&v)) {
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
                        auto r = out.mutable_unchecked<2>();
                        for (uint64_t i = 0; i < n_u; ++i) r(0, i) = vf32->get(i);
                        return out;
                    }
                    py::array_t<float> out({n});
                    auto r = out.mutable_unchecked<1>();
                    for (uint64_t i = 0; i < n_u; ++i) r(i) = vf32->get(i);
                    return out;
                }
                if (auto* vi = dynamic_cast<const DenseVector<int32_t>*>(&v)) {
                    if (as_row) {
                        py::array_t<int32_t> out({static_cast<py::ssize_t>(1), n});
                        auto r = out.mutable_unchecked<2>();
                        for (uint64_t i = 0; i < n_u; ++i) r(0, i) = vi->get(i);
                        return out;
                    }
                    py::array_t<int32_t> out({n});
                    auto r = out.mutable_unchecked<1>();
                    for (uint64_t i = 0; i < n_u; ++i) r(i) = vi->get(i);
                    return out;
                }
                if (auto* vi64 = dynamic_cast<const DenseVector<int64_t>*>(&v)) {
                    if (as_row) {
                        py::array_t<int64_t> out({static_cast<py::ssize_t>(1), n});
                        auto r = out.mutable_unchecked<2>();
                        for (uint64_t i = 0; i < n_u; ++i) r(0, i) = vi64->get(i);
                        return out;
                    }
                    py::array_t<int64_t> out({n});
                    auto r = out.mutable_unchecked<1>();
                    for (uint64_t i = 0; i < n_u; ++i) r(i) = vi64->get(i);
                    return out;
                }
                if (auto* vi8 = dynamic_cast<const DenseVector<int8_t>*>(&v)) {
                    if (as_row) {
                        py::array_t<int8_t> out({static_cast<py::ssize_t>(1), n});
                        auto r = out.mutable_unchecked<2>();
                        for (uint64_t i = 0; i < n_u; ++i) r(0, i) = vi8->get(i);
                        return out;
                    }
                    py::array_t<int8_t> out({n});
                    auto r = out.mutable_unchecked<1>();
                    for (uint64_t i = 0; i < n_u; ++i) r(i) = vi8->get(i);
                    return out;
                }
                if (auto* vi16 = dynamic_cast<const DenseVector<int16_t>*>(&v)) {
                    if (as_row) {
                        py::array_t<int16_t> out({static_cast<py::ssize_t>(1), n});
                        auto r = out.mutable_unchecked<2>();
                        for (uint64_t i = 0; i < n_u; ++i) r(0, i) = vi16->get(i);
                        return out;
                    }
                    py::array_t<int16_t> out({n});
                    auto r = out.mutable_unchecked<1>();
                    for (uint64_t i = 0; i < n_u; ++i) r(i) = vi16->get(i);
                    return out;
                }
                if (auto* vu8 = dynamic_cast<const DenseVector<uint8_t>*>(&v)) {
                    if (as_row) {
                        py::array_t<uint8_t> out({static_cast<py::ssize_t>(1), n});
                        auto r = out.mutable_unchecked<2>();
                        for (uint64_t i = 0; i < n_u; ++i) r(0, i) = vu8->get(i);
                        return out;
                    }
                    py::array_t<uint8_t> out({n});
                    auto r = out.mutable_unchecked<1>();
                    for (uint64_t i = 0; i < n_u; ++i) r(i) = vu8->get(i);
                    return out;
                }
                if (auto* vu16 = dynamic_cast<const DenseVector<uint16_t>*>(&v)) {
                    if (as_row) {
                        py::array_t<uint16_t> out({static_cast<py::ssize_t>(1), n});
                        auto r = out.mutable_unchecked<2>();
                        for (uint64_t i = 0; i < n_u; ++i) r(0, i) = vu16->get(i);
                        return out;
                    }
                    py::array_t<uint16_t> out({n});
                    auto r = out.mutable_unchecked<1>();
                    for (uint64_t i = 0; i < n_u; ++i) r(i) = vu16->get(i);
                    return out;
                }
                if (auto* vu32 = dynamic_cast<const DenseVector<uint32_t>*>(&v)) {
                    if (as_row) {
                        py::array_t<uint32_t> out({static_cast<py::ssize_t>(1), n});
                        auto r = out.mutable_unchecked<2>();
                        for (uint64_t i = 0; i < n_u; ++i) r(0, i) = vu32->get(i);
                        return out;
                    }
                    py::array_t<uint32_t> out({n});
                    auto r = out.mutable_unchecked<1>();
                    for (uint64_t i = 0; i < n_u; ++i) r(i) = vu32->get(i);
                    return out;
                }
                if (auto* vu64 = dynamic_cast<const DenseVector<uint64_t>*>(&v)) {
                    if (as_row) {
                        py::array_t<uint64_t> out({static_cast<py::ssize_t>(1), n});
                        auto r = out.mutable_unchecked<2>();
                        for (uint64_t i = 0; i < n_u; ++i) r(0, i) = vu64->get(i);
                        return out;
                    }
                    py::array_t<uint64_t> out({n});
                    auto r = out.mutable_unchecked<1>();
                    for (uint64_t i = 0; i < n_u; ++i) r(i) = vu64->get(i);
                    return out;
                }
                if (auto* vb = dynamic_cast<const DenseVector<bool>*>(&v)) {
                    if (as_row) {
                        py::array_t<bool> out({static_cast<py::ssize_t>(1), n});
                        auto r = out.mutable_unchecked<2>();
                        for (uint64_t i = 0; i < n_u; ++i) r(0, i) = vb->get(i);
                        return out;
                    }
                    py::array_t<bool> out({n});
                    auto r = out.mutable_unchecked<1>();
                    for (uint64_t i = 0; i < n_u; ++i) r(i) = vb->get(i);
                    return out;
                }

                // Default: float64
                if (as_row) {
                    py::array_t<double> out({static_cast<py::ssize_t>(1), n});
                    auto r = out.mutable_unchecked<2>();
                    for (uint64_t i = 0; i < n_u; ++i) r(0, i) = v.get_element_as_double(i);
                    return out;
                }
                py::array_t<double> out({n});
                auto r = out.mutable_unchecked<1>();
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
            [](const VectorBase& a, const std::shared_ptr<VectorBase>& b) -> py::object {
                const DataType dt_a = a.get_data_type();
                const DataType dt_b = b->get_data_type();
                const bool complex =
                    (dt_a == DataType::COMPLEX_FLOAT16 || dt_a == DataType::COMPLEX_FLOAT32 || dt_a == DataType::COMPLEX_FLOAT64 ||
                     dt_b == DataType::COMPLEX_FLOAT16 || dt_b == DataType::COMPLEX_FLOAT32 || dt_b == DataType::COMPLEX_FLOAT64);
                if (complex) {
                    return py::cast(pycauset::dot_product_complex(a, *b));
                }
                return py::float_(pycauset::dot_product(a, *b));
            },
            py::arg("other"))
        .def(
            "__add__",
            [](const VectorBase& a, const std::shared_ptr<VectorBase>& b) {
                maybe_warn_bit_promotes_to_int32(
                    "add",
                    promotion::BinaryOp::Add,
                    a.get_data_type(),
                    b->get_data_type());
                auto out = pycauset::add_vectors(a, *b, "");
                return std::shared_ptr<VectorBase>(out.release());
            },
            py::is_operator())
        .def(
            "__add__",
            [](const VectorBase& a, const py::array& b) {
                auto vb = vector_from_numpy(b);
                maybe_warn_bit_promotes_to_int32(
                    "add",
                    promotion::BinaryOp::Add,
                    a.get_data_type(),
                    vb->get_data_type());
                auto out = pycauset::add_vectors(a, *vb, "");
                return std::shared_ptr<VectorBase>(out.release());
            },
            py::is_operator())
        .def(
            "__add__",
            [](const VectorBase& a, double s) {
                auto out = pycauset::scalar_add_vector(a, s, "");
                return std::shared_ptr<VectorBase>(out.release());
            },
            py::is_operator())
        .def(
            "__add__",
            [](const VectorBase& a, int64_t s) {
                auto out = pycauset::scalar_add_vector(a, s, "");
                return std::shared_ptr<VectorBase>(out.release());
            },
            py::is_operator())
        .def(
            "__radd__",
            [](const VectorBase& a, double s) {
                auto out = pycauset::scalar_add_vector(a, s, "");
                return std::shared_ptr<VectorBase>(out.release());
            },
            py::is_operator())
        .def(
            "__radd__",
            [](const VectorBase& a, int64_t s) {
                auto out = pycauset::scalar_add_vector(a, s, "");
                return std::shared_ptr<VectorBase>(out.release());
            },
            py::is_operator())
        .def(
            "__radd__",
            [](const VectorBase& a, const py::array& b) {
                auto vb = vector_from_numpy(b);
                maybe_warn_bit_promotes_to_int32(
                    "add",
                    promotion::BinaryOp::Add,
                    vb->get_data_type(),
                    a.get_data_type());
                auto out = pycauset::add_vectors(*vb, a, "");
                return std::shared_ptr<VectorBase>(out.release());
            },
            py::is_operator())
        .def(
            "__sub__",
            [](const VectorBase& a, const std::shared_ptr<VectorBase>& b) {
                maybe_warn_bit_promotes_to_int32(
                    "subtract",
                    promotion::BinaryOp::Subtract,
                    a.get_data_type(),
                    b->get_data_type());
                auto out = pycauset::subtract_vectors(a, *b, "");
                return std::shared_ptr<VectorBase>(out.release());
            },
            py::is_operator())
        .def(
            "__mul__",
            [](const VectorBase& a, double s) {
                auto out = pycauset::scalar_multiply_vector(a, s, "");
                return std::shared_ptr<VectorBase>(out.release());
            },
            py::is_operator())
        .def(
            "__mul__",
            [](const VectorBase& a, int64_t s) {
                auto out = pycauset::scalar_multiply_vector(a, s, "");
                return std::shared_ptr<VectorBase>(out.release());
            },
            py::is_operator())
        .def(
            "__mul__",
            [](const VectorBase& a, std::complex<double> s) {
                auto out = pycauset::scalar_multiply_vector(a, s, "");
                return std::shared_ptr<VectorBase>(out.release());
            },
            py::is_operator())
        .def(
            "__rmul__",
            [](const VectorBase& a, double s) {
                auto out = pycauset::scalar_multiply_vector(a, s, "");
                return std::shared_ptr<VectorBase>(out.release());
            },
            py::is_operator())
        .def(
            "__rmul__",
            [](const VectorBase& a, int64_t s) {
                auto out = pycauset::scalar_multiply_vector(a, s, "");
                return std::shared_ptr<VectorBase>(out.release());
            },
            py::is_operator())
        .def(
            "__rmul__",
            [](const VectorBase& a, std::complex<double> s) {
                auto out = pycauset::scalar_multiply_vector(a, s, "");
                return std::shared_ptr<VectorBase>(out.release());
            },
            py::is_operator())
        .def(
            "__matmul__",
            [](const VectorBase& a, const std::shared_ptr<VectorBase>& b) -> py::object {
                // Col @ Row -> Outer product; otherwise dot.
                if (!a.is_transposed() && b->is_transposed()) {
                    auto out = pycauset::outer_product(a, *b, "");
                    return py::cast(std::shared_ptr<MatrixBase>(out.release()));
                }
                // Type-stable dot for integer-like vectors
                if (auto* ai = dynamic_cast<const DenseVector<int32_t>*>(&a)) {
                    if (auto* bi = dynamic_cast<const DenseVector<int32_t>*>(b.get())) {
                        bindings_warn::warn_once_with_category(
                            "pycauset.dot.accum_widen.int32",
                            "pycauset dot: using int64 accumulator for int32 @ int32 to avoid intermediate overflow; raises on accumulator overflow.",
                            "PyCausetDTypeWarning",
                            /*stacklevel=*/3);
                        int64_t sum = 0;
                        for (uint64_t i = 0; i < a.size(); ++i) {
                            int64_t term = static_cast<int64_t>(ai->get(i)) * static_cast<int64_t>(bi->get(i));
                            sum = checked_add_int64(sum, term);
                        }
                        return py::int_(sum);
                    }
                }
                if (auto* ai16 = dynamic_cast<const DenseVector<int16_t>*>(&a)) {
                    if (auto* bi16 = dynamic_cast<const DenseVector<int16_t>*>(b.get())) {
                        bindings_warn::warn_once_with_category(
                            "pycauset.dot.accum_widen.int16",
                            "pycauset dot: using int64 accumulator for int16 @ int16 to avoid intermediate overflow; raises on accumulator overflow.",
                            "PyCausetDTypeWarning",
                            /*stacklevel=*/3);
                        int64_t sum = 0;
                        for (uint64_t i = 0; i < a.size(); ++i) {
                            int64_t term = static_cast<int64_t>(ai16->get(i)) * static_cast<int64_t>(bi16->get(i));
                            sum = checked_add_int64(sum, term);
                        }
                        return py::int_(sum);
                    }
                }
                if (auto* ab = dynamic_cast<const DenseVector<bool>*>(&a)) {
                    if (auto* bb = dynamic_cast<const DenseVector<bool>*>(b.get())) {
                        int64_t sum = 0;
                        for (uint64_t i = 0; i < a.size(); ++i) {
                            sum = checked_add_int64(sum, (ab->get(i) && bb->get(i)) ? 1 : 0);
                        }
                        return py::int_(sum);
                    }
                }

                const DataType dt_a = a.get_data_type();
                const DataType dt_b = b->get_data_type();
                const bool complex =
                    (dt_a == DataType::COMPLEX_FLOAT16 || dt_a == DataType::COMPLEX_FLOAT32 || dt_a == DataType::COMPLEX_FLOAT64 ||
                     dt_b == DataType::COMPLEX_FLOAT16 || dt_b == DataType::COMPLEX_FLOAT32 || dt_b == DataType::COMPLEX_FLOAT64);
                if (complex) {
                    return py::cast(pycauset::dot_product_complex(a, *b));
                }

                return py::float_(pycauset::dot_product(a, *b));
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

    py::class_<DenseVector<double>, VectorBase, std::shared_ptr<DenseVector<double>>>(m, "FloatVector")
        .def(py::init<uint64_t>(), py::arg("n"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseVector<double>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseVector<double>::get)
        .def("set", &DenseVector<double>::set)
        .def("__getitem__", [](const DenseVector<double>& v, uint64_t i) { return v.get(i); })
        .def("__setitem__", [](DenseVector<double>& v, uint64_t i, double val) { v.set(i, val); });

    py::class_<DenseVector<float>, VectorBase, std::shared_ptr<DenseVector<float>>>(m, "Float32Vector")
        .def(py::init<uint64_t>(), py::arg("n"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseVector<float>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseVector<float>::get)
        .def("set", &DenseVector<float>::set)
        .def("__getitem__", [](const DenseVector<float>& v, uint64_t i) { return v.get(i); })
        .def("__setitem__", [](DenseVector<float>& v, uint64_t i, float val) { v.set(i, val); });

    py::class_<DenseVector<float16_t>, VectorBase, std::shared_ptr<DenseVector<float16_t>>>(m, "Float16Vector")
        .def(py::init<uint64_t>(), py::arg("n"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseVector<float16_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def(
            "get",
            [](const DenseVector<float16_t>& v, uint64_t i) { return static_cast<float>(v.get(i)); })
        .def(
            "set",
            [](DenseVector<float16_t>& v, uint64_t i, double val) { v.set(i, float16_t(val)); })
        .def("__getitem__", [](const DenseVector<float16_t>& v, uint64_t i) { return static_cast<float>(v.get(i)); })
        .def("__setitem__", [](DenseVector<float16_t>& v, uint64_t i, double val) { v.set(i, float16_t(val)); });

    py::class_<ComplexFloat16Vector, VectorBase, std::shared_ptr<ComplexFloat16Vector>>(m, "ComplexFloat16Vector")
        .def(py::init<uint64_t>(), py::arg("n"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<ComplexFloat16Vector>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", [](const ComplexFloat16Vector& v, uint64_t i) { return v.get(i); })
        .def("set", [](ComplexFloat16Vector& v, uint64_t i, std::complex<double> val) { v.set(i, val); })
        .def("__getitem__", [](const ComplexFloat16Vector& v, uint64_t i) { return v.get(i); })
        .def("__setitem__", [](ComplexFloat16Vector& v, uint64_t i, std::complex<double> val) { v.set(i, val); });

    py::class_<DenseVector<std::complex<float>>, VectorBase, std::shared_ptr<DenseVector<std::complex<float>>>>(m, "ComplexFloat32Vector")
        .def(py::init<uint64_t>(), py::arg("n"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseVector<std::complex<float>>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseVector<std::complex<float>>::get)
        .def("set", &DenseVector<std::complex<float>>::set)
        .def("__getitem__", [](const DenseVector<std::complex<float>>& v, uint64_t i) { return v.get(i); })
        .def("__setitem__", [](DenseVector<std::complex<float>>& v, uint64_t i, std::complex<double> val) {
            v.set(i, std::complex<float>(static_cast<float>(val.real()), static_cast<float>(val.imag())));
        });

    py::class_<DenseVector<std::complex<double>>, VectorBase, std::shared_ptr<DenseVector<std::complex<double>>>>(m, "ComplexFloat64Vector")
        .def(py::init<uint64_t>(), py::arg("n"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseVector<std::complex<double>>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseVector<std::complex<double>>::get)
        .def("set", &DenseVector<std::complex<double>>::set)
        .def("__getitem__", [](const DenseVector<std::complex<double>>& v, uint64_t i) { return v.get(i); })
        .def("__setitem__", [](DenseVector<std::complex<double>>& v, uint64_t i, std::complex<double> val) { v.set(i, val); });

    py::class_<DenseVector<int32_t>, VectorBase, std::shared_ptr<DenseVector<int32_t>>>(m, "IntegerVector")
        .def(py::init<uint64_t>(), py::arg("n"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseVector<int32_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseVector<int32_t>::get)
        .def("set", &DenseVector<int32_t>::set)
        .def("__getitem__", [](const DenseVector<int32_t>& v, uint64_t i) { return v.get(i); })
        .def("__setitem__", [](DenseVector<int32_t>& v, uint64_t i, int32_t val) { v.set(i, val); });

    py::class_<DenseVector<int16_t>, VectorBase, std::shared_ptr<DenseVector<int16_t>>>(m, "Int16Vector")
        .def(py::init<uint64_t>(), py::arg("n"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseVector<int16_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseVector<int16_t>::get)
        .def("set", &DenseVector<int16_t>::set)
        .def("__getitem__", [](const DenseVector<int16_t>& v, uint64_t i) { return v.get(i); })
        .def("__setitem__", [](DenseVector<int16_t>& v, uint64_t i, int16_t val) { v.set(i, val); });

    py::class_<DenseVector<int8_t>, VectorBase, std::shared_ptr<DenseVector<int8_t>>>(m, "Int8Vector")
        .def(py::init<uint64_t>(), py::arg("n"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseVector<int8_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseVector<int8_t>::get)
        .def("set", &DenseVector<int8_t>::set)
        .def("__getitem__", [](const DenseVector<int8_t>& v, uint64_t i) { return v.get(i); })
        .def("__setitem__", [](DenseVector<int8_t>& v, uint64_t i, int8_t val) { v.set(i, val); });

    py::class_<DenseVector<int64_t>, VectorBase, std::shared_ptr<DenseVector<int64_t>>>(m, "Int64Vector")
        .def(py::init<uint64_t>(), py::arg("n"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseVector<int64_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseVector<int64_t>::get)
        .def("set", &DenseVector<int64_t>::set)
        .def("__getitem__", [](const DenseVector<int64_t>& v, uint64_t i) { return v.get(i); })
        .def("__setitem__", [](DenseVector<int64_t>& v, uint64_t i, int64_t val) { v.set(i, val); });

    py::class_<DenseVector<uint8_t>, VectorBase, std::shared_ptr<DenseVector<uint8_t>>>(m, "UInt8Vector")
        .def(py::init<uint64_t>(), py::arg("n"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseVector<uint8_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseVector<uint8_t>::get)
        .def("set", &DenseVector<uint8_t>::set)
        .def("__getitem__", [](const DenseVector<uint8_t>& v, uint64_t i) { return v.get(i); })
        .def("__setitem__", [](DenseVector<uint8_t>& v, uint64_t i, uint8_t val) { v.set(i, val); });

    py::class_<DenseVector<uint16_t>, VectorBase, std::shared_ptr<DenseVector<uint16_t>>>(m, "UInt16Vector")
        .def(py::init<uint64_t>(), py::arg("n"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseVector<uint16_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseVector<uint16_t>::get)
        .def("set", &DenseVector<uint16_t>::set)
        .def("__getitem__", [](const DenseVector<uint16_t>& v, uint64_t i) { return v.get(i); })
        .def("__setitem__", [](DenseVector<uint16_t>& v, uint64_t i, uint16_t val) { v.set(i, val); });

    py::class_<DenseVector<uint32_t>, VectorBase, std::shared_ptr<DenseVector<uint32_t>>>(m, "UInt32Vector")
        .def(py::init<uint64_t>(), py::arg("n"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseVector<uint32_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseVector<uint32_t>::get)
        .def("set", &DenseVector<uint32_t>::set)
        .def("__getitem__", [](const DenseVector<uint32_t>& v, uint64_t i) { return v.get(i); })
        .def("__setitem__", [](DenseVector<uint32_t>& v, uint64_t i, uint32_t val) { v.set(i, val); });

    py::class_<DenseVector<uint64_t>, VectorBase, std::shared_ptr<DenseVector<uint64_t>>>(m, "UInt64Vector")
        .def(py::init<uint64_t>(), py::arg("n"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseVector<uint64_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseVector<uint64_t>::get)
        .def("set", &DenseVector<uint64_t>::set)
        .def("__getitem__", [](const DenseVector<uint64_t>& v, uint64_t i) { return v.get(i); })
        .def("__setitem__", [](DenseVector<uint64_t>& v, uint64_t i, uint64_t val) { v.set(i, val); });

    py::class_<DenseVector<bool>, VectorBase, std::shared_ptr<DenseVector<bool>>>(m, "BitVector")
        .def(py::init<uint64_t>(), py::arg("n"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<DenseVector<bool>>(n, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def("get", &DenseVector<bool>::get)
        .def("set", &DenseVector<bool>::set)
        .def("__getitem__", [](const DenseVector<bool>& v, uint64_t i) { return v.get(i); })
        .def("__setitem__", [](DenseVector<bool>& v, uint64_t i, bool val) { v.set(i, val); });

    py::class_<UnitVector, VectorBase, std::shared_ptr<UnitVector>>(m, "UnitVector")
        .def(
            py::init<uint64_t, uint64_t>(),
            py::arg("n"),
            py::arg("active_index"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               uint64_t active_index,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed) {
                return std::make_shared<UnitVector>(n, active_index, backing_file, offset, seed, scalar, is_transposed);
            },
            py::arg("n"),
            py::arg("active_index"),
            py::arg("backing_file"),
            py::arg("offset"),
            py::arg("seed"),
            py::arg("scalar"),
            py::arg("is_transposed"))
        .def_property_readonly("active_index", &UnitVector::get_active_index)
        .def("get", [](const UnitVector& v, uint64_t i) { return v.get_element_as_double(i); })
        .def("__getitem__", [](const UnitVector& v, uint64_t i) { return v.get_element_as_double(i); });

    // Encourage NumPy to prefer VectorBase's reverse ops over coercion.
    py::object vb_cls = m.attr("VectorBase");
    vb_cls.attr("__array_priority__") = py::float_(1000.0);
}
