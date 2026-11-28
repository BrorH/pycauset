#pragma once
// Python scalar -> pycauset::ScalarArg (NumPy 2 / NEP 50 weak/strong scalar rules), shared by
// the matrix and vector bindings.

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "bindings_common.hpp"
#include "pycauset/math/ScalarOps.hpp"

namespace {

// NumPy is a dependency: its import failing is an error, not "not a NumPy scalar".
inline bool is_numpy_scalar(const py::handle& obj) {
    static py::object np_generic = py::module_::import("numpy").attr("generic");
    return py::isinstance(obj, np_generic);
}

// --- Scalar arguments (F-001/F-002) -------------------------------------------------
// Converts a Python int of any size in [INT64_MIN, UINT64_MAX]; larger ones raise
// OverflowError like NumPy.
inline pycauset::ScalarArg scalar_arg_from_pyint(const py::handle& obj) {
    int overflow = 0;
    const long long v = PyLong_AsLongLongAndOverflow(obj.ptr(), &overflow);
    if (overflow == 0) {
        if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
        return pycauset::ScalarArg::from_int64(static_cast<int64_t>(v));
    }
    if (overflow > 0) {
        const unsigned long long u = PyLong_AsUnsignedLongLong(obj.ptr());
        if (!PyErr_Occurred()) return pycauset::ScalarArg::from_uint64(static_cast<uint64_t>(u));
        PyErr_Clear();
    }
    throw std::overflow_error("Python integer out of bounds for 64-bit matrix dtypes");
}

inline pycauset::DataType numpy_scalar_dtype(const py::handle& obj) {
    const std::string name = py::str(obj.attr("dtype").attr("name")).cast<std::string>();
    if (name == "bool") return pycauset::DataType::BIT;
    if (name == "int8") return pycauset::DataType::INT8;
    if (name == "int16") return pycauset::DataType::INT16;
    if (name == "int32") return pycauset::DataType::INT32;
    if (name == "int64") return pycauset::DataType::INT64;
    if (name == "uint8") return pycauset::DataType::UINT8;
    if (name == "uint16") return pycauset::DataType::UINT16;
    if (name == "uint32") return pycauset::DataType::UINT32;
    if (name == "uint64") return pycauset::DataType::UINT64;
    if (name == "float16") return pycauset::DataType::FLOAT16;
    if (name == "float32") return pycauset::DataType::FLOAT32;
    if (name == "float64") return pycauset::DataType::FLOAT64;
    if (name == "complex64") return pycauset::DataType::COMPLEX_FLOAT32;
    if (name == "complex128") return pycauset::DataType::COMPLEX_FLOAT64;
    return pycauset::DataType::UNKNOWN;
}

// True if `obj` is a Python bool/int/float/complex or a NumPy scalar; fills `out`.
inline bool to_scalar_arg(const py::handle& obj, pycauset::ScalarArg& out) {
    PyObject* p = obj.ptr();
    if (PyBool_Check(p)) { out = pycauset::ScalarArg::from_int64(p == Py_True ? 1 : 0); return true; }
    if (PyLong_Check(p)) { out = scalar_arg_from_pyint(obj); return true; }
    // NumPy scalars before Python float / complex: np.float64 and np.complex128 are subclasses
    // of those and were taken as weak Python scalars (float32 matrix + np.float64(1) stayed
    // float32; NumPy gives float64, review 3.3 C-4).
    if (!PyFloat_CheckExact(p) && !PyComplex_CheckExact(p) && is_numpy_scalar(obj)) {
        const pycauset::DataType dt = numpy_scalar_dtype(obj);
        if (dt == pycauset::DataType::UNKNOWN) return false;
        if (dt == pycauset::DataType::COMPLEX_FLOAT32 || dt == pycauset::DataType::COMPLEX_FLOAT64) {
            out = pycauset::ScalarArg::from_complex(obj.cast<std::complex<double>>());
        } else if (dt == pycauset::DataType::FLOAT16 || dt == pycauset::DataType::FLOAT32 || dt == pycauset::DataType::FLOAT64) {
            out = pycauset::ScalarArg::from_double(obj.cast<double>());
        } else if (dt == pycauset::DataType::UINT64) {
            out = pycauset::ScalarArg::from_uint64(obj.cast<uint64_t>());
        } else {
            out = pycauset::ScalarArg::from_int64(obj.cast<int64_t>());
        }
        if (dt != pycauset::DataType::BIT) {
            out.strong = true;
            out.strong_dtype = dt;
        }
        return true;
    }
    if (PyFloat_Check(p)) { out = pycauset::ScalarArg::from_double(PyFloat_AsDouble(p)); return true; }
    if (PyComplex_Check(p)) { out = pycauset::ScalarArg::from_complex(obj.cast<std::complex<double>>()); return true; }
    return false;
}

// Arithmetic operator kinds shared by the matrix and vector operator bindings.
enum class Arith { Add, Sub, Mul, Div };

inline pycauset::ScalarOpKind scalar_kind(Arith op, bool reflected) {
    switch (op) {
        case Arith::Add: return pycauset::ScalarOpKind::Add;
        case Arith::Sub: return reflected ? pycauset::ScalarOpKind::RSub : pycauset::ScalarOpKind::Sub;
        case Arith::Mul: return pycauset::ScalarOpKind::Mul;
        case Arith::Div: return reflected ? pycauset::ScalarOpKind::RTrueDiv : pycauset::ScalarOpKind::TrueDiv;
    }
    return pycauset::ScalarOpKind::Add;
}

}  // namespace
