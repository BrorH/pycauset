#pragma once
// Python scalar -> pycauset::ScalarArg (NumPy 2 / NEP 50 weak/strong scalar rules), shared by
// the matrix and vector bindings.

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "bindings_common.hpp"
#include "numpy_dtypes.hpp"
#include "pycauset/math/ScalarOps.hpp"
#include "pycauset/core/PromotionResolver.hpp"

namespace {

// NumPy is a dependency: its import failing is an error, not "not a NumPy scalar".
inline bool is_numpy_scalar(const py::handle& obj) {
    static py::object np_generic = py::module_::import("numpy").attr("generic");
    return py::isinstance(obj, np_generic);
}

// --- Scalar arguments (F-001/F-002) -------------------------------------------------
// Converts a Python int: exactly in [INT64_MIN, UINT64_MAX]; beyond, a float marked
// `beyond_64_bits` (scalar_op raises OverflowError for integer matrices, as NumPy 2; with a
// float matrix it is a float, so float64 M + 2**64 works; it raised before the dtype was known).
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
    const double d = PyLong_AsDouble(obj.ptr());  // OverflowError beyond float64, as NumPy
    if (d == -1.0 && PyErr_Occurred()) throw py::error_already_set();
    pycauset::ScalarArg out = pycauset::ScalarArg::from_double(d);
    out.beyond_64_bits = true;
    return out;
}

// True if `obj` is a Python bool/int/float/complex or a NumPy scalar; fills `out`.
inline bool to_scalar_arg(const py::handle& obj, pycauset::ScalarArg& out) {
    PyObject* p = obj.ptr();
    // A bool is a bool operand (review 3.6 C-13): B * True is bool, as B * B; it was an int, so
    // B * True was int32.
    if (PyBool_Check(p)) {
        out = pycauset::ScalarArg::from_int64(p == Py_True ? 1 : 0);
        out.strong = true;
        out.strong_dtype = pycauset::DataType::BIT;
        return true;
    }
    if (PyLong_Check(p)) { out = scalar_arg_from_pyint(obj); return true; }
    // NumPy scalars before Python float / complex: np.float64 and np.complex128 are subclasses
    // of those and were taken as weak Python scalars (float32 matrix + np.float64(1) stayed
    // float32; NumPy gives float64, review 3.3 C-4).
    if (!PyFloat_CheckExact(p) && !PyComplex_CheckExact(p) && is_numpy_scalar(obj)) {
        const pycauset::DataType dt = pycauset_bindings::dtype_from_numpy(obj.attr("dtype").cast<py::dtype>());
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
        out.strong = true;
        out.strong_dtype = dt;
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

// The promotion operator and name of an arithmetic operator (for the dtype warnings).
inline pycauset::promotion::BinaryOp arith_binary_op(Arith op) {
    namespace promotion = pycauset::promotion;
    switch (op) {
        case Arith::Add: return promotion::BinaryOp::Add;
        case Arith::Sub: return promotion::BinaryOp::Subtract;
        case Arith::Mul: return promotion::BinaryOp::ElementwiseMultiply;
        case Arith::Div: return promotion::BinaryOp::Divide;
    }
    return promotion::BinaryOp::Add;
}

inline const char* arith_name(Arith op) {
    switch (op) {
        case Arith::Add: return "add";
        case Arith::Sub: return "subtract";
        case Arith::Mul: return "elementwise_multiply";
        case Arith::Div: return "elementwise_divide";
    }
    return "";
}

}  // namespace
