#pragma once
#include "pycauset/math/Reductions.hpp"
// Python value -> matrix/vector element, with NumPy assignment semantics: integers must fit
// the dtype (error otherwise, as NumPy 2); floats assigned to integer dtypes truncate toward
// zero; complex values can't go into real dtypes; bit dtypes take 0/1/False/True only.
#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <type_traits>

#include "bindings_common.hpp"
#include "pycauset/core/Float16.hpp"

namespace pycauset_bindings {

[[noreturn]] inline void throw_overflow(const std::string& msg) {
    PyErr_SetString(PyExc_OverflowError, msg.c_str());
    throw py::error_already_set();
}

// A value stored into a bool element: its truth value, as NumPy's bool assignment (B[0, 0] = 2.5
// is True, = None is False). Only 0/1/False/True were accepted, a TypeError otherwise (review
// 3.7 C-6: NumPy is the oracle).
inline bool coerce_bit_value(const py::handle& value) {
    const int t = PyObject_IsTrue(value.ptr());
    if (t < 0) throw py::error_already_set();
    return t == 1;
}

// float(value), with Python's own errors: a string that is no number is a ValueError and a
// non-number a TypeError, as NumPy's element assignment (pybind11's cast raised RuntimeError,
// review 3.7 C-6). Numeric strings convert, as in NumPy.
inline double py_float(const py::object& value) {
    PyObject* f = PyNumber_Float(value.ptr());
    if (f == nullptr) throw py::error_already_set();
    const double d = PyFloat_AsDouble(f);
    Py_DECREF(f);
    return d;
}

template <typename E>
E py_to_element(const py::object& value) {
    if constexpr (std::is_same_v<E, bool>) {
        return coerce_bit_value(value);
    } else if constexpr (std::is_integral_v<E>) {
        if (PyComplex_Check(value.ptr())) throw py::type_error("can't assign a complex value to an integer matrix");
        if (PyUnicode_Check(value.ptr())) {  // int(value), as NumPy: '3' is 3, '1.5' a ValueError
            PyObject* i = PyNumber_Long(value.ptr());
            if (i == nullptr) throw py::error_already_set();
            return py_to_element<E>(py::reinterpret_steal<py::object>(i));
        }
        if (!PyIndex_Check(value.ptr())) {  // floats (Python or NumPy): truncate like NumPy
            const double d = py_float(value);
            if (!std::isfinite(d)) throw py::value_error("can't assign a non-finite value to an integer matrix");
            const long double t = std::trunc(static_cast<long double>(d));
            if (t < static_cast<long double>(std::numeric_limits<E>::min()) ||
                t > static_cast<long double>(std::numeric_limits<E>::max())) {
                throw_overflow("value out of range for the matrix dtype");
            }
            return static_cast<E>(t);
        }
        py::int_ iv(value);  // accepts Python ints, bools and NumPy integer scalars
        if constexpr (std::is_signed_v<E>) {
            int overflow = 0;
            const long long v = PyLong_AsLongLongAndOverflow(iv.ptr(), &overflow);
            if (overflow != 0 || v < std::numeric_limits<E>::min() || v > std::numeric_limits<E>::max()) {
                throw_overflow("Python integer " + py::str(iv).cast<std::string>() + " out of bounds for the matrix dtype");
            }
            return static_cast<E>(v);
        } else {
            if (PyObject_RichCompareBool(iv.ptr(), py::int_(0).ptr(), Py_LT) == 1) {
                throw_overflow("Python integer " + py::str(iv).cast<std::string>() + " out of bounds for the matrix dtype");
            }
            const unsigned long long v = PyLong_AsUnsignedLongLong(iv.ptr());
            if (PyErr_Occurred() || v > std::numeric_limits<E>::max()) {
                PyErr_Clear();
                throw_overflow("Python integer " + py::str(iv).cast<std::string>() + " out of bounds for the matrix dtype");
            }
            return static_cast<E>(v);
        }
    } else if constexpr (std::is_same_v<E, pycauset::float16_t>) {
        if (PyComplex_Check(value.ptr())) throw py::type_error("can't assign a complex value to a real matrix");
        return pycauset::float16_t(value.is_none() ? std::numeric_limits<double>::quiet_NaN() : py_float(value));
    } else if constexpr (std::is_floating_point_v<E>) {
        // None is NaN for float dtypes, as in NumPy (it raised RuntimeError, review 3.7 C-6).
        if (PyComplex_Check(value.ptr())) throw py::type_error("can't assign a complex value to a real matrix");
        return value.is_none() ? std::numeric_limits<E>::quiet_NaN() : static_cast<E>(py_float(value));
    } else {
        // complex(value), with Python's errors and its string parsing ('1+2j'), as NumPy.
        if (value.is_none()) return E(std::numeric_limits<double>::quiet_NaN(), 0.0);
        static py::object* complex_type = new py::object(py::module_::import("builtins").attr("complex"));
        return static_cast<E>((*complex_type)(value).cast<std::complex<double>>());
    }
}

// Exact Python int from a 128-bit integer (results of exact integer reductions).
inline py::object int128_to_py(__int128 v) {
    if (v >= static_cast<__int128>(std::numeric_limits<int64_t>::min()) &&
        v <= static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
        return py::int_(static_cast<int64_t>(v));
    }
    const bool neg = v < 0;
    const unsigned __int128 mag = neg ? static_cast<unsigned __int128>(-(v + 1)) + 1 : static_cast<unsigned __int128>(v);
    py::object out = py::int_(static_cast<uint64_t>(mag >> 64)).attr("__lshift__")(64).attr("__or__")(
        py::int_(static_cast<uint64_t>(mag)));
    return neg ? out.attr("__neg__")() : out;
}

// NumPy result kinds of a reduction: exact int for integer/bit, float for real, complex for
// complex (F-051, F-059).
inline py::object reduction_to_py(const pycauset::ReductionValue& r) {
    switch (r.kind) {
        case pycauset::ReductionValue::Kind::Int: return int128_to_py(r.i);
        case pycauset::ReductionValue::Kind::Real: return py::float_(r.r);
        case pycauset::ReductionValue::Kind::Complex: return py::cast(r.c);
    }
    return py::none();
}

}  // namespace pycauset_bindings
