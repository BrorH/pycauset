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

inline bool coerce_bit_value(const py::handle& value) {
    // Accepts Python and NumPy booleans, integers and floats equal to 0 or 1.
    if (PyBool_Check(value.ptr())) return value.ptr() == Py_True;
    const std::string type_name = py::str(py::type::of(value).attr("__name__"));
    if (type_name == "bool" || type_name == "bool_") return value.cast<bool>();
    if (!PyComplex_Check(value.ptr()) && PyNumber_Check(value.ptr())) {
        const double v = py::reinterpret_borrow<py::object>(value).cast<double>();
        if (v == 0.0) return false;
        if (v == 1.0) return true;
    }
    throw py::type_error("bit (dtype 'bool') matrices only accept 0/1/False/True");
}

template <typename E>
E py_to_element(const py::object& value) {
    if constexpr (std::is_same_v<E, bool>) {
        return coerce_bit_value(value);
    } else if constexpr (std::is_integral_v<E>) {
        if (PyComplex_Check(value.ptr())) throw py::type_error("can't assign a complex value to an integer matrix");
        if (!PyIndex_Check(value.ptr())) {  // floats (Python or NumPy): truncate like NumPy
            const double d = value.cast<double>();
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
        return pycauset::float16_t(value.cast<double>());
    } else if constexpr (std::is_floating_point_v<E>) {
        if (PyComplex_Check(value.ptr())) throw py::type_error("can't assign a complex value to a real matrix");
        return static_cast<E>(value.cast<double>());
    } else {
        return static_cast<E>(value.cast<std::complex<double>>());
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
