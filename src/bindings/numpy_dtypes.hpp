#pragma once
// The one table between engine dtypes and NumPy's (review 3.7 C-12: the importer, the scalar
// reader, the exporter and the export guard each had their own). Internal to src/bindings.

#include <pybind11/numpy.h>

#include "pycauset/core/Types.hpp"

namespace pycauset_bindings {

struct NumpyDtype {
    pycauset::DataType dt;
    char kind;          // NumPy's dtype.kind
    int itemsize;       // bytes of one element in NumPy
    const char* name;   // the NumPy dtype a matrix of `dt` exports as
};

// complex_float16 has no NumPy dtype: it exports as complex64 (and NumPy never imports as it).
inline constexpr NumpyDtype kNumpyDtypes[] = {
    {pycauset::DataType::BIT, 'b', 1, "bool"},
    {pycauset::DataType::INT8, 'i', 1, "int8"},
    {pycauset::DataType::INT16, 'i', 2, "int16"},
    {pycauset::DataType::INT32, 'i', 4, "int32"},
    {pycauset::DataType::INT64, 'i', 8, "int64"},
    {pycauset::DataType::UINT8, 'u', 1, "uint8"},
    {pycauset::DataType::UINT16, 'u', 2, "uint16"},
    {pycauset::DataType::UINT32, 'u', 4, "uint32"},
    {pycauset::DataType::UINT64, 'u', 8, "uint64"},
    {pycauset::DataType::FLOAT16, 'f', 2, "float16"},
    {pycauset::DataType::FLOAT32, 'f', 4, "float32"},
    {pycauset::DataType::FLOAT64, 'f', 8, "float64"},
    {pycauset::DataType::COMPLEX_FLOAT32, 'c', 8, "complex64"},
    {pycauset::DataType::COMPLEX_FLOAT64, 'c', 16, "complex128"},
    {pycauset::DataType::COMPLEX_FLOAT16, '\0', 8, "complex64"},  // export only
};

// The engine dtype of a NumPy dtype (UNKNOWN for dtypes the engine does not hold).
inline pycauset::DataType dtype_from_numpy(const pybind11::dtype& d) {
    const char k = d.kind();
    const auto n = static_cast<int>(d.itemsize());
    for (const auto& e : kNumpyDtypes)
        if (e.kind == k && e.itemsize == n) return e.dt;
    return pycauset::DataType::UNKNOWN;
}

inline const NumpyDtype* numpy_dtype_of(pycauset::DataType dt) {
    for (const auto& e : kNumpyDtypes)
        if (e.dt == dt) return &e;
    return nullptr;
}

}  // namespace pycauset_bindings
