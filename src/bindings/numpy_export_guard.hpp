#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <string>
#include <stdexcept>

#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "pycauset/core/Types.hpp"
#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/vector/VectorBase.hpp"

namespace pycauset::bindings {

// Global ceiling in bytes. -1 disables the size check.
// Note: this is a process-wide setting (matches the Python-side guard behavior).
inline std::atomic<int64_t> g_numpy_export_max_bytes{-1};

inline bool ends_with(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) {
        return false;
    }
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline bool is_snapshot_path(const std::string& path) {
    return ends_with(path, ".pycauset");
}

inline bool is_file_backed_non_snapshot(const std::string& backing_file) {
    if (backing_file.empty() || backing_file == ":memory:") {
        return false;
    }
    return !is_snapshot_path(backing_file);
}

inline uint64_t numpy_output_itemsize_bytes(DataType dt) {
    // Must reflect actual NumPy materialization size, not packed storage size.
    switch (dt) {
        case DataType::BIT:
            return 1; // exported as bool
        case DataType::INT8:
        case DataType::UINT8:
            return 1;
        case DataType::INT16:
        case DataType::UINT16:
            return 2;
        case DataType::INT32:
        case DataType::UINT32:
            return 4;
        case DataType::INT64:
        case DataType::UINT64:
            return 8;
        case DataType::FLOAT16:
            return 2;
        case DataType::FLOAT32:
            return 4;
        case DataType::FLOAT64:
            return 8;
        case DataType::COMPLEX_FLOAT16:
        case DataType::COMPLEX_FLOAT32:
            return 8; // exported as complex64
        case DataType::COMPLEX_FLOAT64:
            return 16; // exported as complex128
        default:
            // Be conservative.
            return 8;
    }
}

inline bool mul_overflow_u64(uint64_t a, uint64_t b, uint64_t* out) {
    if (a == 0 || b == 0) {
        *out = 0;
        return false;
    }
    if (a > (std::numeric_limits<uint64_t>::max)() / b) {
        return true;
    }
    *out = a * b;
    return false;
}

inline void ensure_numpy_export_allowed_impl(const std::string& backing_file, uint64_t elem_count, DataType dt, bool elem_overflow) {
    if (is_file_backed_non_snapshot(backing_file)) {
        throw std::runtime_error(
            "Export to NumPy is blocked for file-backed/out-of-core objects; pass allow_huge=True via pycauset.to_numpy(...) to override.");
    }

    const int64_t limit = g_numpy_export_max_bytes.load();
    if (limit < 0) {
        return;
    }

    if (elem_overflow) {
        throw std::runtime_error(
            "Export to NumPy exceeds configured materialization limit; pass allow_huge=True via pycauset.to_numpy(...) to override.");
    }

    const uint64_t itemsize = numpy_output_itemsize_bytes(dt);

    uint64_t est_bytes_u64 = 0;
    if (mul_overflow_u64(elem_count, itemsize, &est_bytes_u64)) {
        throw std::runtime_error(
            "Export to NumPy exceeds configured materialization limit; pass allow_huge=True via pycauset.to_numpy(...) to override.");
    }

    const uint64_t limit_u64 = static_cast<uint64_t>(limit);
    if (est_bytes_u64 > limit_u64) {
        throw std::runtime_error(
            "Export to NumPy exceeds configured materialization limit; pass allow_huge=True via pycauset.to_numpy(...) to override.");
    }
}

inline void ensure_numpy_export_allowed(const MatrixBase& mat);
inline void ensure_numpy_export_allowed(const VectorBase& v);

inline void ensure_numpy_export_allowed_for_buffer(const MatrixBase& mat) {
    try {
        ensure_numpy_export_allowed(mat);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        throw py::error_already_set();
    }
}

inline void ensure_numpy_export_allowed_for_buffer(const VectorBase& v) {
    try {
        ensure_numpy_export_allowed(v);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        throw py::error_already_set();
    }
}

inline void ensure_numpy_export_allowed(const MatrixBase& mat) {
    const std::string backing = mat.get_backing_file();
    const uint64_t rows = mat.rows();
    const uint64_t cols = mat.cols();
    uint64_t elems = 0;
    const bool overflow = mul_overflow_u64(rows, cols, &elems);
    ensure_numpy_export_allowed_impl(backing, elems, mat.get_data_type(), overflow);
}

inline void ensure_numpy_export_allowed(const VectorBase& v) {
    const std::string backing = v.get_backing_file();
    ensure_numpy_export_allowed_impl(backing, v.size(), v.get_data_type(), /*elem_overflow=*/false);
}

} // namespace pycauset::bindings
