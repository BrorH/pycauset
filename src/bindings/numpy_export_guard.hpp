#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <string>
#include <stdexcept>

#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "numpy_dtypes.hpp"
#include "pycauset/core/MemoryGovernor.hpp"
#include "pycauset/core/Types.hpp"
#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/vector/VectorBase.hpp"

namespace pycauset::bindings {

// Global ceiling in bytes. -1 disables the size check.
// Note: this is a process-wide setting (matches the Python-side guard behavior).
inline std::atomic<int64_t> g_numpy_export_max_bytes{-1};

// What an export materialises (NumPy's element size, not the packed storage's), from the one
// dtype table (numpy_dtypes.hpp).
inline uint64_t numpy_output_itemsize_bytes(DataType dt) {
    const auto* e = pycauset_bindings::numpy_dtype_of(dt);
    return e ? static_cast<uint64_t>(e->itemsize) : 8;
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

// The guard is by size only (F-016, RESTRUCTURE 3.1 item 7): whether the object is in RAM or
// in a file does not matter (spilling is automatic, so the user cannot predict it). With a
// ceiling set (set_export_max_bytes), an export above it raises; without one, an export that
// would not fit in free RAM raises. allow_huge=True skips the check.
inline void ensure_numpy_export_allowed_impl(uint64_t elem_count, DataType dt, bool elem_overflow) {
    const char* hint = "; pass allow_huge=True via pycauset.to_numpy(...) to override.";
    uint64_t bytes = 0;
    if (elem_overflow || mul_overflow_u64(elem_count, numpy_output_itemsize_bytes(dt), &bytes)) {
        throw std::runtime_error(std::string("Export to NumPy is too large") + hint);
    }
    const int64_t limit = g_numpy_export_max_bytes.load();
    if (limit >= 0) {
        if (bytes > static_cast<uint64_t>(limit)) {
            throw std::runtime_error("Export to NumPy (" + std::to_string(bytes) +
                                     " bytes) exceeds the set_export_max_bytes ceiling" + hint);
        }
        return;
    }
    if (!pycauset::core::MemoryGovernor::instance().can_fit_in_ram(bytes)) {
        throw std::runtime_error("Export to NumPy (" + std::to_string(bytes) +
                                 " bytes) would not fit in free RAM" + hint);
    }
}

inline void ensure_numpy_export_allowed(const MatrixBase& mat);
inline void ensure_numpy_export_allowed(const VectorBase& v);

inline void ensure_numpy_export_allowed(const MatrixBase& mat) {
    const uint64_t rows = mat.rows();
    const uint64_t cols = mat.cols();
    uint64_t elems = 0;
    const bool overflow = mul_overflow_u64(rows, cols, &elems);
    ensure_numpy_export_allowed_impl(elems, mat.get_data_type(), overflow);
}

inline void ensure_numpy_export_allowed(const VectorBase& v) {
    ensure_numpy_export_allowed_impl(v.size(), v.get_data_type(), /*elem_overflow=*/false);
}

} // namespace pycauset::bindings
