#pragma once

#include <atomic>
#include <cstdint>

// Counts calls to the virtual per-element accessors (MatrixBase/VectorBase get/set_element_as_*).
// Off by default (one relaxed load per call). The operation atlas enables it to classify a
// code path as "generic" (one virtual call per element) without anyone tagging the path.
// Counts across all threads, so OpenMP loops are included.
namespace pycauset::access_counter {

extern std::atomic<bool> g_enabled;
extern std::atomic<uint64_t> g_count;

inline void hit() {
    if (g_enabled.load(std::memory_order_relaxed)) {
        g_count.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace pycauset::access_counter
