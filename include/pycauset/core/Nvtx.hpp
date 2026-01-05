#pragma once

#ifdef _WIN32
#include <windows.h>
#endif

// NVTX is a header-only library, but we need to ensure the headers are available.
// If not available, we define dummy macros.
// For this implementation, we assume the user might not have the CUDA Toolkit installed everywhere,
// so we use a weak linking approach or just stub it out if NVTX_ENABLED is not defined.

#ifdef NVTX_ENABLED
#include <nvtx3/nvToolsExt.h>

namespace pycauset {
namespace profiling {

class ScopedRange {
public:
    ScopedRange(const char* message) {
        nvtxRangePushA(message);
    }
    ~ScopedRange() {
        nvtxRangePop();
    }
};

inline void mark(const char* message) {
    nvtxMarkA(message);
}

} // namespace profiling
} // namespace pycauset

#define NVTX_RANGE(name) pycauset::profiling::ScopedRange nvtx_range_##__LINE__(name)
#define NVTX_MARK(name) pycauset::profiling::mark(name)

#else

#define NVTX_RANGE(name)
#define NVTX_MARK(name)

#endif
