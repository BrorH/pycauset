#include "pycauset/core/StorageUtils.hpp"

namespace pycauset {
    // Default threshold: 1 GB
    std::atomic<uint64_t> g_memory_threshold{1024ULL * 1024 * 1024};
}
