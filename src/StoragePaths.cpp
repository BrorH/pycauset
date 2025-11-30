#include "StoragePaths.hpp"

namespace pycauset {
    // Default threshold: 10 MB
    std::atomic<uint64_t> g_memory_threshold{10 * 1024 * 1024};
}
