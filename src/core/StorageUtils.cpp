#include "pycauset/core/StorageUtils.hpp"

namespace pycauset {
    // Default threshold: 1 GB
    static std::atomic<uint64_t> g_memory_threshold{1024ULL * 1024 * 1024};

    void set_memory_threshold(uint64_t bytes) {
        g_memory_threshold.store(bytes);
    }

    uint64_t get_memory_threshold() {
        return g_memory_threshold.load();
    }
}
