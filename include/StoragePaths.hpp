#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace pycauset {

// Global configuration for memory threshold
extern std::atomic<uint64_t> g_memory_threshold;

inline void set_memory_threshold(uint64_t bytes) {
    g_memory_threshold.store(bytes);
}

inline uint64_t get_memory_threshold() {
    return g_memory_threshold.load();
}

inline std::filesystem::path get_storage_root() {
    const char* env = std::getenv("PYCAUSET_STORAGE_DIR");
    std::filesystem::path root;
    if (env && *env != '\0') {
        root = std::filesystem::path(env);
    } else {
        root = std::filesystem::current_path() / ".pycauset";
    }
    std::filesystem::create_directories(root);
    return root;
}

inline std::string make_unique_storage_file(const std::string& prefix) {
    static std::atomic<uint64_t> counter{0};
    auto filename = prefix + "_" + std::to_string(counter++) + ".pycauset";
    return (get_storage_root() / filename).string();
}

}
