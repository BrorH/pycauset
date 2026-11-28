#include "pycauset/core/StorageUtils.hpp"

#include <mutex>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include <utility>

namespace pycauset {
    // Default threshold: 1 GB
    static std::atomic<uint64_t> g_memory_threshold{1024ULL * 1024 * 1024};

    static std::mutex g_storage_root_mu;
    static std::filesystem::path g_storage_root;
    static bool g_storage_root_set = false;
    static std::atomic<uint64_t> g_storage_counter{0};

    void set_memory_threshold(uint64_t bytes) {
        g_memory_threshold.store(bytes);
    }

    uint64_t get_memory_threshold() {
        return g_memory_threshold.load();
    }

    std::filesystem::path get_storage_root() {
        std::lock_guard<std::mutex> lock(g_storage_root_mu);
        if (!g_storage_root_set) {
            g_storage_root = std::filesystem::current_path() / ".pycauset";
            std::filesystem::create_directories(g_storage_root);
            g_storage_root_set = true;
        }
        return g_storage_root;
    }

    void set_storage_root(const std::filesystem::path& root) {
        std::filesystem::path resolved = root;
        std::filesystem::create_directories(resolved);

        std::lock_guard<std::mutex> lock(g_storage_root_mu);
        g_storage_root = std::move(resolved);
        g_storage_root_set = true;
    }

    // <prefix>_<pid>_<counter>.tmp: unique across processes sharing a storage directory (two
    // notebooks in one folder overwrote each other's matrices with a per-process counter alone),
    // and across fork (the child has its own pid). The startup / exit sweep reads the pid to
    // leave live processes' files alone (python/pycauset/_storage.py).
    std::string make_unique_storage_file(const std::string& prefix) {
        const uint64_t id = g_storage_counter.fetch_add(1);
#ifdef _WIN32
        const unsigned long pid = static_cast<unsigned long>(_getpid());
#else
        const unsigned long pid = static_cast<unsigned long>(getpid());
#endif
        const auto filename = prefix + "_" + std::to_string(pid) + "_" + std::to_string(id) + ".tmp";
        return (get_storage_root() / filename).string();
    }
}
