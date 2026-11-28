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

    static thread_local int64_t t_threshold_override = -1;

    int64_t set_thread_memory_threshold(int64_t bytes) {
        const int64_t previous = t_threshold_override;
        t_threshold_override = bytes < 0 ? -1 : bytes;
        return previous;
    }

    uint64_t placement_threshold() {
        return t_threshold_override >= 0 ? static_cast<uint64_t>(t_threshold_override) : g_memory_threshold.load();
    }

    std::filesystem::path get_storage_root() {
        std::lock_guard<std::mutex> lock(g_storage_root_mu);
        if (!g_storage_root_set) {
            g_storage_root = std::filesystem::current_path() / ".pycauset";
            g_storage_root_set = true;
        }
        return g_storage_root;
    }

    void set_storage_root(const std::filesystem::path& root) {
        std::filesystem::path resolved = root;  // created with its first file (review 3.8 C-6)

        std::lock_guard<std::mutex> lock(g_storage_root_mu);
        g_storage_root = std::move(resolved);
        g_storage_root_set = true;
    }

    // <prefix>_<pid>_<counter>.pycauset-tmp: unique across processes sharing a storage
    // directory (two notebooks in one folder overwrote each other's matrices with a per-process
    // counter alone), and across fork (the child has its own pid). The startup / exit sweep
    // deletes only names of this form (a suffix no other program uses: it deleted every *.tmp
    // file, review 3.8 C-1) and reads the pid to leave live processes' files alone
    // (python/pycauset/_storage.py, TEMP_SUFFIX).
    std::string make_unique_storage_file(const std::string& prefix) {
        const uint64_t id = g_storage_counter.fetch_add(1);
#ifdef _WIN32
        const unsigned long pid = static_cast<unsigned long>(_getpid());
#else
        const unsigned long pid = static_cast<unsigned long>(getpid());
#endif
        const auto filename = prefix + "_" + std::to_string(pid) + "_" + std::to_string(id) + ".pycauset-tmp";
        // The directory is created with its first file, not at import or when it is set.
        const auto root = get_storage_root();
        std::filesystem::create_directories(root);
        return (root / filename).string();
    }
}
