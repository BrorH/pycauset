#pragma once

#include <any>
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

namespace pycauset {
// While one of these is alive on a thread, new RAM (:memory:) storage is not zeroed: for a
// result the caller then writes completely (an element-wise or scalar operation's dense
// result). Zeroing a reused 8-16 MB block cost as much as the arithmetic (F-096). Scope it to
// the one construction: `{ UninitializedStorage u; out = std::make_unique<...>(...); }`.
struct UninitializedStorage {
    UninitializedStorage();
    ~UninitializedStorage();
    UninitializedStorage(const UninitializedStorage&) = delete;
    UninitializedStorage& operator=(const UninitializedStorage&) = delete;
    static bool active();
};
}  // namespace pycauset

class MemoryMapper {
public:
    // size is the size of the DATA section.
    // private_writes: map an existing file copy-on-write (MAP_PRIVATE / FILE_MAP_COPY). Reads
    // come from the file; a written page becomes private memory; the file never changes. Used
    // for saved snapshots (pc.load, F-093). A private mapping cannot be remapped.
    MemoryMapper(const std::string& filename, size_t data_size, size_t offset = 0, bool create_new = false,
                 bool private_writes = false);
    ~MemoryMapper();

    void* get_data();
    const void* get_data() const;
    
    size_t get_data_size() const;
    size_t get_offset() const { return offset_; }
    const std::string& get_filename() const { return filename_; }
    bool is_private() const { return private_; }

    void flush();
    void flush(void* ptr, size_t size);

    // Advanced mapping control for large files
    void unmap();
    void map_all();
    void* map_region(size_t offset, size_t size);
    void unmap_region(void* ptr);
    static size_t get_granularity();

    // Pinning support (VirtualLock/mlock)
    bool pin_region(void* ptr, size_t size) const;
    void unpin_region(void* ptr, size_t size) const;

    // Storage version and derived-value cache (review/features/DERIVED_CACHE.md). The storage
    // is the owner: every view of the same payload shares it. `bump_version` runs before any
    // byte can change (PersistentObject::ensure_unique, the write gate of copy-on-write) and
    // drops every cached value. An entry is stored only if the version did not move while it
    // was computed (`cache_put` with the version read before the computation).
    uint64_t version() const { return version_.load(std::memory_order_acquire); }
    void bump_version() {
        version_.fetch_add(1, std::memory_order_acq_rel);
        if (has_cache_.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            cache_.clear();
            has_cache_.store(false, std::memory_order_release);
        }
    }
    bool cache_get(const std::string& key, std::any& out) const {
        if (!has_cache_.load(std::memory_order_acquire)) return false;
        std::lock_guard<std::mutex> lock(cache_mutex_);
        const auto it = cache_.find(key);
        if (it == cache_.end()) return false;
        out = it->second;
        return true;
    }
    // Every current entry (for saving them with the file).
    std::map<std::string, std::any> cache_snapshot() const {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        return cache_;
    }
    void cache_put(const std::string& key, uint64_t computed_at, std::any value) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (version_.load(std::memory_order_acquire) != computed_at) return;  // changed meanwhile
        cache_[key] = std::move(value);
        has_cache_.store(true, std::memory_order_release);
    }

private:
    std::atomic<uint64_t> version_{0};
    std::atomic<bool> has_cache_{false};
    mutable std::mutex cache_mutex_;
    std::map<std::string, std::any> cache_;

    struct FileHeader {
        char magic[8];      // "PYCAUSET"
        uint32_t version;   // 1
        uint8_t reserved[52];
    };
    static_assert(sizeof(FileHeader) == 64, "FileHeader must be 64 bytes");

    std::string filename_;
    size_t data_size_;
    size_t offset_;
    void* mapped_ptr_; 
    void* base_ptr_; // The actual start of the mapping (aligned)
    
#ifdef _WIN32
    void* hFile_;
    void* hMapping_;
#else
    int fd_;
#endif
    bool is_pinned_ = false;
    bool private_ = false;
    bool heap_ = false;  // :memory: storage from the C heap (no mapping)

    void open_file(bool create_new);
    void close_file();
};
