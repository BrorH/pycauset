#pragma once

#include <cstdint>
#include <list>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <memory>
#include <string>

// Forward declaration
class PersistentObject;

namespace pycauset {
namespace core {

/**
 * @brief The MemoryGovernor manages the global RAM budget for the application.
 * 
 * It implements a "Tiered Storage" architecture where objects are kept in RAM
 * as long as possible, and evicted to disk only when memory pressure is high.
 * 
 * It uses an LRU (Least Recently Used) policy to decide which objects to evict.
 */
class MemoryGovernor {
public:
    static MemoryGovernor& instance();
    ~MemoryGovernor();

    // Delete copy/move
    MemoryGovernor(const MemoryGovernor&) = delete;
    MemoryGovernor& operator=(const MemoryGovernor&) = delete;

    /**
     * @brief Registers a new object with the governor.
     * @param obj The object to track.
     * @param size_bytes The size of the object in RAM.
     */
    void register_object(PersistentObject* obj, size_t size_bytes);

    /**
     * @brief Unregisters an object (e.g., on destruction).
     * @param obj The object to stop tracking.
     */
    void unregister_object(PersistentObject* obj);

    /**
     * @brief Notifies the governor that an object has been accessed.
     * This updates its position in the priority queue (LRU).
     * @param obj The object that was accessed.
     */
    void touch(PersistentObject* obj);

    /**
     * @brief Requests permission to allocate memory in RAM.
     * 
     * If sufficient RAM is available, it returns true.
     * If RAM is full, it attempts to evict lower-priority objects to make space.
     * If eviction fails or is insufficient, it returns false (caller must use disk).
     * 
     * @param size_bytes The amount of memory requested.
     * @return true if allocation is allowed in RAM, false otherwise.
     */
    bool request_ram(size_t size_bytes);

    /**
     * @brief Explicitly updates the size of a tracked object (e.g. resize).
     */
    void update_size(PersistentObject* obj, size_t new_size_bytes);

    /**
     * @brief Checks if the requested amount of memory fits in RAM without eviction.
     * This is a lightweight check for optimization purposes.
     * 
     * @param size_bytes The amount of memory to check.
     * @return true if available RAM > size_bytes + safety_margin.
     */
    bool can_fit_in_ram(size_t size_bytes) const;

    /**
     * @brief Determines if an operation should use the "Direct Path" (RAM-resident)
     * or the "Streaming Path" (Out-of-Core).
     * 
     * This encapsulates the "Anti-Nanny" logic:
     * 1. If data fits in RAM, prefer Direct Path (even if pinning fails).
     * 2. If data is huge, force Streaming Path.
     * 
     * @param total_operation_bytes Total memory footprint of the operation (Inputs + Output).
     * @return true if the operation should be performed in-memory (Direct Path).
     */
    bool should_use_direct_path(size_t total_operation_bytes) const;

    // --- Pinned Memory Management (Phase 4) ---

    /**
     * @brief Attempts to reserve a portion of the "Pinned Memory Budget".
     * Pinned memory cannot be swapped out by the OS.
     * 
     * @param size_bytes Amount to pin.
     * @return true if budget allows, false otherwise.
     */
    bool try_pin_memory(size_t size_bytes);

    /**
     * @brief Releases pinned memory budget.
     */
    void unpin_memory(size_t size_bytes);

    /**
     * @brief Returns the current amount of pinned memory.
     */
    uint64_t get_pinned_memory_usage() const;

    /**
     * @brief Returns the maximum allowed pinned memory.
     */
    uint64_t get_max_pinned_memory() const;

    /**
     * @brief Sets the maximum allowed pinned memory.
     */
    void set_max_pinned_memory(uint64_t bytes);

    // --- Statistics & Diagnostics ---

    /**
     * @brief Returns the total physical RAM installed on the system.
     */
    uint64_t get_total_system_ram() const;

    /**
     * @brief Returns the currently available physical RAM on the system.
     * This is a dynamic value polled from the OS.
     */
    uint64_t get_available_system_ram() const;

    /**
     * @brief Returns the amount of RAM currently used by tracked objects.
     */
    uint64_t get_tracked_ram_usage() const;

    /**
     * @brief Returns the configured safety margin (bytes to leave free for OS).
     */
    uint64_t get_safety_margin() const;

    /**
     * @brief Sets the safety margin.
     * @param bytes Bytes to reserve for the OS/other apps.
     */
    void set_safety_margin(uint64_t bytes);

    /**
     * @brief Resets the governor state. FOR TESTING ONLY.
     */
    void reset_for_testing();

private:
    MemoryGovernor();
    // ~MemoryGovernor() = default;

    // Internal helper to poll OS memory
    void refresh_system_stats() const;

    // Internal helper to evict objects until `needed_bytes` are free
    bool evict_until_fits(size_t needed_bytes);

    // State
    mutable std::recursive_mutex mutex_;
    
    // LRU Cache Structure
    // List contains objects ordered by usage (Back = Most Recent, Front = Least Recent)
    std::list<PersistentObject*> lru_list_;
    // Map for O(1) access to list iterators
    std::unordered_map<PersistentObject*, std::list<PersistentObject*>::iterator> lru_map_;
    // Map to track object sizes
    std::unordered_map<PersistentObject*, size_t> object_sizes_;

    // Stats
    std::atomic<uint64_t> tracked_ram_usage_{0};
    std::atomic<uint64_t> safety_margin_{0}; // Default will be set in constructor

    // Pinned Memory Stats
    std::atomic<uint64_t> pinned_memory_usage_{0};
    std::atomic<uint64_t> max_pinned_memory_{0};

    // Cached system stats
    mutable std::atomic<uint64_t> cached_total_ram_{0};
    mutable std::atomic<uint64_t> cached_available_ram_{0};
};

} // namespace core
} // namespace pycauset
