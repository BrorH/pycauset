#pragma once

#include <cstdint>
#include <atomic>

namespace pycauset {
namespace core {

/**
 * @brief The RAM budget: whether a new object fits in RAM (available RAM minus a safety
 * margin), and the pinned-memory budget of the (dormant) GPU path.
 *
 * A new object is placed in RAM when it is under the memory threshold and fits, otherwise
 * in a memory-mapped file. Objects are never moved to disk later: kernels hold raw pointers
 * into their operands' storage (F-120). No registry: the OS's available-RAM figure already
 * counts what PyCauset holds.
 */
class MemoryGovernor {
public:
    static MemoryGovernor& instance();
    ~MemoryGovernor();

    // Delete copy/move
    MemoryGovernor(const MemoryGovernor&) = delete;
    MemoryGovernor& operator=(const MemoryGovernor&) = delete;

    /**
     * @brief Whether size_bytes fits in RAM: available RAM (from the OS, which already counts
     * PyCauset's own allocations) minus the safety margin. Placement of a new object and the
     * direct-path checks of kernels ask this; nothing is reserved and nothing is moved.
     */
    bool can_fit_in_ram(size_t size_bytes) const;

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

    /**
     * @brief Sets a manual pinning budget override (disables dynamic budget updates).
     */
    void set_pinning_budget_override(uint64_t bytes);

    /**
     * @brief Clears the manual pinning budget override (re-enables dynamic budget updates).
     */
    void clear_pinning_budget_override();

    /**
     * @brief Returns true if a manual pinning budget override is active.
     */
    bool has_pinning_budget_override() const;

    /**
     * @brief Computes the dynamic pinning budget (the GPU pinning heuristic).
     */
    uint64_t compute_dynamic_pinning_budget() const;

    /**
     * @brief Applies the dynamic pinning budget unless overridden.
     */
    void apply_dynamic_pinning_budget();

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


    std::atomic<uint64_t> safety_margin_{0}; // Default will be set in constructor

    // Pinned Memory Stats
    std::atomic<uint64_t> pinned_memory_usage_{0};
    std::atomic<uint64_t> max_pinned_memory_{0};
    std::atomic<bool> pinning_budget_override_{false};

    // Cached system stats
    mutable std::atomic<uint64_t> cached_total_ram_{0};
    mutable std::atomic<uint64_t> cached_available_ram_{0};
};

} // namespace core
} // namespace pycauset
