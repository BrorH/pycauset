#include "pycauset/core/MemoryGovernor.hpp"
#include "pycauset/core/PersistentObject.hpp" // We will need this later for eviction
#include <iostream>
#include <algorithm>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

namespace pycauset {
namespace core {

MemoryGovernor& MemoryGovernor::instance() {
    static MemoryGovernor instance;
    return instance;
}

MemoryGovernor::MemoryGovernor() {
    refresh_system_stats();
    // Default safety margin: 10% of total RAM or 2GB, whichever is smaller
    uint64_t ten_percent = cached_total_ram_ / 10;
    uint64_t two_gb = 2ULL * 1024 * 1024 * 1024;
    safety_margin_ = std::min(ten_percent, two_gb);

    // Default pinned memory limit: 20% of total RAM or 4GB, whichever is smaller
    // We are conservative here because pinning too much RAM can crash the OS.
    uint64_t twenty_percent = cached_total_ram_ / 5;
    uint64_t four_gb = 4ULL * 1024 * 1024 * 1024;
    max_pinned_memory_ = std::min(twenty_percent, four_gb);
}

MemoryGovernor::~MemoryGovernor() {
}

void MemoryGovernor::refresh_system_stats() const {
#ifdef _WIN32
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    if (GlobalMemoryStatusEx(&statex)) {
        cached_total_ram_ = statex.ullTotalPhys;
        cached_available_ram_ = statex.ullAvailPhys;
    } else {
        // Fallback if API fails (unlikely)
        cached_available_ram_ = 1024 * 1024 * 1024; // Assume 1GB free
    }
#else
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        cached_total_ram_ = (uint64_t)info.totalram * info.mem_unit;
        cached_available_ram_ = (uint64_t)info.freeram * info.mem_unit;
    } else {
        cached_available_ram_ = 1024 * 1024 * 1024; // Assume 1GB free
    }
#endif
}

uint64_t MemoryGovernor::get_total_system_ram() const {
    // Total RAM doesn't change, but we refresh to be safe/consistent
    if (cached_total_ram_ == 0) refresh_system_stats();
    return cached_total_ram_;
}

uint64_t MemoryGovernor::get_available_system_ram() const {
    refresh_system_stats();
    return cached_available_ram_;
}

uint64_t MemoryGovernor::get_tracked_ram_usage() const {
    return tracked_ram_usage_;
}

uint64_t MemoryGovernor::get_safety_margin() const {
    return safety_margin_;
}

void MemoryGovernor::set_safety_margin(uint64_t bytes) {
    safety_margin_ = bytes;
}

void MemoryGovernor::reset_for_testing() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    lru_list_.clear();
    lru_map_.clear();
    object_sizes_.clear();
    tracked_ram_usage_ = 0;
    pinned_memory_usage_ = 0;
    // Don't reset safety margin or cached stats as they are system dependent
}

bool MemoryGovernor::can_fit_in_ram(size_t size_bytes) const {
    refresh_system_stats();
    uint64_t available = cached_available_ram_;
    uint64_t margin = safety_margin_;
    
    // Check if we have enough space including the safety margin
    return available > (size_bytes + margin);
}

bool MemoryGovernor::should_use_direct_path(size_t total_operation_bytes) const {
    // 1. If it fits in RAM (with safety margin), use Direct Path.
    // This is the "Anti-Nanny" rule: Don't force streaming just because we can't pin.
    // OS paging is efficient enough for RAM-resident workloads.
    if (can_fit_in_ram(total_operation_bytes)) {
        return true;
    }

    // 2. If it doesn't fit in RAM, we MUST use Streaming Path.
    return false;
}

bool MemoryGovernor::try_pin_memory(size_t size_bytes) {
    // We use a CAS loop or just atomic operations since we only care about the total
    uint64_t current = pinned_memory_usage_.load();
    while (true) {
        if (current + size_bytes > max_pinned_memory_) {
            return false;
        }
        if (pinned_memory_usage_.compare_exchange_weak(current, current + size_bytes)) {
            return true;
        }
        // If CAS failed, 'current' is updated to the new value, loop again
    }
}

void MemoryGovernor::unpin_memory(size_t size_bytes) {
    pinned_memory_usage_ -= size_bytes;
    // Sanity check (underflow protection)
    if (pinned_memory_usage_ > max_pinned_memory_ * 2) { // Wrapped around
        pinned_memory_usage_ = 0; 
    }
}

uint64_t MemoryGovernor::get_pinned_memory_usage() const {
    return pinned_memory_usage_;
}

uint64_t MemoryGovernor::get_max_pinned_memory() const {
    return max_pinned_memory_;
}

void MemoryGovernor::set_max_pinned_memory(uint64_t bytes) {
    max_pinned_memory_ = bytes;
}

void MemoryGovernor::register_object(PersistentObject* obj, size_t size_bytes) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    
    if (lru_map_.find(obj) != lru_map_.end()) {
        // Already registered, just update size
        tracked_ram_usage_ -= object_sizes_[obj];
        object_sizes_[obj] = size_bytes;
        tracked_ram_usage_ += size_bytes;
        return;
    }

    // Add to LRU (Back is Most Recent)
    lru_list_.push_back(obj);
    lru_map_[obj] = --lru_list_.end();
    object_sizes_[obj] = size_bytes;
    tracked_ram_usage_ += size_bytes;
}

void MemoryGovernor::unregister_object(PersistentObject* obj) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    
    auto it = lru_map_.find(obj);
    if (it != lru_map_.end()) {
        lru_list_.erase(it->second);
        lru_map_.erase(it);
        tracked_ram_usage_ -= object_sizes_[obj];
        object_sizes_.erase(obj);
    }
}

void MemoryGovernor::touch(PersistentObject* obj) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    
    auto it = lru_map_.find(obj);
    if (it != lru_map_.end()) {
        // Move to back (Most Recent)
        lru_list_.splice(lru_list_.end(), lru_list_, it->second);
    }
}

void MemoryGovernor::update_size(PersistentObject* obj, size_t new_size_bytes) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    
    if (object_sizes_.find(obj) != object_sizes_.end()) {
        tracked_ram_usage_ -= object_sizes_[obj];
        object_sizes_[obj] = new_size_bytes;
        tracked_ram_usage_ += new_size_bytes;
    }
}

bool MemoryGovernor::request_ram(size_t size_bytes) {
    // 1. Check if we have enough RAM right now
    refresh_system_stats();
    uint64_t available = cached_available_ram_;
    uint64_t margin = safety_margin_;

    // If we have plenty of space (available > requested + margin)
    if (available > (size_bytes + margin)) {
        return true;
    }

    // 2. Not enough space. Try to evict.
    return evict_until_fits(size_bytes);
}

bool MemoryGovernor::evict_until_fits(size_t needed_bytes) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    
    // We need to free up enough bytes so that:
    // (Available + Freed) > (Needed + Margin)
    // => Freed > (Needed + Margin) - Available
    
    refresh_system_stats();
    uint64_t available = cached_available_ram_;
    uint64_t margin = safety_margin_;
    
    if (available > (needed_bytes + margin)) return true; // Race condition check

    uint64_t target_to_free = (needed_bytes + margin) - available;
    uint64_t freed_so_far = 0;

    // Iterate from Front (Least Recent)
    auto it = lru_list_.begin();
    while (it != lru_list_.end() && freed_so_far < target_to_free) {
        PersistentObject* victim = *it;
        size_t victim_size = object_sizes_[victim];
        
        // Save next iterator because 'it' might be invalidated by spill_to_disk -> unregister_object
        auto next_it = std::next(it);

        // Attempt to spill to disk
        if (victim->spill_to_disk()) {
            freed_so_far += victim_size;
            
            // Since victim was removed from lru_list_, 'it' is invalid.
            // 'next_it' is still valid (unless victim was the last element, then next_it is end()).
            // However, since we modified the list, let's just restart from begin() to be safe and simple.
            // The next LRU object is now at begin().
            it = lru_list_.begin();
        } else {
            // Failed to spill (maybe pinned or error), skip it
            it = next_it;
        }
    }
    
    if (freed_so_far >= target_to_free) {
        return true;
    }

    return false;
}

} // namespace core
} // namespace pycauset
