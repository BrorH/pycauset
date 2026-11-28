#include <atomic>
#include <chrono>
#include "pycauset/core/MemoryGovernor.hpp"
#include <algorithm>
#include <fstream>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <unistd.h>
#else
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

namespace pycauset {
namespace core {

MemoryGovernor& MemoryGovernor::instance() {
    // Intentionally leaked: objects made or freed during interpreter finalization may still
    // ask it, and the destruction order of a Meyers singleton is undefined then.
    static MemoryGovernor* instance = new MemoryGovernor();
    return *instance;
}

MemoryGovernor::MemoryGovernor() {
    refresh_system_stats();
    // Default safety margin: 10% of total RAM or 2GB, whichever is smaller
    uint64_t ten_percent = cached_total_ram_ / 10;
    uint64_t two_gb = 2ULL * 1024 * 1024 * 1024;
    safety_margin_ = std::min(ten_percent, two_gb);

    // Default pinned memory limit:
    // min(SystemRAM * 0.5, FreeRAM * 0.8, 8GB)
    max_pinned_memory_ = compute_dynamic_pinning_budget();
}

MemoryGovernor::~MemoryGovernor() {
}

void MemoryGovernor::refresh_system_stats() const {
    // At most every 20 ms: reading /proc/meminfo (or the OS equivalent) on every allocation
    // and every kernel's direct-path check cost ~6 us each, most of a small operation
    // (review N.6, F-096). Storage placement is a heuristic, never a result.
    {
        using clock = std::chrono::steady_clock;
        static std::atomic<int64_t> last_ns{0};
        const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count();
        const int64_t prev = last_ns.load(std::memory_order_relaxed);
        if (prev != 0 && now - prev < 20'000'000 && cached_total_ram_ != 0) return;
        last_ns.store(now, std::memory_order_relaxed);
    }
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
#elif defined(__APPLE__)
    int mib[2];
    int64_t physical_memory;
    size_t length;

    // Get Total RAM
    mib[0] = CTL_HW;
    mib[1] = HW_MEMSIZE;
    length = sizeof(int64_t);
    sysctl(mib, 2, &physical_memory, &length, NULL, 0);
    cached_total_ram_ = physical_memory;

    // Get Free/Available RAM. macOS's "free" page count is strictly free pages,
    // which excludes the reclaimable file cache and reads artificially low right
    // after a build or a big mmap write, so a tiny object can be wrongly routed
    // to disk. Use free + inactive pages (inactive pages are clean and reclaimable
    // without swapping), mirroring the Linux MemAvailable fix below.
    mach_msg_type_number_t count = HOST_VM_INFO_COUNT;
    vm_statistics_data_t vm_stat;
    if (host_statistics(mach_host_self(), HOST_VM_INFO, (host_info_t)&vm_stat, &count) == KERN_SUCCESS) {
        uint64_t available_pages = (uint64_t)vm_stat.free_count + (uint64_t)vm_stat.inactive_count;
        cached_available_ram_ = available_pages * sysconf(_SC_PAGESIZE);
    } else {
         cached_available_ram_ = 1024 * 1024 * 1024; // Assume 1GB
    }
#else
    // Linux: sysinfo().freeram reports MemFree, which excludes reclaimable page
    // cache and reads artificially low right after a build (most RAM sits in
    // cache), causing can_fit_in_ram()/request_ram() to wrongly route small and
    // bit-matrix operations to disk/streaming. Use MemAvailable from
    // /proc/meminfo (the kernel's "memory available without swapping" estimate)
    // instead. Fall back to sysinfo() if /proc/meminfo is unavailable.
    uint64_t mem_available = 0;
    uint64_t mem_total = 0;
    {
        std::ifstream meminfo("/proc/meminfo");
        std::string key;
        uint64_t kb = 0;
        while (meminfo >> key) {
            if (key == "MemAvailable:") {
                if (meminfo >> kb) mem_available = kb * 1024;
            } else if (key == "MemTotal:") {
                if (meminfo >> kb) mem_total = kb * 1024;
            }
        }
    }
    if (mem_available != 0) {
        cached_total_ram_ = (mem_total != 0) ? mem_total : cached_total_ram_.load();
        cached_available_ram_ = mem_available;
    } else {
        struct sysinfo info;
        if (sysinfo(&info) == 0) {
            cached_total_ram_ = (uint64_t)info.totalram * info.mem_unit;
            cached_available_ram_ = (uint64_t)info.freeram * info.mem_unit;
        } else {
            cached_available_ram_ = 1024 * 1024 * 1024; // Assume 1GB free
        }
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

uint64_t MemoryGovernor::get_safety_margin() const {
    return safety_margin_;
}

void MemoryGovernor::set_safety_margin(uint64_t bytes) {
    safety_margin_ = bytes;
}

void MemoryGovernor::reset_for_testing() {
    pinned_memory_usage_ = 0;
    pinning_budget_override_ = false;
    // Not the safety margin or the cached stats: they are system dependent.
}

bool MemoryGovernor::can_fit_in_ram(size_t size_bytes) const {
    refresh_system_stats();
    uint64_t available = cached_available_ram_;
    uint64_t margin = safety_margin_;
    
    // Check if we have enough space including the safety margin
    return available > (size_bytes + margin);
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

void MemoryGovernor::set_pinning_budget_override(uint64_t bytes) {
    max_pinned_memory_ = bytes;
    pinning_budget_override_ = true;
}

void MemoryGovernor::clear_pinning_budget_override() {
    pinning_budget_override_ = false;
}

bool MemoryGovernor::has_pinning_budget_override() const {
    return pinning_budget_override_.load();
}

uint64_t MemoryGovernor::compute_dynamic_pinning_budget() const {
    refresh_system_stats();

    uint64_t total = cached_total_ram_.load();
    uint64_t available = cached_available_ram_.load();
    if (total == 0 || available == 0) return 0;

    uint64_t half_total = total / 2;
    uint64_t eighty_percent_free = (available * 8) / 10;
    uint64_t eight_gb = 8ULL * 1024 * 1024 * 1024;

    return std::min({half_total, eighty_percent_free, eight_gb});
}

void MemoryGovernor::apply_dynamic_pinning_budget() {
    if (pinning_budget_override_.load()) return;
    max_pinned_memory_ = compute_dynamic_pinning_budget();
}

} // namespace core
} // namespace pycauset
