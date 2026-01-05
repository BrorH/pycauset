#include "pycauset/core/IOAccelerator.hpp"
#include "pycauset/core/DebugTrace.hpp"
#include <iostream>
#include <algorithm>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <memoryapi.h>
// Link with kernel32.lib
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace pycauset {
namespace core {

IOAccelerator::IOAccelerator(MemoryMapper* mapper) : mapper_(mapper) {}

IOAccelerator::~IOAccelerator() {}

void IOAccelerator::prefetch(size_t offset, size_t size) {
    pycauset::debug_trace::set_last_io("io.prefetch");
    if (!mapper_) return;
    
    uint8_t* base = static_cast<uint8_t*>(mapper_->get_data());
    if (!base) return;

    // Clamp to valid range
    size_t max_size = mapper_->get_data_size();
    if (offset >= max_size) return;
    size_t actual_size = std::min(size, max_size - offset);

    prefetch_impl(base + offset, actual_size);
}

void IOAccelerator::discard(size_t offset, size_t size) {
    pycauset::debug_trace::set_last_io("io.discard");
    if (!mapper_) return;

    uint8_t* base = static_cast<uint8_t*>(mapper_->get_data());
    if (!base) return;

    size_t max_size = mapper_->get_data_size();
    if (offset >= max_size) return;
    size_t actual_size = std::min(size, max_size - offset);

    discard_impl(base + offset, actual_size);
}

std::future<void> IOAccelerator::flush_async(size_t offset, size_t size) {
    return std::async(std::launch::async, [this, offset, size]() {
        if (mapper_) {
            uint8_t* base = static_cast<uint8_t*>(mapper_->get_data());
            if (base) {
                size_t max_size = mapper_->get_data_size();
                if (offset < max_size) {
                    size_t actual_size = std::min(size, max_size - offset);
                    mapper_->flush(base + offset, actual_size);
                }
            }
        }
    });
}

void IOAccelerator::process_hint(const MemoryHint& hint) {
    if (!mapper_) return;
    uint8_t* base = static_cast<uint8_t*>(mapper_->get_data());
    if (!base) return;
    size_t max_size = mapper_->get_data_size();

    if (hint.pattern == AccessPattern::Sequential) {
        prefetch(hint.start_offset, hint.length);
    }
    else if (hint.pattern == AccessPattern::Strided) {
        if (hint.block_bytes == 0) return;
        
        // Calculate number of blocks to fetch
        // We interpret hint.length as the total bytes of interest (sum of blocks)
        // or the span? The helper in MemoryHints.hpp sets length=len.
        // Let's assume length is the total bytes to read (count * block).
        // If the user passed span, they should have adjusted.
        // Actually, let's be robust:
        // If stride > block, we have gaps.
        // We iterate until we cover 'length' bytes of payload or hit max_size.
        
        size_t num_blocks = hint.length / hint.block_bytes;
        if (num_blocks == 0 && hint.length > 0) num_blocks = 1;

        std::vector<std::pair<void*, size_t>> ranges;
        const size_t BATCH_SIZE = 256; // Windows limit is usually high, but let's be safe
        ranges.reserve(BATCH_SIZE);
        
        for (size_t i = 0; i < num_blocks; ++i) {
            size_t offset = hint.start_offset + i * hint.stride_bytes;
            if (offset >= max_size) break;
            
            size_t len = std::min(hint.block_bytes, max_size - offset);
            ranges.push_back({base + offset, len});
            
            if (ranges.size() >= BATCH_SIZE) {
                prefetch_ranges_impl(ranges);
                ranges.clear();
            }
        }
        if (!ranges.empty()) {
            prefetch_ranges_impl(ranges);
        }
    }
}

// --- Platform Specific Implementations ---

#ifdef _WIN32

// Define VM_OFFER_PRIORITY if missing (e.g. older SDKs)
#ifndef VM_OFFER_PRIORITY
typedef enum _VM_OFFER_PRIORITY {
    VmOfferPriorityVeryLow = 1,
    VmOfferPriorityLow,
    VmOfferPriorityBelowNormal,
    VmOfferPriorityNormal
} VM_OFFER_PRIORITY;
#endif

// Function pointer type for PrefetchVirtualMemory (in case we are on old Windows, though unlikely for C++20)
typedef BOOL (WINAPI *PPrefetchVirtualMemory)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
typedef DWORD (WINAPI *POfferVirtualMemory)(PVOID, SIZE_T, VM_OFFER_PRIORITY);

void IOAccelerator::prefetch_impl(void* addr, size_t len) {
    static PPrefetchVirtualMemory pPrefetch = (PPrefetchVirtualMemory)GetProcAddress(GetModuleHandleA("kernel32.dll"), "PrefetchVirtualMemory");
    
    if (pPrefetch) {
        WIN32_MEMORY_RANGE_ENTRY entry;
        entry.VirtualAddress = addr;
        entry.NumberOfBytes = len;
        
        if (!pPrefetch(GetCurrentProcess(), 1, &entry, 0)) {
            // Failed
        }
    }
}

void IOAccelerator::prefetch_ranges_impl(const std::vector<std::pair<void*, size_t>>& ranges) {
    if (ranges.empty()) return;
    
    static PPrefetchVirtualMemory pPrefetch = (PPrefetchVirtualMemory)GetProcAddress(GetModuleHandleA("kernel32.dll"), "PrefetchVirtualMemory");
    
    if (pPrefetch) {
        std::vector<WIN32_MEMORY_RANGE_ENTRY> entries;
        entries.reserve(ranges.size());
        for (const auto& r : ranges) {
            WIN32_MEMORY_RANGE_ENTRY e;
            e.VirtualAddress = r.first;
            e.NumberOfBytes = r.second;
            entries.push_back(e);
        }
        
        pPrefetch(GetCurrentProcess(), static_cast<ULONG_PTR>(entries.size()), entries.data(), 0);
    }
}

void IOAccelerator::discard_impl(void* addr, size_t len) {
    // R1_SAFETY: Windows Discard Implementation
    // 1. Flush changes to disk so we don't lose data
    FlushViewOfFile(addr, len);

    // 2. Offer memory to OS for reclamation
    // VmOfferPriorityVeryLow (1) allows the OS to discard the pages immediately if needed.
    // Note: If we access this memory again, we must Reclaim it, or we get garbage/fault.
    // But discard() implies we are done. If we reload from file later, the OS handles the page fault
    // by reading from disk (since we flushed).
    // Wait, OfferVirtualMemory makes pages "purgeable". If purged, they are zeroed or garbage?
    // MSDN: "If the content is discarded, the memory is in the reset state."
    // Reset state = Zeros? Or undefined?
    // If we map the file again, do we see the file content?
    // OfferVirtualMemory operates on the *virtual address*.
    // If we unmap and remap, we get a new view.
    // So this is safe for "I am done with this view".
    
    static POfferVirtualMemory pOffer = (POfferVirtualMemory)GetProcAddress(GetModuleHandleA("kernel32.dll"), "OfferVirtualMemory");
    
    if (pOffer) {
        // 1 = VmOfferPriorityVeryLow
        pOffer(addr, len, (VM_OFFER_PRIORITY)1);
    } else {
        // Fallback: VirtualUnlock is mostly a no-op but better than nothing
        VirtualUnlock(addr, len);
    }
}

#else

void IOAccelerator::prefetch_impl(void* addr, size_t len) {
    // Linux: madvise with MADV_WILLNEED
    madvise(addr, len, MADV_WILLNEED);
    
    // Huge Pages Investigation (Phase 2)
    // If the region is large enough (> 2MB), advise the kernel to use huge pages.
    // This reduces TLB pressure for large matrices.
    if (len >= 2 * 1024 * 1024) {
#ifdef MADV_HUGEPAGE
        madvise(addr, len, MADV_HUGEPAGE);
#endif
    }
}

void IOAccelerator::prefetch_ranges_impl(const std::vector<std::pair<void*, size_t>>& ranges) {
    for (const auto& r : ranges) {
        madvise(r.first, r.second, MADV_WILLNEED);
    }
}

void IOAccelerator::discard_impl(void* addr, size_t len) {
    // Linux: madvise with MADV_DONTNEED
    // This tells the kernel we are done with these pages and they can be reclaimed immediately.
    madvise(addr, len, MADV_DONTNEED);
}

#endif

} // namespace core
} // namespace pycauset
