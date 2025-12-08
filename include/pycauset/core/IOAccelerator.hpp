#pragma once

#include <cstdint>
#include <vector>
#include <future>
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/core/MemoryHints.hpp"

namespace pycauset {
namespace core {

/**
 * @brief Handles OS-specific I/O optimization hints and async operations.
 * 
 * This component is responsible for:
 * 1. Prefetching data into RAM before it is needed (hiding latency).
 * 2. Advising the OS to discard data we are done with (preventing thrashing).
 * 3. Managing background flush operations.
 */
class IOAccelerator {
public:
    explicit IOAccelerator(MemoryMapper* mapper);
    ~IOAccelerator();

    /**
     * @brief Hints to the OS that the specified range will be accessed sequentially soon.
     * Triggers asynchronous prefetching from disk to RAM.
     * 
     * @param offset Byte offset relative to the start of the mapped data.
     * @param size Number of bytes to prefetch.
     */
    void prefetch(size_t offset, size_t size);

    /**
     * @brief Hints to the OS that the specified range is no longer needed.
     * The OS may discard these pages from RAM immediately.
     * 
     * @param offset Byte offset relative to the start of the mapped data.
     * @param size Number of bytes to discard.
     */
    void discard(size_t offset, size_t size);

    /**
     * @brief Flushes the specified range to disk asynchronously.
     * Returns a future that completes when the flush is done.
     */
    std::future<void> flush_async(size_t offset, size_t size);

    /**
     * @brief Processes a high-level memory access hint.
     * Translates the hint into specific prefetch/discard commands.
     */
    void process_hint(const MemoryHint& hint);

private:
    MemoryMapper* mapper_;
    
    // Platform-specific implementation for scatter-gather prefetch
    void prefetch_impl(void* addr, size_t len);
    void discard_impl(void* addr, size_t len);
    void prefetch_ranges_impl(const std::vector<std::pair<void*, size_t>>& ranges);
};

} // namespace core
} // namespace pycauset
