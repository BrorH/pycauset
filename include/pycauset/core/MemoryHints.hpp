#pragma once

#include <cstddef>
#include <cstdint>

namespace pycauset {
namespace core {

/**
 * @brief Describes the intended memory access pattern for a PersistentObject.
 * 
 * This is used by the "Lookahead Protocol" to inform the IOAccelerator
 * about future access patterns, allowing it to optimize prefetching and caching.
 */
enum class AccessPattern {
    Sequential, // Reading 0..N (Standard)
    Reverse,    // Reading N..0
    Strided,    // Reading column-wise in row-major storage (or vice versa)
    Random,     // Graph traversal / Sparse access
    Once        // Read once and discard (Stream)
};

/**
 * @brief A hint describing a specific memory access intent.
 */
struct MemoryHint {
    AccessPattern pattern;
    size_t start_offset;    // Byte offset where access begins
    size_t length;          // Total bytes to be accessed
    size_t stride_bytes;    // For Strided: bytes to skip between reads (0 if not strided)
    size_t block_bytes;     // For Strided: bytes to read at each step (default: element size)

    // Default constructor
    MemoryHint() 
        : pattern(AccessPattern::Sequential), start_offset(0), length(0), stride_bytes(0), block_bytes(0) {}

    // Helper for Sequential access
    static MemoryHint sequential(size_t start, size_t len) {
        MemoryHint h;
        h.pattern = AccessPattern::Sequential;
        h.start_offset = start;
        h.length = len;
        return h;
    }

    // Helper for Strided access (e.g. reading columns)
    static MemoryHint strided(size_t start, size_t len, size_t stride, size_t block) {
        MemoryHint h;
        h.pattern = AccessPattern::Strided;
        h.start_offset = start;
        h.length = len;
        h.stride_bytes = stride;
        h.block_bytes = block;
        return h;
    }
};

} // namespace core
} // namespace pycauset
