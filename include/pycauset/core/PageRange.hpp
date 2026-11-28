#pragma once
// Page-aligned ranges for msync / madvise (POSIX needs a page-aligned start). Data pointers
// sit 64 bytes past the start of a mapping (the header), so they are never aligned (F-122).
#ifndef _WIN32
#include <cstddef>
#include <cstdint>
#include <unistd.h>

namespace pycauset::core {

struct PageRange {
    void* addr;
    size_t len;
};

inline uintptr_t page_size() {
    static const uintptr_t p = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
    return p;
}

// The pages that touch [ptr, ptr + len): for syncing and read-ahead.
inline PageRange pages_covering(const void* ptr, size_t len) {
    const uintptr_t p = reinterpret_cast<uintptr_t>(ptr), page = page_size();
    const uintptr_t start = p & ~(page - 1);
    return {reinterpret_cast<void*>(start), len + (p - start)};
}

// The pages wholly inside [ptr, ptr + len): for dropping pages (never a neighbour's bytes).
inline PageRange pages_inside(const void* ptr, size_t len) {
    const uintptr_t p = reinterpret_cast<uintptr_t>(ptr), page = page_size();
    const uintptr_t start = (p + page - 1) & ~(page - 1), end = (p + len) & ~(page - 1);
    return {reinterpret_cast<void*>(start), end > start ? end - start : 0};
}

}  // namespace pycauset::core
#endif
