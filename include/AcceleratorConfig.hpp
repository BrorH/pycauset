#pragma once

#include <cstddef>

namespace pycauset {

struct AcceleratorConfig {
    int device_id = 0;
    size_t memory_limit_bytes = 0; // 0 = Auto-detect (usually 80-90% of VRAM)
    size_t stream_buffer_size = 1024 * 1024 * 64; // 64MB default for streaming buffers
    bool enable_async = true; // Enable CUDA Streams / Overlap
};

} // namespace pycauset
