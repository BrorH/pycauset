#include "pycauset/compute/WorkloadPartitioner.hpp"
#include <algorithm>

namespace pycauset {

PartitionPlan WorkloadPartitioner::partition_matmul(uint64_t n, double cpu_ratio, uint64_t min_chunk_size) {
    PartitionPlan plan;
    plan.use_hybrid = false;

    // Sanity checks
    if (cpu_ratio <= 0.001) {
        // All GPU
        plan.gpu_rows_start = 0;
        plan.gpu_rows_count = n;
        return plan;
    }
    if (cpu_ratio >= 0.999) {
        // All CPU
        plan.cpu_rows_start = 0;
        plan.cpu_rows_count = n;
        return plan;
    }

    uint64_t cpu_rows = static_cast<uint64_t>(n * cpu_ratio);
    
    // Alignment: Round cpu_rows to nearest multiple of 64 for cache sanity
    // (Optional, but good practice)
    uint64_t align = 64;
    cpu_rows = (cpu_rows + align/2) / align * align;

    // Boundary checks after alignment
    if (cpu_rows < min_chunk_size) {
        // Too small for CPU to be worth the complexity overhead?
        // Actually, if n is massive, 1024 rows is fine. 
        // If n is small (e.g. 2048), cpu_rows might be 200.
        // Let's rely on min_chunk_size.
        if (cpu_rows == 0) {
             plan.gpu_rows_start = 0;
             plan.gpu_rows_count = n;
             return plan;
        }
    }
    
    // Ensure GPU also gets a meaningful chunk
    if (n - cpu_rows < min_chunk_size) {
        // Just give it all to CPU
        plan.cpu_rows_start = 0;
        plan.cpu_rows_count = n;
        return plan;
    }

    // Valid Hybrid Split
    plan.use_hybrid = true;
    
    // Strategy: CPU takes the TOP rows (0 to k), GPU takes BOTTOM rows (k to N).
    // Or vice versa?
    // CPU memory is contiguous. 
    // GPU transfer involves pinned memory.
    // It doesn't matter much for functionality.
    
    plan.cpu_rows_start = 0;
    plan.cpu_rows_count = cpu_rows;
    
    plan.gpu_rows_start = cpu_rows;
    plan.gpu_rows_count = n - cpu_rows;
    
    return plan;
}

} // namespace pycauset
