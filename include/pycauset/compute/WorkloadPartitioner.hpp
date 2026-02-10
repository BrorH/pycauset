#pragma once

#include <cstdint>
#include <utility>
#include "pycauset/compute/AutoSolver.hpp" 

namespace pycauset {

struct PartitionPlan {
    // MatMul slicing (Row Partitioning of A, Full B, Row Partitioning of C)
    uint64_t cpu_rows_start = 0;
    uint64_t cpu_rows_count = 0;
    
    uint64_t gpu_rows_start = 0;
    uint64_t gpu_rows_count = 0;
    
    bool use_hybrid = false;
};

class WorkloadPartitioner {
public:
    static PartitionPlan partition_matmul(uint64_t n, double cpu_ratio, uint64_t min_chunk_size = 1024);
};

} // namespace pycauset
