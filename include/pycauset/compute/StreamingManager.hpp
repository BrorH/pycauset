#pragma once

#include "pycauset/compute/ComputeWorker.hpp"
#include "pycauset/matrix/MatrixBase.hpp"
#include <vector>

namespace pycauset {

/**
 * @brief Manages out-of-core / tiled execution for operations that exceed RAM budgets.
 * 
 * The StreamingManager is responsible for:
 * 1. Planning: determining optimal tile sizes based on MemoryGovernor budgets.
 * 2. Orchestration: driving the loop over tiles (prefetching, double-buffering logic).
 * 3. Execution: dispatching tile computations to the abstract ComputeWorker.
 * 
 * It is backend-agnostic; it works with any ComputeWorker (CPU or GPU).
 */
class StreamingManager {
public:
    explicit StreamingManager(ComputeWorker& worker);
    
    /**
     * @brief Executes a tiled Matrix Multiplication: C = A * B
     * 
     * Handles huge matrices by breaking them into tiles that fit in allowed RAM.
     * 
     * @param a_dense Input matrix A (huge)
     * @param b_dense Input matrix B (huge)
     * @param c_dense Output matrix C (huge)
     */
    void matmul(const MatrixBase& a_dense, const MatrixBase& b_dense, MatrixBase& c_dense);

    /**
     * @brief Executes a tiled elementwise operation: C = A op B
     * 
     * @param a_dense Input matrix A (huge)
     * @param b_dense Input matrix B (huge)
     * @param c_dense Output matrix C (huge)
     * @param op The operation to perform
     */
    void elementwise(
        const MatrixBase& a_dense, 
        const MatrixBase& b_dense, 
        MatrixBase& c_dense, 
        ComputeWorker::ElementwiseOp op
    );

private:
    ComputeWorker& worker_;

    // Helper to calculate optimal block size
    size_t calculate_block_size(size_t n, size_t element_size);
};

} // namespace pycauset
