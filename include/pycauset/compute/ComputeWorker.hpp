#pragma once

#include "pycauset/matrix/MatrixBase.hpp"

namespace pycauset {

/**
 * @brief Shared worker interface for streaming drivers.
 * 
 * The ComputeWorker provides a unified interface for executing operations 
 * on pre-loaded tiles (in-memory). Both CPU and GPU backends must implement this.
 * 
 * Drivers (like MatmulDriver) decompose large problems into tiles, 
 * handle I/O and prefetching, and then delegate the actual computation 
 * to the active ComputeWorker.
 */
class ComputeWorker {
public:
    enum class ElementwiseOp {
        ADD,
        SUBTRACT,
        MULTIPLY,
        DIVIDE
    };

    virtual ~ComputeWorker() = default;

    /**
     * @brief Execute a matrix multiplication on tiles.
     * C = alpha * A * B + beta * C
     * 
     * @param a_tile Input tile A (in-memory)
     * @param b_tile Input tile B (in-memory)
     * @param c_tile Output tile C (in-memory, read/write)
     * @param alpha Scalar multiplier for A*B
     * @param beta Scalar multiplier for C (accumulation)
     */
    virtual void matmul_tile(
        const MatrixBase& a_tile, 
        const MatrixBase& b_tile, 
        MatrixBase& c_tile,
        double alpha, 
        double beta
    ) = 0;

    /**
     * @brief Execute an elementwise operation on tiles.
     * C = A op B
     * 
     * @param a_tile Input tile A
     * @param b_tile Input tile B
     * @param c_tile Output tile C
     * @param op Operation type
     */
    virtual void elementwise_tile(
        const MatrixBase& a_tile,
        const MatrixBase& b_tile,
        MatrixBase& c_tile,
        ElementwiseOp op
    ) = 0;

    // TODO: Add other tile operations (elementwise, inverse, eigen) as they are migrated.
};

} // namespace pycauset
