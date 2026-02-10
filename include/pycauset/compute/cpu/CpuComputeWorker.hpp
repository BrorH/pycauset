#pragma once

#include "pycauset/compute/ComputeWorker.hpp"
#include "pycauset/compute/cpu/CpuSolver.hpp"

namespace pycauset {

/**
 * @brief CPU implementation of the shared ComputeWorker interface.
 * 
 * Delegates concrete operations to the underlying CpuSolver.
 * Used by streaming drivers to execute CPU-based tile computations.
 */
class CpuComputeWorker : public ComputeWorker {
public:
    explicit CpuComputeWorker(CpuSolver& solver);
    ~CpuComputeWorker() override = default;

    void matmul_tile(
        const MatrixBase& a_tile, 
        const MatrixBase& b_tile, 
        MatrixBase& c_tile,
        double alpha, 
        double beta
    ) override;

    void elementwise_tile(
        const MatrixBase& a_tile,
        const MatrixBase& b_tile,
        MatrixBase& c_tile,
        ElementwiseOp op
    ) override;

private:
    CpuSolver& solver_;
};

} // namespace pycauset
