#include "pycauset/compute/cpu/CpuComputeWorker.hpp"
#include "pycauset/compute/cpu/CpuSolver.hpp"

namespace pycauset {

CpuComputeWorker::CpuComputeWorker(CpuSolver& solver) 
    : solver_(solver) {}

void CpuComputeWorker::matmul_tile(
    const MatrixBase& a_tile, 
    const MatrixBase& b_tile, 
    MatrixBase& c_tile,
    double alpha, 
    double beta
) {
    // Delegate to the solver's GEMM implementation
    // The solver is responsible for dispatching to BLAS or fallback based on dtypes.
    solver_.gemm(a_tile, b_tile, c_tile, alpha, beta);
}

void CpuComputeWorker::elementwise_tile(
    const MatrixBase& a_tile,
    const MatrixBase& b_tile,
    MatrixBase& c_tile,
    ElementwiseOp op
) {
    switch (op) {
        case ElementwiseOp::ADD:
            solver_.add(a_tile, b_tile, c_tile);
            break;
        case ElementwiseOp::SUBTRACT:
            solver_.subtract(a_tile, b_tile, c_tile);
            break;
        case ElementwiseOp::MULTIPLY:
            solver_.elementwise_multiply(a_tile, b_tile, c_tile);
            break;
        case ElementwiseOp::DIVIDE:
            solver_.elementwise_divide(a_tile, b_tile, c_tile);
            break;
        default:
            throw std::runtime_error("Unsupported elementwise operation in CpuComputeWorker");
    }
}

} // namespace pycauset
