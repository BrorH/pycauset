#include "pycauset/compute/ComputeWorker.hpp"
#include "pycauset/core/OpRegistry.hpp"

namespace pycauset {

/**
 * @brief Extension Template for adding new Ops to ComputeWorker
 * 
 * To add a new op:
 * 1. Define the virtual method in ComputeWorker.hpp
 * 2. Implement it here (delegating to solver).
 * 3. Register the OpContract in a static block.
 */

/* 
// Example: Adding "trace"
// In ComputeWorker.hpp:
// virtual double trace_tile(const MatrixBase& tile) = 0;

// In CpuComputeWorker.cpp:
double CpuComputeWorker::trace_tile(const MatrixBase& tile) {
    return solver_.trace(tile);
}

// In Op Registration (somewhere central, e.g., Operations.cpp)
static bool register_trace = []() {
    OpContract c;
    c.name = "trace";
    c.supports_streaming = true; // It's a reduction, so yes, if implemented
    OpRegistry::instance().register_op(c);
    return true;
}();
*/

} // namespace pycauset
