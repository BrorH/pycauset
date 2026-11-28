#pragma once

#include "pycauset/compute/cpu/CpuSolver.hpp"

namespace pycauset {

// The compute device of the entry functions (math/): the CPU kernels. The GPU dispatch layer
// (AutoSolver, a device interface, the CUDA plugin loader and the hardware profile) forwarded
// every call to these and loaded `libpycauset_cuda.so` from the library search path at start;
// it is gone until CUDA returns from experimental/cuda (review 3.4 C-13).
class ComputeContext {
public:
    static ComputeContext& instance();
    CpuSolver* get_device() { return &cpu_; }

private:
    ComputeContext() = default;
    CpuSolver cpu_;
};

}  // namespace pycauset
