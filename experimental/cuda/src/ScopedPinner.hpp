#pragma once

#include <cuda_runtime.h>
#include "pycauset/core/MemoryGovernor.hpp"

namespace pycauset {

// RAII wrapper for pinning host memory using CUDA
class ScopedPinner {
public:
    ScopedPinner(const void* ptr, size_t size) 
        : ptr_(const_cast<void*>(ptr)), size_(size) 
    {
        // Heuristic: Pinning overhead (~0.2ms) is worth it for > 2MB transfers
        if (size_ < 2 * 1024 * 1024) return;

        if (core::MemoryGovernor::instance().try_pin_memory(size_)) {
            cudaError_t err = cudaHostRegister(ptr_, size_, cudaHostRegisterDefault);
            if (err == cudaSuccess) {
                pinned_ = true;
            } else {
                core::MemoryGovernor::instance().unpin_memory(size_);
            }
        }
    }

    ~ScopedPinner() {
        if (pinned_) {
            cudaHostUnregister(ptr_);
            core::MemoryGovernor::instance().unpin_memory(size_);
        }
    }

    // Disable copying
    ScopedPinner(const ScopedPinner&) = delete;
    ScopedPinner& operator=(const ScopedPinner&) = delete;

private:
    void* ptr_;
    size_t size_;
    bool pinned_ = false;
};

} // namespace pycauset
