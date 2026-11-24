#pragma once

#include "pycauset/compute/ComputeDevice.hpp"
#include "ScopedPinner.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <memory>

namespace pycauset {

class CudaFuture : public ComputeFuture {
public:
    CudaFuture() {
        cudaStreamCreate(&stream_);
        cudaEventCreate(&event_);
    }

    ~CudaFuture() {
        cudaEventDestroy(event_);
        cudaStreamDestroy(stream_);
    }

    void record() {
        cudaEventRecord(event_, stream_);
    }
    
    cudaStream_t get_stream() const { return stream_; }

    void add_pinner(std::unique_ptr<ScopedPinner> pinner) {
        resources_.push_back(std::move(pinner));
    }

    void wait() override {
        cudaEventSynchronize(event_);
        // Resources are freed when destructor runs or vector clears
    }

    bool is_ready() override {
        cudaError_t status = cudaEventQuery(event_);
        return status == cudaSuccess;
    }

private:
    cudaEvent_t event_;
    cudaStream_t stream_;
    std::vector<std::unique_ptr<ScopedPinner>> resources_;
};

} // namespace pycauset
