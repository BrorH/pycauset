#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace pycauset {

/**
 * @brief Manages double-buffered asynchronous streaming between Host and Device.
 * 
 * This class handles:
 * - Pinned Host Memory allocation (cudaMallocHost)
 * - Device Memory allocation
 * - Independent Transfer Stream
 * - Synchronization Events
 * 
 * Usage:
 * 1. Producer (CPU) calls wait_for_write_buffer() to ensure buffer is free.
 * 2. Producer writes to get_host_write_buffer().
 * 3. Producer calls submit_transfer() to start H2D copy.
 * 4. Consumer (GPU) calls get_device_read_buffer() to get the pointer (injects wait on compute stream).
 * 5. Consumer launches kernels using this pointer.
 * 6. Consumer calls release_device_buffer() to mark it as done (injects event record).
 */
template <typename T>
class AsyncStreamer {
public:
    AsyncStreamer(size_t buffer_elements, int device_id, bool enable_async = true) 
        : buffer_size_(buffer_elements), enable_async_(enable_async) {
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess) throw std::runtime_error("AsyncStreamer: cudaSetDevice failed");

        // Allocate Pinned Memory (Host) - 2 buffers
        err = cudaMallocHost((void**)&h_buffers_[0], buffer_size_ * sizeof(T));
        if (err != cudaSuccess) throw std::runtime_error("AsyncStreamer: cudaMallocHost failed");
        
        err = cudaMallocHost((void**)&h_buffers_[1], buffer_size_ * sizeof(T));
        if (err != cudaSuccess) { cudaFreeHost(h_buffers_[0]); throw std::runtime_error("AsyncStreamer: cudaMallocHost failed"); }
        
        // Allocate Device Memory - 2 buffers
        err = cudaMalloc(&d_buffers_[0], buffer_size_ * sizeof(T));
        if (err != cudaSuccess) throw std::runtime_error("AsyncStreamer: cudaMalloc failed");
        
        err = cudaMalloc(&d_buffers_[1], buffer_size_ * sizeof(T));
        if (err != cudaSuccess) throw std::runtime_error("AsyncStreamer: cudaMalloc failed");
        
        // Create Transfer Stream (Non-blocking to allow overlap with Compute stream)
        cudaStreamCreateWithFlags(&stream_transfer_, cudaStreamNonBlocking);
        
        // Events for synchronization
        // Disable timing for performance
        cudaEventCreateWithFlags(&event_transfer_complete_[0], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&event_transfer_complete_[1], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&event_compute_complete_[0], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&event_compute_complete_[1], cudaEventDisableTiming);
    }

    ~AsyncStreamer() {
        if (h_buffers_[0]) cudaFreeHost(h_buffers_[0]);
        if (h_buffers_[1]) cudaFreeHost(h_buffers_[1]);
        if (d_buffers_[0]) cudaFree(d_buffers_[0]);
        if (d_buffers_[1]) cudaFree(d_buffers_[1]);
        
        cudaStreamDestroy(stream_transfer_);
        
        cudaEventDestroy(event_transfer_complete_[0]);
        cudaEventDestroy(event_transfer_complete_[1]);
        cudaEventDestroy(event_compute_complete_[0]);
        cudaEventDestroy(event_compute_complete_[1]);
    }

    // Get the buffer that CPU should write to (Next Batch)
    T* get_host_write_buffer() {
        return h_buffers_[write_idx_];
    }

    // CPU Sync: Wait until the NEXT write buffer is free to write to
    // (i.e., GPU has finished computing the PREVIOUS usage of this buffer)
    void wait_for_write_buffer() {
        // We want to write to 'write_idx_'.
        // We must ensure the GPU is done with it (from the previous cycle).
        // If this is the first time we use this index, the event is not recorded, returns immediately.
        if (enable_async_) {
            cudaEventSynchronize(event_compute_complete_[write_idx_]);
        }
    }

    // Submit the transfer of the current write buffer to the device
    // This records an event that the transfer is done.
    // 'count' allows transferring less than full buffer size
    void submit_transfer(size_t count) {
        if (count > buffer_size_) count = buffer_size_;

        // Copy Host[write_idx] -> Device[write_idx] on transfer stream
        cudaMemcpyAsync(d_buffers_[write_idx_], h_buffers_[write_idx_], 
                        count * sizeof(T), cudaMemcpyHostToDevice, stream_transfer_);
        
        // Record event: Transfer for this index is done
        cudaEventRecord(event_transfer_complete_[write_idx_], stream_transfer_);
        
        if (!enable_async_) {
            cudaStreamSynchronize(stream_transfer_);
        }

        // Toggle index
        write_idx_ = 1 - write_idx_;
    }

    // Get the buffer that GPU should read from (Current Batch)
    // This waits (on the compute stream) for the transfer to complete.
    T* get_device_read_buffer(cudaStream_t compute_stream) {
        // The compute stream must wait for the transfer of 'read_idx_' to complete
        cudaStreamWaitEvent(compute_stream, event_transfer_complete_[read_idx_], 0);
        
        return d_buffers_[read_idx_];
    }
    
    // Mark that the GPU is done reading this buffer
    // This allows the CPU to overwrite it in the next cycle
    void release_device_buffer(cudaStream_t compute_stream) {
        // Record event: Compute for this index is done
        cudaEventRecord(event_compute_complete_[read_idx_], compute_stream);
        
        if (!enable_async_) {
            cudaStreamSynchronize(compute_stream);
        }

        // Toggle read index
        read_idx_ = 1 - read_idx_;
    }

    // Helper to synchronize everything (e.g. at end of job)
    void sync_all() {
        cudaStreamSynchronize(stream_transfer_);
    }

private:
    size_t buffer_size_;
    T* h_buffers_[2] = {nullptr, nullptr};
    T* d_buffers_[2] = {nullptr, nullptr};
    
    cudaStream_t stream_transfer_;
    
    cudaEvent_t event_transfer_complete_[2];
    cudaEvent_t event_compute_complete_[2];
    
    int write_idx_ = 0; // CPU writes here
    int read_idx_ = 0;  // GPU reads here
    bool enable_async_;
};

} // namespace pycauset
