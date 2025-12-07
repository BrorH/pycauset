#pragma once

#include "ComputeDevice.hpp"
#include "AcceleratorConfig.hpp"
#include "AutoSolver.hpp"
#include <memory>

namespace pycauset {

class ComputeContext {
public:
    static ComputeContext& instance();
    
    ComputeDevice* get_device() { return &auto_solver_; }
    // void set_device(std::unique_ptr<ComputeDevice> device); // Removed, use AutoSolver
    
    // Helper to check if GPU is available/active
    bool is_gpu_active() const;

    // Manual control
    void enable_gpu(const AcceleratorConfig& config = AcceleratorConfig());
    void disable_gpu();

    const AcceleratorConfig& get_config() const { return current_config; }

    // Memory Management
    void* allocate_pinned(size_t size);
    void free_pinned(void* ptr);
    void register_host_memory(void* ptr, size_t size);
    void unregister_host_memory(void* ptr);

private:
    ComputeContext();
    void try_load_cuda(const AcceleratorConfig& config);
    
    AutoSolver auto_solver_;
    AcceleratorConfig current_config;
};

} // namespace pycauset
