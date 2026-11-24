#pragma once

#include "pycauset/compute/ComputeDevice.hpp"
#include "pycauset/compute/AcceleratorConfig.hpp"
#include "pycauset/compute/AutoSolver.hpp"
#include "pycauset/compute/HardwareProfile.hpp"
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

    // Routing control
    void force_backend(BackendPreference pref);
    BackendPreference get_backend_preference() const;

    // Hardware profiling
    bool benchmark_gpu(bool force, HardwareProfile& out);
    bool get_hardware_profile(HardwareProfile& out) const;

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
