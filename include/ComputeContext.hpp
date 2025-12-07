#pragma once

#include "ComputeDevice.hpp"
#include "AcceleratorConfig.hpp"
#include <memory>

namespace pycauset {

class ComputeContext {
public:
    static ComputeContext& instance();
    
    ComputeDevice* get_device();
    void set_device(std::unique_ptr<ComputeDevice> device);
    
    // Helper to check if GPU is available/active
    bool is_gpu_active() const;

    // Manual control
    void enable_gpu(const AcceleratorConfig& config = AcceleratorConfig());
    void disable_gpu();

    const AcceleratorConfig& get_config() const { return current_config; }

private:
    ComputeContext();
    void try_load_cuda(const AcceleratorConfig& config);
    std::unique_ptr<ComputeDevice> current_device;
    AcceleratorConfig current_config;
};

} // namespace pycauset
