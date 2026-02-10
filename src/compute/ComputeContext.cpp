#include "pycauset/compute/ComputeContext.hpp"
#include "pycauset/compute/cpu/CpuDevice.hpp"
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace pycauset {

typedef ComputeDevice* (*CreateDeviceFunc)(const AcceleratorConfig*);

ComputeContext& ComputeContext::instance() {
    static ComputeContext ctx;
    return ctx;
}

ComputeContext::ComputeContext() {
    // AutoSolver initializes with CPU by default
    try_load_cuda(AcceleratorConfig());
}

void ComputeContext::try_load_cuda(const AcceleratorConfig& config) {
    // std::cerr << "[PyCauset] Attempting to load CUDA accelerator..." << std::endl;
#ifdef _WIN32
    const char* lib_name = "pycauset_cuda.dll";
    // Try to load from the same directory as the module if possible, 
    // but for now rely on standard search path (augmented by Python)
    HMODULE handle = LoadLibraryA(lib_name);
    if (!handle) {
        // Silent failure is okay, user can check status manually
        // std::cerr << "[PyCauset] Failed to load " << lib_name << ". Error code: " << GetLastError() << std::endl;
        return;
    }
    // std::cerr << "[PyCauset] Loaded " << lib_name << " successfully." << std::endl;

    CreateDeviceFunc create_func = (CreateDeviceFunc)GetProcAddress(handle, "create_cuda_device");
    if (!create_func) {
        std::cerr << "[PyCauset] Failed to find symbol 'create_cuda_device' in " << lib_name << ". Error code: " << GetLastError() << std::endl;
    }
#else
    const char* lib_name = "libpycauset_cuda.so";
    // RTLD_GLOBAL might be needed if the plugin needs symbols from the main module 
    // that are not exported by default, but we link against it so it should be fine.
    void* handle = dlopen(lib_name, RTLD_LAZY);
    if (!handle) {
        // std::cerr << "[PyCauset] Failed to load " << lib_name << ". Error: " << dlerror() << std::endl;
        return;
    }
    std::cerr << "[PyCauset] Loaded " << lib_name << " successfully." << std::endl;

    CreateDeviceFunc create_func = (CreateDeviceFunc)dlsym(handle, "create_cuda_device");
    if (!create_func) {
        std::cerr << "[PyCauset] Failed to find symbol 'create_cuda_device' in " << lib_name << ". Error: " << dlerror() << std::endl;
    }
#endif

    if (create_func) {
        try {
            ComputeDevice* device = create_func(&config);
            if (device) {
                // std::cerr << "[PyCauset] CUDA device created successfully. Switching to GPU." << std::endl;
                auto_solver_.set_gpu_device(std::unique_ptr<ComputeDevice>(device));
                current_config = config;
            } else {
                std::cerr << "[PyCauset] create_cuda_device() returned null." << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[PyCauset] Exception during CUDA device creation: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[PyCauset] Unknown exception during CUDA device creation." << std::endl;
        }
    }
}



bool ComputeContext::is_gpu_active() const {
    return auto_solver_.is_gpu_active();
}

void ComputeContext::force_backend(BackendPreference pref) {
    auto_solver_.set_backend_preference(pref);
}

BackendPreference ComputeContext::get_backend_preference() const {
    return auto_solver_.get_backend_preference();
}

bool ComputeContext::benchmark_gpu(bool force, HardwareProfile& out) {
    return auto_solver_.benchmark(force, out);
}

bool ComputeContext::get_hardware_profile(HardwareProfile& out) const {
    return auto_solver_.get_hardware_profile(out);
}

void ComputeContext::enable_gpu(const AcceleratorConfig& config) {
    // Always try to load if requested, even if already active (to change config)
    try_load_cuda(config);
}

void ComputeContext::disable_gpu() {
    auto_solver_.disable_gpu();
}

void* ComputeContext::allocate_pinned(size_t size) {
    return auto_solver_.allocate_pinned(size);
}

void ComputeContext::free_pinned(void* ptr) {
    auto_solver_.free_pinned(ptr);
}

void ComputeContext::register_host_memory(void* ptr, size_t size) {
    auto_solver_.register_host_memory(ptr, size);
}

void ComputeContext::unregister_host_memory(void* ptr) {
    auto_solver_.unregister_host_memory(ptr);
}

} // namespace pycauset
