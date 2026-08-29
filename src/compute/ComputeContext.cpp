#include "pycauset/compute/ComputeContext.hpp"
#include "pycauset/compute/cpu/CpuDevice.hpp"
#include <iostream>
#include <string>
#include <atomic>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace pycauset {

typedef ComputeDevice* (*CreateDeviceFunc)(const AcceleratorConfig*);

// Guards against re-entrancy during ComputeContext::instance() construction.
// MemoryMapper::open_file() consults ComputeContext::instance().is_gpu_active()
// to decide whether to allocate pinned memory for ":memory:" backing. During the
// first instance() call, the CUDA device-load path runs a CPU benchmark that
// constructs DenseMatrix objects, which in turn construct MemoryMappers, i.e.
// they call back into instance() while the function-local static is mid-init.
// That re-enters the magic-static guard and deadlocks. While construction is in
// progress we therefore report "GPU not active" so the mapper takes the plain
// VirtualAlloc path instead of the pinned path.
static std::atomic<bool> g_constructing{false};

struct ConstructionScope {
    ConstructionScope() { g_constructing.store(true, std::memory_order_release); }
    ~ConstructionScope() { g_constructing.store(false, std::memory_order_release); }
};

ComputeContext& ComputeContext::instance() {
    // Intentionally leaked: object teardown (MemoryMapper::close_file, CudaDevice
    // destruction) can reach the context during interpreter finalization, where a
    // Meyers singleton's destruction order is undefined. Leaking it also skips the
    // CUDA handle teardown at exit (cublas/cusolver destroy), which the driver
    // reclaims on process exit anyway, removing a second teardown-hang source.
    static ComputeContext* ctx = new ComputeContext();
    return *ctx;
}

bool compute_context_is_constructing() {
    return g_constructing.load(std::memory_order_acquire);
}

ComputeContext::ComputeContext() {
    // AutoSolver initializes with CPU by default
    ConstructionScope scope;
    try_load_cuda(AcceleratorConfig());
}

void ComputeContext::try_load_cuda(const AcceleratorConfig& config) {
    // std::cerr << "[PyCauset] Attempting to load CUDA accelerator..." << std::endl;
#ifdef _WIN32
    const char* lib_name = "pycauset_cuda.dll";
    // Resolve the plugin relative to this (pycauset_core) DLL's own directory.
    // LoadLibraryA's default search order does NOT include the loading module's
    // directory, so a bare relative name silently fails even when the plugin sits
    // right next to pycauset_core.dll. This is why the GPU never activated.
    std::string resolved_name = lib_name;
    HMODULE core_handle = GetModuleHandleA("pycauset_core.dll");
    if (core_handle) {
        char core_path[MAX_PATH] = {0};
        DWORD len = GetModuleFileNameA(core_handle, core_path, MAX_PATH);
        if (len > 0 && len < MAX_PATH) {
            std::string dir(core_path);
            size_t sep = dir.find_last_of("\\/");
            if (sep != std::string::npos) {
                resolved_name = dir.substr(0, sep + 1) + lib_name;
            }
        }
    }
    HMODULE handle = LoadLibraryA(resolved_name.c_str());
    if (!handle) {
        // Silent failure is okay, user can check status manually
        // std::cerr << "[PyCauset] Failed to load " << resolved_name << ". Error code: " << GetLastError() << std::endl;
        return;
    }
    // std::cerr << "[PyCauset] Loaded " << resolved_name << " successfully." << std::endl;

    CreateDeviceFunc create_func = (CreateDeviceFunc)GetProcAddress(handle, "create_cuda_device");
    if (!create_func) {
        std::cerr << "[PyCauset] Failed to find symbol 'create_cuda_device' in " << resolved_name << ". Error code: " << GetLastError() << std::endl;
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
    if (g_constructing.load(std::memory_order_acquire)) return false;
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
    if (g_constructing.load(std::memory_order_acquire)) return nullptr;
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
