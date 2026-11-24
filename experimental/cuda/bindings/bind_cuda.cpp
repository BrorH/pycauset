/**
 * @file bind_cuda.cpp
 * @brief Python bindings for the pycauset CUDA subsystem (discovery + control).
 *
 * This module is built as `_pycauset_cuda` and exposed as `pycauset.cuda`. It is
 * intentionally thin: the heavy acceleration lives in `pycauset_cuda` (the
 * `CudaDevice` plugin loaded via `create_cuda_device`), while this module answers
 * "is a GPU available?" and exposes the device name for skip-guards and the
 * `pycauset.cuda` facade.
 */

#include <pybind11/pybind11.h>

#include <cuda_runtime.h>

#include <string>

namespace py = pybind11;

namespace {

bool cuda_is_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        cudaGetLastError();  // clear the error so it does not poison later CUDA calls
        return false;
    }
    return count > 0;
}

std::string cuda_current_device() {
    if (!cuda_is_available()) {
        return "No CUDA device";
    }
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        return "No CUDA device";
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        return "Unknown CUDA device";
    }
    return std::string(prop.name);
}

}  // namespace

PYBIND11_MODULE(_pycauset_cuda, m) {
    m.doc() = "pycauset CUDA discovery/control module";
    m.def("is_available", &cuda_is_available, "True if a CUDA device is present and usable.");
    m.def("current_device", &cuda_current_device, "Name of the current CUDA device (or a no-device string).");
}
