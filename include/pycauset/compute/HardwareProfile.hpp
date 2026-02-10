#pragma once

// HardwareProfile
// What: Stores GPU hardware properties and micro-benchmark results for dispatch modeling.
// Why: AutoSolver uses this to apply cost-model routing and to persist hardware audit data.
// Dependencies: Filled by ComputeDevice implementations (e.g., CudaDevice) and consumed by AutoSolver.

#include <cstdint>
#include <string>

namespace pycauset {

struct HardwareProfile {
    int version = 1;
    int device_id = -1;
    std::string device_name;
    int cc_major = 0;
    int cc_minor = 0;
    double pci_bandwidth_gbps = 0.0;
    double sgemm_gflops = 0.0;
    double dgemm_gflops = 0.0;
    uint64_t timestamp_unix = 0;

    bool is_compatible_with(const HardwareProfile& device) const {
        return device_id == device.device_id &&
               cc_major == device.cc_major &&
               cc_minor == device.cc_minor &&
               device_name == device.device_name;
    }

    bool has_benchmarks() const {
        return pci_bandwidth_gbps > 0.0 && (sgemm_gflops > 0.0 || dgemm_gflops > 0.0);
    }
};

} // namespace pycauset
