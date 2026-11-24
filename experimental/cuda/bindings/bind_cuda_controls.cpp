// The former `pycauset.cuda` control surface (removed from the core module when CUDA
// went on hold). Re-attach by calling this from bind_core.cpp when CUDA comes back.
// Needs: ComputeContext, AcceleratorConfig, MemoryGovernor, HardwareProfile, and the
// parse_backend_preference helper below.
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace {
std::string to_lower(std::string s) { for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); return s; }
inline pycauset::BackendPreference parse_backend_preference(const std::string& s_raw) {
    const std::string s = to_lower(s_raw);
    if (s == "auto") return pycauset::BackendPreference::Auto;
    if (s == "cpu") return pycauset::BackendPreference::CPU;
    if (s == "gpu") return pycauset::BackendPreference::GPU;
    throw std::invalid_argument("Unknown backend preference: " + s_raw);
}

}  // namespace

void bind_cuda_controls(py::module_& m) {
    py::module_ cuda = m.def_submodule("cuda", "CUDA acceleration controls");
    cuda.def("is_available", []() { return pycauset::ComputeContext::instance().is_gpu_active(); });

    cuda.def(
        "force_backend",
        [](const std::string& mode) {
            pycauset::ComputeContext::instance().force_backend(parse_backend_preference(mode));
        },
        py::arg("mode"));

    cuda.def(
        "enable",
        [](py::object memory_limit, bool enable_async, int device_id, size_t stream_buffer_size) {
            pycauset::AcceleratorConfig cfg;
            cfg.enable_async = enable_async;
            cfg.device_id = device_id;
            cfg.stream_buffer_size = stream_buffer_size;

            if (!memory_limit.is_none()) {
                cfg.memory_limit_bytes = memory_limit.cast<size_t>();
            }

            // If the CUDA plugin cannot be loaded, this is a no-op.
            pycauset::ComputeContext::instance().enable_gpu(cfg);
        },
        py::arg("memory_limit") = py::none(),
        py::arg("enable_async") = true,
        py::arg("device_id") = 0,
        py::arg("stream_buffer_size") = static_cast<size_t>(1024ULL * 1024 * 64));

    cuda.def("disable", []() { pycauset::ComputeContext::instance().disable_gpu(); });

    cuda.def(
        "set_pinning_budget",
        [](size_t bytes) { pycauset::core::MemoryGovernor::instance().set_pinning_budget_override(bytes); },
        py::arg("bytes"));

    cuda.def(
        "benchmark",
        [](bool force) -> py::object {
            pycauset::HardwareProfile profile;
            if (!pycauset::ComputeContext::instance().benchmark_gpu(force, profile)) {
                return py::none();
            }
            py::dict out;
            out["device_id"] = profile.device_id;
            out["device_name"] = profile.device_name;
            out["cc_major"] = profile.cc_major;
            out["cc_minor"] = profile.cc_minor;
            out["pci_bandwidth_gbps"] = profile.pci_bandwidth_gbps;
            out["sgemm_gflops"] = profile.sgemm_gflops;
            out["dgemm_gflops"] = profile.dgemm_gflops;
            out["timestamp_unix"] = profile.timestamp_unix;
            return out;
        },
        py::arg("force") = false);

    cuda.def("current_device", []() {
        // Always return a string; if GPU isn't active this will be CPU-only.
        return pycauset::ComputeContext::instance().get_device()->name();
    });

}
