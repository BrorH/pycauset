#include "pycauset/compute/AutoSolver.hpp"
#include "pycauset/compute/cpu/CpuDevice.hpp"
#include "pycauset/compute/MatrixPropertyTraits.hpp"
#include "pycauset/core/DebugTrace.hpp"
#include "pycauset/core/MemoryGovernor.hpp"
#include "pycauset/core/SystemUtils.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/TriangularBitMatrix.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <cmath>

namespace {

std::filesystem::path get_hardware_profile_path() {
    const std::string home = pycauset::SystemUtils::get_home_directory();
    if (home.empty()) return {};

    std::filesystem::path root = std::filesystem::path(home) / ".pycauset";
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    if (ec) return {};
    return root / "hardware_profile.json";
}

std::string escape_json_string(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    for (char c : input) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

std::optional<double> parse_json_number(const std::string& text, const std::string& key) {
    const std::string token = "\"" + key + "\"";
    size_t pos = text.find(token);
    if (pos == std::string::npos) return std::nullopt;
    pos = text.find(':', pos);
    if (pos == std::string::npos) return std::nullopt;
    pos++;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) pos++;
    if (pos >= text.size()) return std::nullopt;

    const char* start = text.c_str() + pos;
    char* end = nullptr;
    double value = std::strtod(start, &end);
    if (start == end) return std::nullopt;
    return value;
}

std::optional<std::string> parse_json_string(const std::string& text, const std::string& key) {
    const std::string token = "\"" + key + "\"";
    size_t pos = text.find(token);
    if (pos == std::string::npos) return std::nullopt;
    pos = text.find(':', pos);
    if (pos == std::string::npos) return std::nullopt;
    pos++;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) pos++;
    if (pos >= text.size() || text[pos] != '"') return std::nullopt;
    pos++;
    std::string out;
    while (pos < text.size()) {
        char c = text[pos++];
        if (c == '"') break;
        if (c == '\\' && pos < text.size()) {
            char esc = text[pos++];
            switch (esc) {
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case '\\': out += '\\'; break;
                case '"': out += '"'; break;
                default: out += esc; break;
            }
        } else {
            out += c;
        }
    }
    return out;
}

bool load_profile_from_disk(const std::filesystem::path& path, pycauset::HardwareProfile& out) {
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in) return false;
    std::ostringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();
    if (text.empty()) return false;

    auto version = parse_json_number(text, "version");
    if (!version || static_cast<int>(*version) != 1) return false;

    auto device_id = parse_json_number(text, "device_id");
    auto device_name = parse_json_string(text, "device_name");
    auto cc_major = parse_json_number(text, "cc_major");
    auto cc_minor = parse_json_number(text, "cc_minor");
    auto pci_bw = parse_json_number(text, "pci_bandwidth_gbps");
    auto sgemm = parse_json_number(text, "sgemm_gflops");
    auto dgemm = parse_json_number(text, "dgemm_gflops");
    auto timestamp = parse_json_number(text, "timestamp_unix");

    if (!device_id || !device_name || !cc_major || !cc_minor) return false;

    out.version = 1;
    out.device_id = static_cast<int>(*device_id);
    out.device_name = *device_name;
    out.cc_major = static_cast<int>(*cc_major);
    out.cc_minor = static_cast<int>(*cc_minor);
    out.pci_bandwidth_gbps = pci_bw.value_or(0.0);
    out.sgemm_gflops = sgemm.value_or(0.0);
    out.dgemm_gflops = dgemm.value_or(0.0);
    out.timestamp_unix = static_cast<uint64_t>(timestamp.value_or(0.0));
    return true;
}

bool save_profile_to_disk(const std::filesystem::path& path, const pycauset::HardwareProfile& profile) {
    std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out) return false;

    out << "{\n";
    out << "  \"version\": 1,\n";
    out << "  \"device_id\": " << profile.device_id << ",\n";
    out << "  \"device_name\": \"" << escape_json_string(profile.device_name) << "\",\n";
    out << "  \"cc_major\": " << profile.cc_major << ",\n";
    out << "  \"cc_minor\": " << profile.cc_minor << ",\n";
    out << "  \"pci_bandwidth_gbps\": " << profile.pci_bandwidth_gbps << ",\n";
    out << "  \"sgemm_gflops\": " << profile.sgemm_gflops << ",\n";
    out << "  \"dgemm_gflops\": " << profile.dgemm_gflops << ",\n";
    out << "  \"timestamp_unix\": " << profile.timestamp_unix << "\n";
    out << "}\n";
    return true;
}

double bytes_per_element(pycauset::DataType dtype) {
    switch (dtype) {
        case pycauset::DataType::FLOAT64: return 8.0;
        case pycauset::DataType::FLOAT32: return 4.0;
        case pycauset::DataType::INT32: return 4.0;
        case pycauset::DataType::BIT: return 0.125;
        default: return 0.0;
    }
}

bool prefers_cpu_for_properties(const pycauset::MatrixBase& m) {
    const auto traits = pycauset::traits_for(m);
    return traits.is_identity || traits.is_diagonal || traits.is_upper_triangular || traits.is_lower_triangular;
}

} // namespace

namespace pycauset {

AutoSolver::AutoSolver() {
    cpu_device_ = std::make_unique<CpuDevice>();
}

AutoSolver::~AutoSolver() = default;

void AutoSolver::set_backend_preference(BackendPreference pref) {
    backend_preference_ = pref;
}

BackendPreference AutoSolver::get_backend_preference() const {
    return backend_preference_;
}

bool AutoSolver::benchmark(bool force, HardwareProfile& out) {
    if (!run_benchmark(force)) return false;
    if (!hardware_profile_valid_) return false;
    out = hardware_profile_;
    return true;
}

bool AutoSolver::get_hardware_profile(HardwareProfile& out) const {
    if (!hardware_profile_valid_) return false;
    out = hardware_profile_;
    return true;
}

void AutoSolver::set_gpu_device(std::unique_ptr<ComputeDevice> device) {
    gpu_device_ = std::move(device);
    benchmark_done_ = false;
    hardware_profile_valid_ = false;
    if (gpu_device_) {
        apply_dynamic_pinning_budget();
        run_benchmark(false);
    }
}

bool AutoSolver::run_benchmark(bool force) {
    if (!gpu_device_) return false;
    if (benchmark_done_ && !force) return hardware_profile_valid_;

    benchmark_done_ = false;
    hardware_profile_valid_ = false;

    try {
        HardwareProfile device_profile;
        if (!gpu_device_->fill_hardware_profile(device_profile, false)) {
            return false;
        }

        if (!force && load_cached_profile(device_profile)) {
            hardware_profile_valid_ = true;
        } else {
            HardwareProfile profile = device_profile;
            if (!gpu_device_->fill_hardware_profile(profile, true)) {
                return false;
            }
            profile.timestamp_unix = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count());

            hardware_profile_ = profile;
            hardware_profile_valid_ = profile.has_benchmarks();

            if (hardware_profile_valid_) {
                const auto path = get_hardware_profile_path();
                if (!path.empty()) {
                    save_profile_to_disk(path, hardware_profile_);
                }
            }
        }

        auto benchmark_cpu = [this](uint64_t n, bool use_float32) -> double {
            try {
                if (use_float32) {
                    DenseMatrix<float> A(n);
                    DenseMatrix<float> B(n);
                    DenseMatrix<float> C(n);
                    cpu_device_->matmul(A, B, C);
                    auto start = std::chrono::high_resolution_clock::now();
                    cpu_device_->matmul(A, B, C);
                    auto end = std::chrono::high_resolution_clock::now();
                    double seconds = std::chrono::duration<double>(end - start).count();
                    double ops = 2.0 * static_cast<double>(n) * n * n;
                    return seconds > 0.0 ? (ops / seconds) / 1e9 : 0.0;
                }

                DenseMatrix<double> A(n);
                DenseMatrix<double> B(n);
                DenseMatrix<double> C(n);
                cpu_device_->matmul(A, B, C);
                auto start = std::chrono::high_resolution_clock::now();
                cpu_device_->matmul(A, B, C);
                auto end = std::chrono::high_resolution_clock::now();
                double seconds = std::chrono::duration<double>(end - start).count();
                double ops = 2.0 * static_cast<double>(n) * n * n;
                return seconds > 0.0 ? (ops / seconds) / 1e9 : 0.0;
            } catch (...) {
                return 0.0;
            }
        };

        const uint64_t bench_n = 1024;
        cpu_sgemm_gflops_ = benchmark_cpu(bench_n, true);
        cpu_dgemm_gflops_ = benchmark_cpu(bench_n, false);

        if (hardware_profile_valid_ && cpu_dgemm_gflops_ > 0.0 && hardware_profile_.dgemm_gflops > 0.0) {
            gpu_speedup_factor_ = hardware_profile_.dgemm_gflops / cpu_dgemm_gflops_;
        } else {
            gpu_speedup_factor_ = 0.0;
        }
    } catch (const std::exception& e) {
        std::cerr << "[PyCauset] GPU Initialization/Benchmark failed (" << e.what() << "). Disabling GPU." << std::endl;
        gpu_device_.reset();
        gpu_speedup_factor_ = 0.0;
        benchmark_done_ = true;
        return false;
    }

    benchmark_done_ = true;
    return hardware_profile_valid_;
}

void AutoSolver::disable_gpu() {
    gpu_device_.reset();
    hardware_profile_valid_ = false;
    benchmark_done_ = false;
}

bool AutoSolver::is_gpu_active() const {
    return gpu_device_ != nullptr;
}

std::string AutoSolver::name() const {
    if (is_gpu_active()) {
        return "AutoSolver (CPU + " + gpu_device_->name() + ")";
    }
    return "AutoSolver (CPU Only)";
}

bool AutoSolver::is_gpu() const {
    return is_gpu_active();
}

int AutoSolver::preferred_precision() const {
    if (is_gpu_active()) {
        return gpu_device_->preferred_precision();
    }
    return cpu_device_->preferred_precision();
}

bool AutoSolver::load_cached_profile(const HardwareProfile& device_profile) {
    const auto path = get_hardware_profile_path();
    if (path.empty()) return false;

    HardwareProfile cached;
    if (!load_profile_from_disk(path, cached)) return false;
    if (!cached.is_compatible_with(device_profile)) return false;
    if (!cached.has_benchmarks()) return false;

    hardware_profile_ = cached;
    hardware_profile_valid_ = true;
    return true;
}

void AutoSolver::apply_dynamic_pinning_budget() {
    core::MemoryGovernor::instance().apply_dynamic_pinning_budget();
}

double AutoSolver::estimate_gpu_time(double ops, double bytes, DataType dtype) const {
    if (!hardware_profile_valid_) return std::numeric_limits<double>::infinity();

    double bandwidth = hardware_profile_.pci_bandwidth_gbps;
    double gflops = (dtype == DataType::FLOAT32) ? hardware_profile_.sgemm_gflops : hardware_profile_.dgemm_gflops;

    if (bandwidth <= 0.0 || gflops <= 0.0) return std::numeric_limits<double>::infinity();

    double transfer_time = (bytes / (bandwidth * 1e9));
    double compute_time = ops / (gflops * 1e9);
    return transfer_time + compute_time + gpu_dispatch_latency_seconds_;
}

double AutoSolver::estimate_cpu_time(double ops, DataType dtype) const {
    double gflops = (dtype == DataType::FLOAT32) ? cpu_sgemm_gflops_ : cpu_dgemm_gflops_;
    if (gflops <= 0.0) return std::numeric_limits<double>::infinity();
    return ops / (gflops * 1e9);
}

bool AutoSolver::should_use_gpu(double ops, double bytes, DataType dtype) const {
    if (!is_gpu_active()) return false;

    if (backend_preference_ == BackendPreference::CPU) {
        return false;
    }
    if (backend_preference_ == BackendPreference::GPU) {
        return true;
    }

    if (!hardware_profile_valid_) {
        // Pessimistic fallback when the profile is unavailable.
        return false;
    }

    double gpu_time = estimate_gpu_time(ops, bytes, dtype);
    double cpu_time = estimate_cpu_time(ops, dtype);

    if (!std::isfinite(gpu_time) || !std::isfinite(cpu_time)) {
        return false;
    }

    return gpu_time < cpu_time;
}

// --- Memory Management ---

void* AutoSolver::allocate_pinned(size_t size) {
    if (is_gpu_active()) {
        return gpu_device_->allocate_pinned(size);
    }
    return cpu_device_->allocate_pinned(size);
}

void AutoSolver::free_pinned(void* ptr) {
    if (is_gpu_active()) {
        gpu_device_->free_pinned(ptr);
    } else {
        cpu_device_->free_pinned(ptr);
    }
}

void AutoSolver::register_host_memory(void* ptr, size_t size) {
    if (is_gpu_active()) {
        gpu_device_->register_host_memory(ptr, size);
    } else {
        cpu_device_->register_host_memory(ptr, size);
    }
}

void AutoSolver::unregister_host_memory(void* ptr) {
    if (is_gpu_active()) {
        gpu_device_->unregister_host_memory(ptr);
    } else {
        cpu_device_->unregister_host_memory(ptr);
    }
}

// --- Device Selection Logic ---

ComputeDevice* AutoSolver::select_device(uint64_t n_elements) const {
    if (backend_preference_ == BackendPreference::CPU) return cpu_device_.get();
    if (backend_preference_ == BackendPreference::GPU && is_gpu_active()) return gpu_device_.get();

    if (is_gpu_active() && n_elements >= gpu_threshold_elements_) {
        if (gpu_speedup_factor_ < 0.9) {
            return cpu_device_.get();
        }
        return gpu_device_.get();
    }
    return cpu_device_.get();
}

ComputeDevice* AutoSolver::select_device_for_matrix(const MatrixBase& m) const {
    if (backend_preference_ == BackendPreference::CPU) return cpu_device_.get();
    if (backend_preference_ == BackendPreference::GPU && is_gpu_active()) return gpu_device_.get();

    uint64_t elements = m.size();
    if (elements < gpu_threshold_elements_) {
        return cpu_device_.get();
    }

    if (is_gpu_active()) {
        if (gpu_speedup_factor_ < 0.9) {
            return cpu_device_.get();
        }
        return gpu_device_.get();
    }

    return cpu_device_.get();
}

// --- Operations ---

void AutoSolver::matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    // Heuristic: Use GPU if result matrix is large enough AND types are supported
    uint64_t m = a.rows();
    uint64_t k = a.cols();
    uint64_t n = b.cols();

    double ops = 2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n);
    double bytes = 0.0;

    bool use_gpu = false;
    if (is_gpu_active()) {
        if (prefers_cpu_for_properties(a) || prefers_cpu_for_properties(b)) {
            use_gpu = false;
        }
        bool a_ok = (a.get_matrix_type() == MatrixType::DENSE_FLOAT);
        bool b_ok = (b.get_matrix_type() == MatrixType::DENSE_FLOAT);

        DataType dt_a = a.get_data_type();
        DataType dt_b = b.get_data_type();
        DataType dt_r = result.get_data_type();

        bool float_ok = (dt_a == dt_b) && (dt_b == dt_r) && (dt_r == DataType::FLOAT64 || dt_r == DataType::FLOAT32);
        bool bit_ok = (dt_a == DataType::BIT) && (dt_b == DataType::BIT) && (dt_r == DataType::INT32);

        if (a_ok && b_ok && (float_ok || bit_ok) && !prefers_cpu_for_properties(a) && !prefers_cpu_for_properties(b)) {
            if (backend_preference_ == BackendPreference::GPU) {
                use_gpu = true;
            } else if (backend_preference_ == BackendPreference::CPU) {
                use_gpu = false;
            } else if (float_ok) {
                double elem_bytes = bytes_per_element(dt_r);
                bytes = elem_bytes * (static_cast<double>(m) * k + static_cast<double>(k) * n + static_cast<double>(m) * n);
                use_gpu = should_use_gpu(ops, bytes, dt_r);
            } else {
                uint64_t max_size = std::max({a.size(), b.size(), result.size()});
                uint64_t elements = max_size * max_size;
                use_gpu = elements >= gpu_threshold_elements_;
            }
        }
    }
    
    if (use_gpu) {
        try {
            gpu_device_->matmul(a, b, result);
        } catch (const std::exception& e) {
            std::cerr << "[PyCauset] GPU Error in matmul: " << e.what() << ". Falling back to CPU." << std::endl;
            gpu_device_.reset();
            cpu_device_->matmul(a, b, result);
        }
    } else {
        cpu_device_->matmul(a, b, result);
    }
}

void AutoSolver::inverse(const MatrixBase& in, MatrixBase& out) {
    uint64_t n = in.rows();
    double ops = (2.0 / 3.0) * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);

    bool use_gpu = false;
    if (is_gpu_active()) {
        if (in.get_matrix_type() == MatrixType::DENSE_FLOAT) {
            DataType dt = in.get_data_type();
            if (dt == DataType::FLOAT64 || dt == DataType::FLOAT32) {
                if (prefers_cpu_for_properties(in)) {
                    use_gpu = false;
                } else
                if (backend_preference_ == BackendPreference::GPU) {
                    use_gpu = true;
                } else if (backend_preference_ == BackendPreference::CPU) {
                    use_gpu = false;
                } else {
                    double elem_bytes = bytes_per_element(dt);
                    double bytes = elem_bytes * (2.0 * static_cast<double>(n) * n);
                    use_gpu = should_use_gpu(ops, bytes, dt);
                }
            }
        }
    }
    
    if (use_gpu) {
        try {
            gpu_device_->inverse(in, out);
        } catch (const std::exception& e) {
            std::cerr << "[PyCauset] GPU Error in inverse: " << e.what() << ". Falling back to CPU." << std::endl;
            gpu_device_.reset();
            cpu_device_->inverse(in, out);
        }
    } else {
        cpu_device_->inverse(in, out);
    }
}

void AutoSolver::cholesky(const MatrixBase& in, MatrixBase& out) {
    uint64_t n = in.rows();
    double ops = (1.0 / 3.0) * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);

    bool use_gpu = false;
    if (is_gpu_active()) {
        if (in.get_matrix_type() == MatrixType::DENSE_FLOAT) {
            DataType dt = in.get_data_type();
            if (dt == DataType::FLOAT64 || dt == DataType::FLOAT32) {
                if (prefers_cpu_for_properties(in)) {
                    use_gpu = false;
                } else
                if (backend_preference_ == BackendPreference::GPU) {
                    use_gpu = true;
                } else if (backend_preference_ == BackendPreference::CPU) {
                    use_gpu = false;
                } else {
                    double elem_bytes = bytes_per_element(dt);
                    double bytes = elem_bytes * 2.0 * static_cast<double>(n) * n;
                    use_gpu = should_use_gpu(ops, bytes, dt);
                }
            }
        }
    }

    if (use_gpu) {
        try {
            gpu_device_->cholesky(in, out);
        } catch (const std::exception& e) {
            std::cerr << "[PyCauset] GPU Error in cholesky: " << e.what() << ". Falling back to CPU." << std::endl;
            gpu_device_.reset();
            cpu_device_->cholesky(in, out);
        }
    } else {
        cpu_device_->cholesky(in, out);
    }
}

void AutoSolver::batch_gemv(const MatrixBase& A, const double* x_data, double* y_data, size_t b) {
    uint64_t n = A.rows();
    DataType dt = A.get_data_type();
    double ops = 2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(b);
    double elem_bytes = bytes_per_element(dt);
    double bytes = elem_bytes * (static_cast<double>(n) * n + 2.0 * static_cast<double>(n) * b);

    bool use_gpu = false;
    if (is_gpu_active() && (dt == DataType::FLOAT64 || dt == DataType::FLOAT32)) {
        if (backend_preference_ == BackendPreference::GPU) {
            use_gpu = true;
        } else if (backend_preference_ == BackendPreference::CPU) {
            use_gpu = false;
        } else {
            use_gpu = should_use_gpu(ops, bytes, dt);
        }
    }

    if (use_gpu) {
        try {
            gpu_device_->batch_gemv(A, x_data, y_data, b);
        } catch (const std::exception& e) {
            std::cerr << "[PyCauset] GPU Error in batch_gemv: " << e.what() << ". Falling back to CPU." << std::endl;
            gpu_device_.reset();
            cpu_device_->batch_gemv(A, x_data, y_data, b);
        }
    } else {
        cpu_device_->batch_gemv(A, x_data, y_data, b);
    }
}

void AutoSolver::matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, VectorBase& result) {
    // CUDA matrix_vector_multiply is not implemented; keep this CPU-only.
    cpu_device_->matrix_vector_multiply(m, v, result);
}

void AutoSolver::vector_matrix_multiply(const VectorBase& v, const MatrixBase& m, VectorBase& result) {
    // CUDA vector_matrix_multiply is not implemented; keep this CPU-only.
    cpu_device_->vector_matrix_multiply(v, m, result);
}

void AutoSolver::outer_product(const VectorBase& a, const VectorBase& b, MatrixBase& result) {
    // CUDA outer_product is not implemented; keep this CPU-only.
    cpu_device_->outer_product(a, b, result);
}

void AutoSolver::add(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    uint64_t m = a.rows();
    uint64_t n = a.cols();
    double ops = static_cast<double>(m) * static_cast<double>(n);

    bool use_gpu = false;
    if (is_gpu_active()) {
        bool type_ok = (a.get_matrix_type() == MatrixType::DENSE_FLOAT) &&
                       (b.get_matrix_type() == MatrixType::DENSE_FLOAT) &&
                       (result.get_matrix_type() == MatrixType::DENSE_FLOAT);

        DataType dt_a = a.get_data_type();
        DataType dt_b = b.get_data_type();
        DataType dt_r = result.get_data_type();
        bool dtype_ok = (dt_a == dt_b) && (dt_b == dt_r) &&
                        (dt_r == DataType::FLOAT64 || dt_r == DataType::FLOAT32);

        if (type_ok && dtype_ok) {
            if (backend_preference_ == BackendPreference::GPU) {
                use_gpu = true;
            } else if (backend_preference_ == BackendPreference::CPU) {
                use_gpu = false;
            } else {
                double elem_bytes = bytes_per_element(dt_r);
                double bytes = 3.0 * elem_bytes * static_cast<double>(m) * n;
                use_gpu = should_use_gpu(ops, bytes, dt_r);
            }
        }
    }

    if (use_gpu) {
        // Test-only observability: GPU add routing.
        // CpuSolver::add also sets a trace when CPU is chosen.
        if (result.get_data_type() == DataType::FLOAT64) {
            debug_trace::set_last("gpu.add.f64");
        } else if (result.get_data_type() == DataType::FLOAT32) {
            debug_trace::set_last("gpu.add.f32");
        } else {
            debug_trace::set_last("gpu.add");
        }
        try {
            gpu_device_->add(a, b, result);
        } catch (const std::exception& e) {
            std::cerr << "[PyCauset] GPU Error in add: " << e.what() << ". Falling back to CPU." << std::endl;
            gpu_device_.reset();
            cpu_device_->add(a, b, result);
        }
    } else {
        cpu_device_->add(a, b, result);
    }
}

void AutoSolver::subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    uint64_t m = a.rows();
    uint64_t n = a.cols();
    double ops = static_cast<double>(m) * static_cast<double>(n);

    bool use_gpu = false;
    if (is_gpu_active()) {
        bool type_ok = (a.get_matrix_type() == MatrixType::DENSE_FLOAT) &&
                       (b.get_matrix_type() == MatrixType::DENSE_FLOAT) &&
                       (result.get_matrix_type() == MatrixType::DENSE_FLOAT);

        DataType dt_a = a.get_data_type();
        DataType dt_b = b.get_data_type();
        DataType dt_r = result.get_data_type();
        bool dtype_ok = (dt_a == dt_b) && (dt_b == dt_r) &&
                        (dt_r == DataType::FLOAT64 || dt_r == DataType::FLOAT32);

        if (type_ok && dtype_ok) {
            if (backend_preference_ == BackendPreference::GPU) {
                use_gpu = true;
            } else if (backend_preference_ == BackendPreference::CPU) {
                use_gpu = false;
            } else {
                double elem_bytes = bytes_per_element(dt_r);
                double bytes = 3.0 * elem_bytes * static_cast<double>(m) * n;
                use_gpu = should_use_gpu(ops, bytes, dt_r);
            }
        }
    }

    if (use_gpu) {
        // Test-only observability: GPU subtract routing.
        // CpuSolver::subtract sets a trace when CPU is chosen.
        if (result.get_data_type() == DataType::FLOAT64) {
            debug_trace::set_last("gpu.subtract.f64");
        } else if (result.get_data_type() == DataType::FLOAT32) {
            debug_trace::set_last("gpu.subtract.f32");
        } else {
            debug_trace::set_last("gpu.subtract");
        }
        try {
            gpu_device_->subtract(a, b, result);
        } catch (const std::exception& e) {
            std::cerr << "[PyCauset] GPU Error in subtract: " << e.what() << ". Falling back to CPU." << std::endl;
            gpu_device_.reset();
            cpu_device_->subtract(a, b, result);
        }
    } else {
        cpu_device_->subtract(a, b, result);
    }
}

void AutoSolver::elementwise_multiply(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    // CUDA elementwise_multiply is not implemented; keep this CPU-only.
    cpu_device_->elementwise_multiply(a, b, result);
}

void AutoSolver::elementwise_divide(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    // CUDA elementwise_divide is not implemented; keep this CPU-only.
    cpu_device_->elementwise_divide(a, b, result);
}

void AutoSolver::multiply_scalar(const MatrixBase& a, double scalar, MatrixBase& result) {
    uint64_t m = a.rows();
    uint64_t n = a.cols();
    double ops = static_cast<double>(m) * static_cast<double>(n);

    bool use_gpu = false;
    if (is_gpu_active()) {
        bool type_ok = (a.get_matrix_type() == MatrixType::DENSE_FLOAT) &&
                       (result.get_matrix_type() == MatrixType::DENSE_FLOAT);

        DataType dt_a = a.get_data_type();
        DataType dt_r = result.get_data_type();
        bool dtype_ok = (dt_a == dt_r) && (dt_r == DataType::FLOAT64 || dt_r == DataType::FLOAT32);

        if (type_ok && dtype_ok) {
            if (backend_preference_ == BackendPreference::GPU) {
                use_gpu = true;
            } else if (backend_preference_ == BackendPreference::CPU) {
                use_gpu = false;
            } else {
                double elem_bytes = bytes_per_element(dt_r);
                double bytes = 2.0 * elem_bytes * static_cast<double>(m) * n;
                use_gpu = should_use_gpu(ops, bytes, dt_r);
            }
        }
    }

    if (use_gpu) {
        try {
            gpu_device_->multiply_scalar(a, scalar, result);
        } catch (const std::exception& e) {
            std::cerr << "[PyCauset] GPU Error in multiply_scalar: " << e.what() << ". Falling back to CPU." << std::endl;
            gpu_device_.reset();
            cpu_device_->multiply_scalar(a, scalar, result);
        }
    } else {
        cpu_device_->multiply_scalar(a, scalar, result);
    }
}

double AutoSolver::dot(const VectorBase& a, const VectorBase& b) {
    // Always CPU for now
    return cpu_device_->dot(a, b);
}

std::complex<double> AutoSolver::dot_complex(const VectorBase& a, const VectorBase& b) {
    // Always CPU for now
    return cpu_device_->dot_complex(a, b);
}

std::complex<double> AutoSolver::sum(const VectorBase& v) {
    // Always CPU for now
    return cpu_device_->sum(v);
}

double AutoSolver::l2_norm(const VectorBase& v) {
    // Always CPU for now
    return cpu_device_->l2_norm(v);
}

void AutoSolver::add_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) {
    cpu_device_->add_vector(a, b, result);
}

void AutoSolver::subtract_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) {
    cpu_device_->subtract_vector(a, b, result);
}

void AutoSolver::scalar_multiply_vector(const VectorBase& a, double scalar, VectorBase& result) {
    cpu_device_->scalar_multiply_vector(a, scalar, result);
}

void AutoSolver::scalar_multiply_vector_complex(const VectorBase& a, std::complex<double> scalar, VectorBase& result) {
    // Always CPU for now
    cpu_device_->scalar_multiply_vector_complex(a, scalar, result);
}

void AutoSolver::scalar_add_vector(const VectorBase& a, double scalar, VectorBase& result) {
    cpu_device_->scalar_add_vector(a, scalar, result);
}

void AutoSolver::cross_product(const VectorBase& a, const VectorBase& b, VectorBase& result) {
    // Always CPU for now
    cpu_device_->cross_product(a, b, result);
}

std::unique_ptr<TriangularMatrix<double>> AutoSolver::compute_k_matrix(
    const TriangularMatrix<bool>& C,
    double a,
    const std::string& output_path,
    int num_threads
) {
    // Always CPU for now (structured, bit-packed triangular)
    return cpu_device_->compute_k_matrix(C, a, output_path, num_threads);
}

double AutoSolver::frobenius_norm(const MatrixBase& m) {
    // Always CPU for now
    return cpu_device_->frobenius_norm(m);
}

std::complex<double> AutoSolver::sum(const MatrixBase& m) {
    // Always CPU for now
    return cpu_device_->sum(m);
}

double AutoSolver::trace(const MatrixBase& m) {
    // Always CPU for now
    return cpu_device_->trace(m);
}

double AutoSolver::determinant(const MatrixBase& m) {
    // Always CPU for now
    return cpu_device_->determinant(m);
}

void AutoSolver::qr(const MatrixBase& in, MatrixBase& Q, MatrixBase& R) {
    // Always CPU for now
    cpu_device_->qr(in, Q, R);
}

void AutoSolver::lu(const MatrixBase& in, MatrixBase& P, MatrixBase& L, MatrixBase& U) {
    // Always CPU for now
    cpu_device_->lu(in, P, L, U);
}

void AutoSolver::svd(const MatrixBase& in, MatrixBase& U, VectorBase& S, MatrixBase& VT) {
    // Always CPU for now
    cpu_device_->svd(in, U, S, VT);
}

void AutoSolver::solve(const MatrixBase& A, const MatrixBase& B, MatrixBase& X) {
    // Always CPU for now
    cpu_device_->solve(A, B, X);
}

void AutoSolver::eigvals_arnoldi(const MatrixBase& a, VectorBase& out, int k, int m, double tol) {
    DataType dt = a.get_data_type();
    bool use_gpu = false;
    if (is_gpu_active() && (dt == DataType::FLOAT64 || dt == DataType::FLOAT32)) {
        if (backend_preference_ == BackendPreference::GPU) {
            use_gpu = true;
        } else if (backend_preference_ == BackendPreference::CPU) {
            use_gpu = false;
        } else {
            uint64_t n = a.rows();
            double ops = 2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(m);
            double elem_bytes = bytes_per_element(dt);
            double bytes = elem_bytes * static_cast<double>(n) * n;
            use_gpu = should_use_gpu(ops, bytes, dt);
        }
    }

    if (use_gpu) {
        try {
            gpu_device_->eigvals_arnoldi(a, out, k, m, tol);
        } catch (const std::exception& e) {
            std::cerr << "[PyCauset] GPU Error in eigvals_arnoldi: " << e.what() << ". Falling back to CPU." << std::endl;
            gpu_device_.reset();
            cpu_device_->eigvals_arnoldi(a, out, k, m, tol);
        }
    } else {
        cpu_device_->eigvals_arnoldi(a, out, k, m, tol);
    }
}

void AutoSolver::eigh(const MatrixBase& in, VectorBase& eigenvalues, MatrixBase& eigenvectors, char uplo) {
    // TODO: Add GPU support
    cpu_device_->eigh(in, eigenvalues, eigenvectors, uplo);
}

void AutoSolver::eigvalsh(const MatrixBase& in, VectorBase& eigenvalues, char uplo) {
    // TODO: Add GPU support
    cpu_device_->eigvalsh(in, eigenvalues, uplo);
}

void AutoSolver::eig(const MatrixBase& in, VectorBase& eigenvalues, MatrixBase& eigenvectors) {
    // TODO: Add GPU support
    cpu_device_->eig(in, eigenvalues, eigenvectors);
}

void AutoSolver::eigvals(const MatrixBase& in, VectorBase& eigenvalues) {
    // TODO: Add GPU support
    cpu_device_->eigvals(in, eigenvalues);
}

} // namespace pycauset
