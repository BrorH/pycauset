#include "pycauset/compute/AutoSolver.hpp"
#include "pycauset/compute/cpu/CpuDevice.hpp"
#include "pycauset/core/DebugTrace.hpp"
#include "pycauset/core/MemoryGovernor.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/TriangularBitMatrix.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include <iostream>
#include <algorithm>
#include <chrono>

namespace pycauset {

AutoSolver::AutoSolver() {
    cpu_device_ = std::make_unique<CpuDevice>();
}

AutoSolver::~AutoSolver() = default;

void AutoSolver::set_gpu_device(std::unique_ptr<ComputeDevice> device) {
    gpu_device_ = std::move(device);
    if (gpu_device_) {
        run_benchmark();
    }
}

void AutoSolver::run_benchmark() {
    if (benchmark_done_ || !gpu_device_) return;
    
    try {
        // Micro-benchmark: 1024x1024 Matrix Multiply
        uint64_t n = 1024;
        DenseMatrix<double> A(n);
        DenseMatrix<double> B(n);
        DenseMatrix<double> C_cpu(n);
        DenseMatrix<double> C_gpu(n);
        
        // Initialize with dummy data to avoid NaN/Inf issues
        // We don't need random data for performance test, just valid floats.
        // Parallel initialization to be fast
        // Actually, A and B are zero-initialized by default constructor (or uninitialized?)
        // DenseMatrix constructor initializes storage.
        // Let's just run.
        
        // Warmup CPU
        cpu_device_->matmul(A, B, C_cpu);
        
        // Measure CPU
        auto start_cpu = std::chrono::high_resolution_clock::now();
        cpu_device_->matmul(A, B, C_cpu);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        
        // Warmup GPU
        gpu_device_->matmul(A, B, C_gpu);
        
        // Measure GPU
        auto start_gpu = std::chrono::high_resolution_clock::now();
        gpu_device_->matmul(A, B, C_gpu);
        auto end_gpu = std::chrono::high_resolution_clock::now();
        
        double cpu_time = std::chrono::duration<double>(end_cpu - start_cpu).count();
        double gpu_time = std::chrono::duration<double>(end_gpu - start_gpu).count();
        
        if (gpu_time > 0) {
            gpu_speedup_factor_ = cpu_time / gpu_time;
        }
        
        // std::cout << "[PyCauset] Hardware Profiling: CPU=" << cpu_time << "s, GPU=" << gpu_time << "s. Speedup=" << gpu_speedup_factor_ << "x." << std::endl;
        
    } catch (const std::exception& e) {
        // R1_SAFETY: Pessimistic Fallback
        std::cerr << "[PyCauset] GPU Initialization/Benchmark failed (" << e.what() << "). Disabling GPU." << std::endl;
        gpu_device_.reset();
        gpu_speedup_factor_ = 0.0;
    }
    
    benchmark_done_ = true;
}

void AutoSolver::disable_gpu() {
    gpu_device_.reset();
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
    if (is_gpu_active() && n_elements >= gpu_threshold_elements_) {
        // Smart Dispatch: If GPU is slower than CPU (speedup < 0.9), prefer CPU.
        if (gpu_speedup_factor_ < 0.9) {
            return cpu_device_.get();
        }
        return gpu_device_.get();
    }
    return cpu_device_.get();
}

ComputeDevice* AutoSolver::select_device_for_matrix(const MatrixBase& m) const {
    uint64_t n = m.size();
    uint64_t elements = n * n;
    
    if (elements < gpu_threshold_elements_) {
        return cpu_device_.get();
    }
    
    if (is_gpu_active()) {
        // Smart Dispatch
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
    
    uint64_t max_size = std::max({a.size(), b.size(), result.size()});
    uint64_t elements = max_size * max_size;
    
    bool use_gpu = false;
    if (is_gpu_active() && elements >= gpu_threshold_elements_) {
        // Check types
        // GPU currently supports DenseMatrix (Float32/64) and DenseBitMatrix
        bool a_ok = (a.get_matrix_type() == MatrixType::DENSE_FLOAT);
        bool b_ok = (b.get_matrix_type() == MatrixType::DENSE_FLOAT);
        
        // Also check data types. GPU kernels require dtype compatibility.
        DataType dt_a = a.get_data_type();
        DataType dt_b = b.get_data_type();
        DataType dt_r = result.get_data_type();

        bool float_ok = (dt_a == dt_b) && (dt_b == dt_r) && (dt_r == DataType::FLOAT64 || dt_r == DataType::FLOAT32);
        bool bit_ok = (dt_a == DataType::BIT) && (dt_b == DataType::BIT) && (dt_r == DataType::INT32);

        if (a_ok && b_ok && (float_ok || bit_ok)) {
            use_gpu = true;
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
    uint64_t elements = in.size() * in.size();
    
    bool use_gpu = false;
    if (is_gpu_active() && elements >= gpu_threshold_elements_) {
        // GPU supports Dense Float/Double inverse
        if (in.get_matrix_type() == MatrixType::DENSE_FLOAT) {
            DataType dt = in.get_data_type();
            if (dt == DataType::FLOAT64 || dt == DataType::FLOAT32) {
                use_gpu = true;
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

void AutoSolver::batch_gemv(const MatrixBase& A, const double* x_data, double* y_data, size_t b) {
    // A is N*N.
    uint64_t elements = A.size() * A.size();
    ComputeDevice* device = select_device(elements);
    
    if (device == gpu_device_.get()) {
        try {
            device->batch_gemv(A, x_data, y_data, b);
        } catch (const std::exception& e) {
             std::cerr << "[PyCauset] GPU Error in batch_gemv: " << e.what() << ". Falling back to CPU." << std::endl;
             gpu_device_.reset();
             cpu_device_->batch_gemv(A, x_data, y_data, b);
        }
    } else {
        device->batch_gemv(A, x_data, y_data, b);
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
    uint64_t elements = a.size() * a.size();

    bool use_gpu = false;
    if (is_gpu_active() && elements >= gpu_threshold_elements_) {
        // CUDA add supports only dense float32/float64 with matching dtypes.
        bool type_ok = (a.get_matrix_type() == MatrixType::DENSE_FLOAT) &&
                       (b.get_matrix_type() == MatrixType::DENSE_FLOAT) &&
                       (result.get_matrix_type() == MatrixType::DENSE_FLOAT);

        DataType dt_a = a.get_data_type();
        DataType dt_b = b.get_data_type();
        DataType dt_r = result.get_data_type();
        bool dtype_ok = (dt_a == dt_b) && (dt_b == dt_r) &&
                        (dt_r == DataType::FLOAT64 || dt_r == DataType::FLOAT32);

        use_gpu = type_ok && dtype_ok;
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
    uint64_t elements = a.size() * a.size();

    bool use_gpu = false;
    if (is_gpu_active() && elements >= gpu_threshold_elements_) {
        // CUDA subtract supports only dense float32/float64 with matching dtypes.
        bool type_ok = (a.get_matrix_type() == MatrixType::DENSE_FLOAT) &&
                       (b.get_matrix_type() == MatrixType::DENSE_FLOAT) &&
                       (result.get_matrix_type() == MatrixType::DENSE_FLOAT);

        DataType dt_a = a.get_data_type();
        DataType dt_b = b.get_data_type();
        DataType dt_r = result.get_data_type();
        bool dtype_ok = (dt_a == dt_b) && (dt_b == dt_r) &&
                        (dt_r == DataType::FLOAT64 || dt_r == DataType::FLOAT32);

        use_gpu = type_ok && dtype_ok;
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
    uint64_t elements = a.size() * a.size();

    bool use_gpu = false;
    if (is_gpu_active() && elements >= gpu_threshold_elements_) {
        // CUDA multiply_scalar supports only dense float32/float64 with matching dtypes.
        bool type_ok = (a.get_matrix_type() == MatrixType::DENSE_FLOAT) &&
                       (result.get_matrix_type() == MatrixType::DENSE_FLOAT);

        DataType dt_a = a.get_data_type();
        DataType dt_r = result.get_data_type();
        bool dtype_ok = (dt_a == dt_r) && (dt_r == DataType::FLOAT64 || dt_r == DataType::FLOAT32);

        use_gpu = type_ok && dtype_ok;
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

} // namespace pycauset
