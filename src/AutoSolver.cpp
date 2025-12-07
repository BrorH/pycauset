#include "AutoSolver.hpp"
#include "CpuDevice.hpp"
#include <iostream>
#include <algorithm>

namespace pycauset {

AutoSolver::AutoSolver() {
    cpu_device_ = std::make_unique<CpuDevice>();
}

AutoSolver::~AutoSolver() = default;

void AutoSolver::set_gpu_device(std::unique_ptr<ComputeDevice> device) {
    gpu_device_ = std::move(device);
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
        return gpu_device_.get();
    }
    return cpu_device_.get();
}

ComputeDevice* AutoSolver::select_device_for_matrix(const MatrixBase& m) const {
    // Check if matrix type is supported by GPU
    // Currently CudaDevice supports Dense, DenseBit, and handles Triangular/Diagonal by converting/copying?
    // Actually CudaDevice::matmul checks for DenseBitMatrix.
    // For now, we assume GPU supports everything via fallback or direct implementation,
    // but for performance, we only send large matrices.
    
    // Special case: If matrix is very small, always CPU.
    uint64_t n = m.size();
    if (n * n < gpu_threshold_elements_) {
        return cpu_device_.get();
    }
    
    if (is_gpu_active()) {
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
        
        // Also check data types
        // We support FLOAT64, FLOAT32, BIT
        auto is_supported_dtype = [](DataType dt) {
            return dt == DataType::FLOAT64 || dt == DataType::FLOAT32 || dt == DataType::BIT;
        };
        
        if (a_ok && b_ok && is_supported_dtype(a.get_data_type()) && is_supported_dtype(b.get_data_type())) {
            use_gpu = true;
        }
    }
    
    if (use_gpu) {
        gpu_device_->matmul(a, b, result);
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
        gpu_device_->inverse(in, out);
    } else {
        cpu_device_->inverse(in, out);
    }
}

void AutoSolver::eigvals(const MatrixBase& matrix, ComplexVector& result) {
    uint64_t elements = matrix.size() * matrix.size();
    
    bool use_gpu = false;
    if (is_gpu_active() && elements >= gpu_threshold_elements_) {
        // GPU supports Dense Float/Double eigvals
        if (matrix.get_matrix_type() == MatrixType::DENSE_FLOAT) {
            DataType dt = matrix.get_data_type();
            if (dt == DataType::FLOAT64 || dt == DataType::FLOAT32) {
                use_gpu = true;
            }
        }
    }
    
    if (use_gpu) {
        gpu_device_->eigvals(matrix, result);
    } else {
        cpu_device_->eigvals(matrix, result);
    }
}

void AutoSolver::batch_gemv(const MatrixBase& A, const double* x_data, double* y_data, size_t b) {
    // A is N*N.
    uint64_t elements = A.size() * A.size();
    select_device(elements)->batch_gemv(A, x_data, y_data, b);
}

void AutoSolver::matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, VectorBase& result) {
    uint64_t elements = m.size() * m.size();
    select_device(elements)->matrix_vector_multiply(m, v, result);
}

void AutoSolver::vector_matrix_multiply(const VectorBase& v, const MatrixBase& m, VectorBase& result) {
    uint64_t elements = m.size() * m.size();
    select_device(elements)->vector_matrix_multiply(v, m, result);
}

void AutoSolver::outer_product(const VectorBase& a, const VectorBase& b, MatrixBase& result) {
    uint64_t elements = a.size() * b.size();
    select_device(elements)->outer_product(a, b, result);
}

void AutoSolver::add(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    uint64_t elements = a.size() * a.size();
    select_device(elements)->add(a, b, result);
}

void AutoSolver::subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    uint64_t elements = a.size() * a.size();
    select_device(elements)->subtract(a, b, result);
}

void AutoSolver::elementwise_multiply(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    uint64_t elements = a.size() * a.size();
    select_device(elements)->elementwise_multiply(a, b, result);
}

void AutoSolver::multiply_scalar(const MatrixBase& a, double scalar, MatrixBase& result) {
    uint64_t elements = a.size() * a.size();
    select_device(elements)->multiply_scalar(a, scalar, result);
}

double AutoSolver::dot(const VectorBase& a, const VectorBase& b) {
    // Always CPU for now
    return cpu_device_->dot(a, b);
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

void AutoSolver::scalar_add_vector(const VectorBase& a, double scalar, VectorBase& result) {
    cpu_device_->scalar_add_vector(a, scalar, result);
}

} // namespace pycauset
