#pragma once

#include "pycauset/compute/ComputeDevice.hpp"
#include "pycauset/compute/cpu/CpuDevice.hpp"
#include "pycauset/compute/AcceleratorConfig.hpp"
#include <memory>
#include <string>

namespace pycauset {

// Forward declaration to avoid including CudaDevice header here if possible,
// but we might need it for the unique_ptr.
// Actually, we can use ComputeDevice pointer for the GPU device to keep this header clean
// of CUDA dependencies if we want, but we need to know it's a CudaDevice to construct it.
class CudaDevice;

class AutoSolver : public ComputeDevice {
public:
    AutoSolver();
    ~AutoSolver() override;

    // Initialize GPU support
    // Takes ownership of the GPU device
    void set_gpu_device(std::unique_ptr<ComputeDevice> device);
    void disable_gpu();
    bool is_gpu_active() const;

    // --- ComputeDevice Interface ---

    void matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) override;
    void inverse(const MatrixBase& in, MatrixBase& out) override;
    void eigvals(const MatrixBase& matrix, ComplexVector& result) override;
    void batch_gemv(const MatrixBase& A, const double* x_data, double* y_data, size_t b) override;

    void matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, VectorBase& result) override;
    void vector_matrix_multiply(const VectorBase& v, const MatrixBase& m, VectorBase& result) override;
    void outer_product(const VectorBase& a, const VectorBase& b, MatrixBase& result) override;

    void add(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) override;
    void subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) override;
    void elementwise_multiply(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) override;
    void multiply_scalar(const MatrixBase& a, double scalar, MatrixBase& result) override;

    double dot(const VectorBase& a, const VectorBase& b) override;
    void add_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) override;
    void subtract_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) override;
    void scalar_multiply_vector(const VectorBase& a, double scalar, VectorBase& result) override;
    void scalar_add_vector(const VectorBase& a, double scalar, VectorBase& result) override;

    std::string name() const override;
    bool is_gpu() const override; // Returns true if GPU is *enabled* and *active*
    int preferred_precision() const override;

    // Memory Management
    void* allocate_pinned(size_t size) override;
    void free_pinned(void* ptr) override;
    void register_host_memory(void* ptr, size_t size) override;
    void unregister_host_memory(void* ptr) override;

private:
    std::unique_ptr<CpuDevice> cpu_device_;
    std::unique_ptr<ComputeDevice> gpu_device_; // Polymorphic to avoid CUDA headers here?
    
    // Thresholds
    uint64_t gpu_threshold_elements_ = 512 * 512; // Default threshold (approx 250K elements)

    // Helper to decide device
    ComputeDevice* select_device(uint64_t n_elements) const;
    ComputeDevice* select_device_for_matrix(const MatrixBase& m) const;
};

} // namespace pycauset
