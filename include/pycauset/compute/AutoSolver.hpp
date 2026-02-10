#pragma once

#include "pycauset/compute/ComputeDevice.hpp"
#include "pycauset/compute/cpu/CpuDevice.hpp"
#include "pycauset/compute/AcceleratorConfig.hpp"
#include "pycauset/compute/HardwareProfile.hpp"
#include <complex>
#include <memory>
#include <string>

namespace pycauset {

// Forward declaration to avoid including CudaDevice header here if possible,
// but we might need it for the unique_ptr.
// Actually, we can use ComputeDevice pointer for the GPU device to keep this header clean
// of CUDA dependencies if we want, but we need to know it's a CudaDevice to construct it.
class CudaDevice;

enum class BackendPreference {
    Auto = 0,
    CPU = 1,
    GPU = 2
};

class AutoSolver : public ComputeDevice {
public:
    AutoSolver();
    ~AutoSolver() override;

    // Initialize GPU support
    // Takes ownership of the GPU device
    void set_gpu_device(std::unique_ptr<ComputeDevice> device);
    void disable_gpu();
    bool is_gpu_active() const;

    // Routing control
    void set_backend_preference(BackendPreference pref);
    BackendPreference get_backend_preference() const;

    // Hardware profiling
    bool benchmark(bool force, HardwareProfile& out);
    bool get_hardware_profile(HardwareProfile& out) const;

    // --- ComputeDevice Interface ---

    void matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) override;
    void inverse(const MatrixBase& in, MatrixBase& out) override;
    void cholesky(const MatrixBase& in, MatrixBase& out) override;
    void batch_gemv(const MatrixBase& A, const double* x_data, double* y_data, size_t b) override;

    void matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, VectorBase& result) override;
    void vector_matrix_multiply(const VectorBase& v, const MatrixBase& m, VectorBase& result) override;
    void outer_product(const VectorBase& a, const VectorBase& b, MatrixBase& result) override;

    void add(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) override;
    void subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) override;
    void elementwise_multiply(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) override;
    void elementwise_divide(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) override;
    void multiply_scalar(const MatrixBase& a, double scalar, MatrixBase& result) override;

    double dot(const VectorBase& a, const VectorBase& b) override;
    std::complex<double> dot_complex(const VectorBase& a, const VectorBase& b) override;
    std::complex<double> sum(const VectorBase& v) override;
    double l2_norm(const VectorBase& v) override;
    void add_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) override;
    void subtract_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) override;
    void scalar_multiply_vector(const VectorBase& a, double scalar, VectorBase& result) override;
    void scalar_multiply_vector_complex(const VectorBase& a, std::complex<double> scalar, VectorBase& result) override;
    void scalar_add_vector(const VectorBase& a, double scalar, VectorBase& result) override;

    void cross_product(const VectorBase& a, const VectorBase& b, VectorBase& result) override;

    std::unique_ptr<TriangularMatrix<double>> compute_k_matrix(
        const TriangularMatrix<bool>& C,
        double a,
        const std::string& output_path,
        int num_threads) override;

    double frobenius_norm(const MatrixBase& m) override;
    std::complex<double> sum(const MatrixBase& m) override;
    double trace(const MatrixBase& m) override;
    double determinant(const MatrixBase& m) override;
    void qr(const MatrixBase& in, MatrixBase& Q, MatrixBase& R) override;
    void lu(const MatrixBase& in, MatrixBase& P, MatrixBase& L, MatrixBase& U) override;
    void svd(const MatrixBase& in, MatrixBase& U, VectorBase& S, MatrixBase& VT) override;
    void solve(const MatrixBase& A, const MatrixBase& B, MatrixBase& X) override;
    void eigvals_arnoldi(const MatrixBase& a, VectorBase& out, int k, int m, double tol) override;
    void eigh(const MatrixBase& in, VectorBase& eigenvalues, MatrixBase& eigenvectors, char uplo) override;
    void eigvalsh(const MatrixBase& in, VectorBase& eigenvalues, char uplo) override;
    void eig(const MatrixBase& in, VectorBase& eigenvalues, MatrixBase& eigenvectors) override;
    void eigvals(const MatrixBase& in, VectorBase& eigenvalues) override;

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

    // Smart Dispatch
    double gpu_speedup_factor_ = 1.0;
    bool benchmark_done_ = false;
    bool hardware_profile_valid_ = false;
    double cpu_sgemm_gflops_ = 0.0;
    double cpu_dgemm_gflops_ = 0.0;
    double gpu_dispatch_latency_seconds_ = 0.0002;
    BackendPreference backend_preference_ = BackendPreference::Auto;
    HardwareProfile hardware_profile_;

    bool run_benchmark(bool force);
    bool should_use_gpu(double ops, double bytes, DataType dtype) const;
    double estimate_gpu_time(double ops, double bytes, DataType dtype) const;
    double estimate_cpu_time(double ops, DataType dtype) const;
    bool load_cached_profile(const HardwareProfile& device_profile);
    void apply_dynamic_pinning_budget();

    // Helper to decide device
    ComputeDevice* select_device(uint64_t n_elements) const;
    ComputeDevice* select_device_for_matrix(const MatrixBase& m) const;
};

} // namespace pycauset
