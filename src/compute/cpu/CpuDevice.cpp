#include "pycauset/compute/cpu/CpuDevice.hpp"
#include "pycauset/compute/cpu/CpuComputeWorker.hpp"
#include "pycauset/compute/StreamingManager.hpp"
#include "pycauset/core/MemoryGovernor.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/matrix/IdentityMatrix.hpp"
#include <stdexcept>

namespace pycauset {

void CpuDevice::matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    auto& governor = core::MemoryGovernor::instance();
    // Check if we should stream
    
    // Calculate required bytes (approx)
    DataType dt = result.get_data_type();
    size_t bytes_per_elem = 8; // Default double
    if (dt == DataType::FLOAT32) bytes_per_elem = 4;
    else if (dt == DataType::BIT) bytes_per_elem = 1; // Approx
    
    size_t required = (a.size() + b.size() + result.size()) * bytes_per_elem;
    
    if (!governor.can_fit_in_ram(required) || !governor.should_use_direct_path(required)) {
        // Use Streaming Manager
        CpuComputeWorker worker(solver_);
        StreamingManager manager(worker);
        manager.matmul(a, b, result);
    } else {
        // Direct Path
        solver_.matmul(a, b, result);
    }
}

void CpuDevice::inverse(const MatrixBase& in, MatrixBase& out) {
    solver_.inverse(in, out);
}

void CpuDevice::cholesky(const MatrixBase& in, MatrixBase& out) {
    solver_.cholesky(in, out);
}

void CpuDevice::batch_gemv(const MatrixBase& A, const double* x_data, double* y_data, size_t b) {
    solver_.batch_gemv(A, x_data, y_data, b);
}

void CpuDevice::matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, VectorBase& result) {
    solver_.matrix_vector_multiply(m, v, result);
}

void CpuDevice::vector_matrix_multiply(const VectorBase& v, const MatrixBase& m, VectorBase& result) {
    solver_.vector_matrix_multiply(v, m, result);
}

void CpuDevice::outer_product(const VectorBase& a, const VectorBase& b, MatrixBase& result) {
    solver_.outer_product(a, b, result);
}

void CpuDevice::add(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    auto& governor = core::MemoryGovernor::instance();
    size_t required = (a.size() + b.size() + result.size()) * 8; 
    if (!governor.can_fit_in_ram(required) || !governor.should_use_direct_path(required)) {
         CpuComputeWorker worker(solver_);
         StreamingManager manager(worker);
         manager.elementwise(a, b, result, ComputeWorker::ElementwiseOp::ADD);
    } else {
        solver_.add(a, b, result);
    }
}

void CpuDevice::subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    auto& governor = core::MemoryGovernor::instance();
    size_t required = (a.size() + b.size() + result.size()) * 8;
    if (!governor.can_fit_in_ram(required) || !governor.should_use_direct_path(required)) {
         CpuComputeWorker worker(solver_);
         StreamingManager manager(worker);
         manager.elementwise(a, b, result, ComputeWorker::ElementwiseOp::SUBTRACT);
    } else {
        solver_.subtract(a, b, result);
    }
}

void CpuDevice::elementwise_multiply(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    auto& governor = core::MemoryGovernor::instance();
    size_t required = (a.size() + b.size() + result.size()) * 8;
    if (!governor.can_fit_in_ram(required) || !governor.should_use_direct_path(required)) {
         CpuComputeWorker worker(solver_);
         StreamingManager manager(worker);
         manager.elementwise(a, b, result, ComputeWorker::ElementwiseOp::MULTIPLY);
    } else {
        solver_.elementwise_multiply(a, b, result);
    }
}

void CpuDevice::elementwise_divide(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    auto& governor = core::MemoryGovernor::instance();
    size_t required = (a.size() + b.size() + result.size()) * 8;
    if (!governor.can_fit_in_ram(required) || !governor.should_use_direct_path(required)) {
         CpuComputeWorker worker(solver_);
         StreamingManager manager(worker);
         manager.elementwise(a, b, result, ComputeWorker::ElementwiseOp::DIVIDE);
    } else {
        solver_.elementwise_divide(a, b, result);
    }
}

void CpuDevice::multiply_scalar(const MatrixBase& a, double scalar, MatrixBase& result) {
    solver_.multiply_scalar(a, scalar, result);
}

double CpuDevice::dot(const VectorBase& a, const VectorBase& b) {
    return solver_.dot(a, b);
}

std::complex<double> CpuDevice::dot_complex(const VectorBase& a, const VectorBase& b) {
    return solver_.dot_complex(a, b);
}

std::complex<double> CpuDevice::sum(const VectorBase& v) {
    return solver_.sum(v);
}

double CpuDevice::l2_norm(const VectorBase& v) {
    return solver_.l2_norm(v);
}

void CpuDevice::add_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) {
    solver_.add_vector(a, b, result);
}

void CpuDevice::subtract_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) {
    solver_.subtract_vector(a, b, result);
}

void CpuDevice::scalar_multiply_vector(const VectorBase& a, double scalar, VectorBase& result) {
    solver_.scalar_multiply_vector(a, scalar, result);
}

void CpuDevice::scalar_multiply_vector_complex(const VectorBase& a, std::complex<double> scalar, VectorBase& result) {
    solver_.scalar_multiply_vector_complex(a, scalar, result);
}

void CpuDevice::scalar_add_vector(const VectorBase& a, double scalar, VectorBase& result) {
    solver_.scalar_add_vector(a, scalar, result);
}

void CpuDevice::cross_product(const VectorBase& a, const VectorBase& b, VectorBase& result) {
    solver_.cross_product(a, b, result);
}

std::unique_ptr<TriangularMatrix<double>> CpuDevice::compute_k_matrix(
    const TriangularMatrix<bool>& C,
    double a,
    const std::string& output_path,
    int num_threads
) {
    return solver_.compute_k_matrix(C, a, output_path, num_threads);
}

double CpuDevice::frobenius_norm(const MatrixBase& m) {
    return solver_.frobenius_norm(m);
}

std::complex<double> CpuDevice::sum(const MatrixBase& m) {
    return solver_.sum(m);
}

double CpuDevice::trace(const MatrixBase& m) {
    return solver_.trace(m);
}

double CpuDevice::determinant(const MatrixBase& m) {
    return solver_.determinant(m);
}

void CpuDevice::qr(const MatrixBase& in, MatrixBase& Q, MatrixBase& R) {
    solver_.qr(in, Q, R);
}

void CpuDevice::lu(const MatrixBase& in, MatrixBase& P, MatrixBase& L, MatrixBase& U) {
    solver_.lu(in, P, L, U);
}

void CpuDevice::svd(const MatrixBase& in, MatrixBase& U, VectorBase& S, MatrixBase& VT) {
    solver_.svd(in, U, S, VT);
}

void CpuDevice::solve(const MatrixBase& A, const MatrixBase& B, MatrixBase& X) {
    solver_.solve(A, B, X);
}

void CpuDevice::eigvals_arnoldi(const MatrixBase& a, VectorBase& out, int k, int m, double tol) {
    solver_.eigvals_arnoldi(a, out, k, m, tol);
}

bool CpuDevice::fill_hardware_profile(HardwareProfile& profile, bool run_benchmarks) {
    (void)profile;
    (void)run_benchmarks;
    return false;
}

void CpuDevice::eigh(const MatrixBase& in, VectorBase& eigenvalues, MatrixBase& eigenvectors, char uplo) {
    solver_.eigh(in, eigenvalues, eigenvectors, uplo);
}

void CpuDevice::eigvalsh(const MatrixBase& in, VectorBase& eigenvalues, char uplo) {
    solver_.eigvalsh(in, eigenvalues, uplo);
}

void CpuDevice::eig(const MatrixBase& in, VectorBase& eigenvalues, MatrixBase& eigenvectors) {
    solver_.eig(in, eigenvalues, eigenvectors);
}

void CpuDevice::eigvals(const MatrixBase& in, VectorBase& eigenvalues) {
    solver_.eigvals(in, eigenvalues);
}

} // namespace pycauset
