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

namespace {
// The StreamingManager copies tiles straight from raw row-major storage of dense
// same-dtype operands, so it is only correct and supported for plain views: no transpose,
// slice offset, lazy scalar or conjugation (F-040). Everything else falls back to the
// direct kernels, which honour view metadata and still work out of core through mmap
// paging (slower), instead of producing wrong values or throwing mid-operation.
bool plain_view(const MatrixBase& m) {
    return !m.is_transposed() && !m.is_subview() && !m.is_conjugated() &&
           m.get_scalar() == std::complex<double>(1.0, 0.0) &&
           m.get_structure() == Structure::DENSE;
}
bool same_dtype_plain(const MatrixBase& a, const MatrixBase& b, const MatrixBase& c) {
    return plain_view(a) && plain_view(b) && plain_view(c) && a.get_data_type() == b.get_data_type() &&
           b.get_data_type() == c.get_data_type();
}
bool can_stream_matmul(const MatrixBase& a, const MatrixBase& b, const MatrixBase& c) {
    const DataType dt = c.get_data_type();
    return same_dtype_plain(a, b, c) && (dt == DataType::FLOAT64 || dt == DataType::FLOAT32);
}
bool can_stream_elementwise(const MatrixBase& a, const MatrixBase& b, const MatrixBase& c) {
    const DataType dt = c.get_data_type();
    return same_dtype_plain(a, b, c) && (dt == DataType::FLOAT64 || dt == DataType::FLOAT32 ||
                                         dt == DataType::INT32 || dt == DataType::INT64);
}
}  // namespace

void CpuDevice::matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    auto& governor = core::MemoryGovernor::instance();
    // Check if we should stream
    
    // Calculate required bytes (approx)
    DataType dt = result.get_data_type();
    size_t bytes_per_elem = 8; // Default double
    if (dt == DataType::FLOAT32) bytes_per_elem = 4;
    else if (dt == DataType::BIT) bytes_per_elem = 1; // Approx
    
    size_t required = (a.size() + b.size() + result.size()) * bytes_per_elem;
    
    if ((!governor.can_fit_in_ram(required) || !governor.should_use_direct_path(required)) &&
        can_stream_matmul(a, b, result)) {
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
    if ((!governor.can_fit_in_ram(required) || !governor.should_use_direct_path(required)) &&
        can_stream_elementwise(a, b, result)) {
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
    if ((!governor.can_fit_in_ram(required) || !governor.should_use_direct_path(required)) &&
        can_stream_elementwise(a, b, result)) {
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
    if ((!governor.can_fit_in_ram(required) || !governor.should_use_direct_path(required)) &&
        can_stream_elementwise(a, b, result)) {
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
    if ((!governor.can_fit_in_ram(required) || !governor.should_use_direct_path(required)) &&
        can_stream_elementwise(a, b, result)) {
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


double CpuDevice::l2_norm(const VectorBase& v) {
    return solver_.l2_norm(v);
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



LuPivots CpuDevice::lu_pivots(const MatrixBase& m) {
    return solver_.lu_pivots(m);
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

void CpuDevice::eigvals_skew(const MatrixBase& a, VectorBase& out, int k) {
    solver_.eigvals_skew(a, out, k);
}

void CpuDevice::eig_skew(const MatrixBase& a, VectorBase& eigenvalues, MatrixBase& eigenvectors, int k) {
    solver_.eig_skew(a, eigenvalues, eigenvectors, k);
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
