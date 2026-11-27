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
// Does the call exceed the RAM budget? Operands and result at their element size (only
// float32/float64/int32/int64 stream, so 4 or 8 bytes).
bool out_of_core(const MatrixBase& a, const MatrixBase& b, const MatrixBase& c) {
    const DataType dt = c.get_data_type();
    const size_t elem = (dt == DataType::FLOAT32 || dt == DataType::INT32) ? 4 : 8;
    const size_t required = (a.size() + b.size() + c.size()) * elem;
    auto& governor = core::MemoryGovernor::instance();
    return !governor.can_fit_in_ram(required);
}
}  // namespace

void CpuDevice::matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    if (can_stream_matmul(a, b, result) && out_of_core(a, b, result)) {
        CpuComputeWorker worker(solver_);
        StreamingManager(worker).matmul(a, b, result);
    } else {
        solver_.matmul(a, b, result);
    }
}

void CpuDevice::cholesky(const MatrixBase& in, MatrixBase& out) {
    solver_.cholesky(in, out);
}

void CpuDevice::matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, VectorBase& result) {
    solver_.matrix_vector_multiply(m, v, result);
}

void CpuDevice::add(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    if (can_stream_elementwise(a, b, result) && out_of_core(a, b, result)) {
        CpuComputeWorker worker(solver_);
        StreamingManager(worker).elementwise(a, b, result, ComputeWorker::ElementwiseOp::ADD);
    } else {
        solver_.add(a, b, result);
    }
}

void CpuDevice::subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    if (can_stream_elementwise(a, b, result) && out_of_core(a, b, result)) {
        CpuComputeWorker worker(solver_);
        StreamingManager(worker).elementwise(a, b, result, ComputeWorker::ElementwiseOp::SUBTRACT);
    } else {
        solver_.subtract(a, b, result);
    }
}

void CpuDevice::elementwise_multiply(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    if (can_stream_elementwise(a, b, result) && out_of_core(a, b, result)) {
        CpuComputeWorker worker(solver_);
        StreamingManager(worker).elementwise(a, b, result, ComputeWorker::ElementwiseOp::MULTIPLY);
    } else {
        solver_.elementwise_multiply(a, b, result);
    }
}

void CpuDevice::elementwise_divide(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    if (can_stream_elementwise(a, b, result) && out_of_core(a, b, result)) {
        CpuComputeWorker worker(solver_);
        StreamingManager(worker).elementwise(a, b, result, ComputeWorker::ElementwiseOp::DIVIDE);
    } else {
        solver_.elementwise_divide(a, b, result);
    }
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

void CpuDevice::lstsq(const MatrixBase& A, const MatrixBase& B, MatrixBase& X) {
    solver_.lstsq(A, B, X);
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
