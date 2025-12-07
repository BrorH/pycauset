#include "CpuDevice.hpp"
#include "MatrixOperations.hpp"
#include "ParallelUtils.hpp"
#include "TriangularMatrix.hpp"
#include "DenseMatrix.hpp"
#include "DiagonalMatrix.hpp"
#include "IdentityMatrix.hpp"
#include <stdexcept>

namespace pycauset {

// Helper template for executing binary operations (Moved from MatrixOperations.cpp)
template <typename T, typename Op>
void execute_binary_op_cpu(const MatrixBase& a, const MatrixBase& b, MatrixBase& result, Op op) {
    uint64_t n = a.size();
    
    // Try to cast to TriangularMatrix<T>
    auto* tri_res = dynamic_cast<TriangularMatrix<T>*>(&result);
    if (tri_res) {
        ParallelFor(0, n, [&](size_t i) {
            for (uint64_t j = i + 1; j < n; ++j) {
                T val = op(static_cast<T>(a.get_element_as_double(i, j)), 
                           static_cast<T>(b.get_element_as_double(i, j)));
                if (val != static_cast<T>(0)) {
                    tri_res->set(i, j, val);
                }
            }
        });
        return;
    }

    // Try to cast to DiagonalMatrix<T>
    auto* diag_res = dynamic_cast<DiagonalMatrix<T>*>(&result);
    if (diag_res) {
        if (result.get_matrix_type() == MatrixType::IDENTITY) {
             if (n > 0) {
                 T val = op(static_cast<T>(a.get_element_as_double(0, 0)), 
                            static_cast<T>(b.get_element_as_double(0, 0)));
                 result.set_scalar(static_cast<double>(val));
             }
             return;
        }

        ParallelFor(0, n, [&](size_t i) {
            T val = op(static_cast<T>(a.get_element_as_double(i, i)), 
                       static_cast<T>(b.get_element_as_double(i, i)));
            diag_res->set(i, i, val);
        });
        return;
    }

    // Try to cast to DenseMatrix<T>
    auto* dense_res = dynamic_cast<DenseMatrix<T>*>(&result);
    if (dense_res) {
        ParallelFor(0, n, [&](size_t i) {
            for (uint64_t j = 0; j < n; ++j) {
                T val = op(static_cast<T>(a.get_element_as_double(i, j)), 
                           static_cast<T>(b.get_element_as_double(i, j)));
                dense_res->set(i, j, val);
            }
        });
        return;
    }

    throw std::runtime_error("Unknown result matrix type in execute_binary_op_cpu");
}

void CpuDevice::matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    solver_.matmul(a, b, result);
}

void CpuDevice::inverse(const MatrixBase& in, MatrixBase& out) {
    solver_.inverse(in, out);
}

void CpuDevice::eigvals(const MatrixBase& matrix, ComplexVector& result) {
    solver_.eigvals(matrix, result);
}

void CpuDevice::batch_gemv(const MatrixBase& A, const double* x_data, double* y_data, size_t b) {
    solver_.batch_gemv(A, x_data, y_data, b);
}

void CpuDevice::add(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    solver_.add(a, b, result);
}

void CpuDevice::subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) {
    solver_.subtract(a, b, result);
}

void CpuDevice::multiply_scalar(const MatrixBase& a, double scalar, MatrixBase& result) {
    solver_.multiply_scalar(a, scalar, result);
}

} // namespace pycauset
