#pragma once

#include "pycauset/matrix/MatrixBase.hpp"
#include <complex>
#include <memory>
#include <string>
#include <vector>

namespace pycauset {
class VectorBase; // Forward declaration
template <typename T>
class TriangularMatrix;

class CpuSolver {
public:
    CpuSolver() = default;
    ~CpuSolver() = default;

    // Core Operations
    void matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);
    void inverse(const MatrixBase& in, MatrixBase& out);
    void batch_gemv(const MatrixBase& A, const double* x_data, double* y_data, size_t b);

    void matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, VectorBase& result);
    void vector_matrix_multiply(const VectorBase& v, const MatrixBase& m, VectorBase& result);
    void outer_product(const VectorBase& a, const VectorBase& b, MatrixBase& result);

    void add(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);
    void subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);
    void elementwise_multiply(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);
    void elementwise_divide(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);
    void multiply_scalar(const MatrixBase& a, double scalar, MatrixBase& result);

    double dot(const VectorBase& a, const VectorBase& b);
    std::complex<double> dot_complex(const VectorBase& a, const VectorBase& b);
    std::complex<double> sum(const VectorBase& v);
    double l2_norm(const VectorBase& v);
    void add_vector(const VectorBase& a, const VectorBase& b, VectorBase& result);
    void subtract_vector(const VectorBase& a, const VectorBase& b, VectorBase& result);
    void scalar_multiply_vector(const VectorBase& a, double scalar, VectorBase& result);
    void scalar_multiply_vector_complex(const VectorBase& a, std::complex<double> scalar, VectorBase& result);
    void scalar_add_vector(const VectorBase& a, double scalar, VectorBase& result);

    void cross_product(const VectorBase& a, const VectorBase& b, VectorBase& result);

    std::unique_ptr<TriangularMatrix<double>> compute_k_matrix(
        const TriangularMatrix<bool>& C,
        double a,
        const std::string& output_path,
        int num_threads);

    double frobenius_norm(const MatrixBase& m);
    std::complex<double> sum(const MatrixBase& m);

    double trace(const MatrixBase& m);
    double determinant(const MatrixBase& m);
    void qr(const MatrixBase& in, MatrixBase& Q, MatrixBase& R);

private:
    void matmul_dense(const MatrixBase& a, const MatrixBase& b, MatrixBase& result);
};

} // namespace pycauset
