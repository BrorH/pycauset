#pragma once

#include "pycauset/compute/ComputeDevice.hpp"
#include "pycauset/compute/cpu/CpuSolver.hpp"
#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/vector/VectorBase.hpp"
#include <complex>

namespace pycauset {

class CpuDevice : public ComputeDevice {
public:
    void matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) override;
    void inverse(const MatrixBase& in, MatrixBase& out) override;
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

    std::string name() const override { return "CPU"; }
    bool is_gpu() const override { return false; }

private:
    CpuSolver solver_;
};

} // namespace pycauset
