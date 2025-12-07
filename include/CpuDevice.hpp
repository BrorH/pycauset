#pragma once

#include "ComputeDevice.hpp"
#include "CpuSolver.hpp"
#include "MatrixBase.hpp"
#include "VectorBase.hpp"

namespace pycauset {

class CpuDevice : public ComputeDevice {
public:
    void matmul(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) override;
    void inverse(const MatrixBase& in, MatrixBase& out) override;
    void eigvals(const MatrixBase& matrix, ComplexVector& result) override;
    void batch_gemv(const MatrixBase& A, const double* x_data, double* y_data, size_t b) override;

    void add(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) override;
    void subtract(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) override;
    void multiply_scalar(const MatrixBase& a, double scalar, MatrixBase& result) override;

    std::string name() const override { return "CPU"; }
    bool is_gpu() const override { return false; }

private:
    CpuSolver solver_;
};

} // namespace pycauset
