#pragma once

#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/vector/ComplexVector.hpp"
#include <memory>

namespace pycauset {

class CpuEigenSolver {
public:
    // Main entry point for out-of-core eigenvalue solving
    static void eigvals_outofcore(const MatrixBase& matrix, ComplexVector& result);

private:
    // Block Jacobi Algorithm Implementation
    static void block_jacobi(const MatrixBase& matrix, ComplexVector& result);
    
    // Helper to solve a 2x2 block (or small block)
    static void solve_2x2_block(double& aii, double& ajj, double& aij, double& c, double& s);
    static void solve_2x2_block_complex(std::complex<double>& aii, std::complex<double>& ajj, std::complex<double>& aij, double& c, std::complex<double>& s);
};

} // namespace pycauset
