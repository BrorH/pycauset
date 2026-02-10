#include "pycauset/compute/cpu/CpuSolver.hpp"
#include "pycauset/compute/cpu/CpuComputeWorker.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/core/MemoryGovernor.hpp" // Ensure governor is linked if needed by DenseMatrix ctor
#include <iostream>
#include <cassert>
#include <cstdlib>

/**
 * @brief Contract Test for R1_CPU Phase 1
 * 
 * Verifies that the CpuComputeWorker correctly delegates to CpuSolver
 * and adheres to the ComputeWorker interface for standard calling conventions.
 */
void test_matmul_contract() {
    pycauset::CpuSolver solver;
    pycauset::CpuComputeWorker worker(solver);

    using namespace pycauset;
    
    std::cout << "[Test] Initializing Matrices..." << std::endl;
    // Create Inputs (2x2)
    // Note: DenseMatrix default ctor might require MemoryMapper/Governor logic. 
    // Assuming we can create small RAM matrices easily (using 0 args or sizes).
    // If DenseMatrix(rows, cols) assumes RAM, this works.
    DenseMatrix<double> A(2, 2);
    DenseMatrix<double> B(2, 2);
    DenseMatrix<double> C(2, 2);

    // Fill A (Identity)
    {
        double* data = A.data();
        data[0] = 1.0; data[1] = 0.0;
        data[2] = 0.0; data[3] = 1.0;
    }
    // Fill B (2,3; 4,5)
    {
        double* data = B.data();
        data[0] = 2.0; data[1] = 3.0;
        data[2] = 4.0; data[3] = 5.0; 
    }

    std::cout << "[Test] Executing matmul_tile..." << std::endl;
    // C = 1.0 * A * B + 0.0 * C
    worker.matmul_tile(A, B, C, 1.0, 0.0);

    // Verify
    double* c_data = C.data();
    std::cout << "[Test] Result: " << c_data[0] << ", " << c_data[1] << ", " << c_data[2] << ", " << c_data[3] << std::endl;

    if (c_data[0] != 2.0 || c_data[1] != 3.0 || 
        c_data[2] != 4.0 || c_data[3] != 5.0) {
        std::cerr << "Contract Test Failed: Matmul result incorrect" << std::endl;
        std::exit(1);
    }
    
    std::cout << "Contract Test Passed: Matmul tile execution successful." << std::endl;
}

int main() {
    try {
        test_matmul_contract();
    } catch (const std::exception& e) {
        std::cerr << "Test Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
