#include <gtest/gtest.h>
#include "DenseMatrix.hpp"
#include "DenseBitMatrix.hpp"
#include "TriangularMatrix.hpp"
#include "DiagonalMatrix.hpp"
#include "ComputeContext.hpp"
#include "AutoSolver.hpp"
#include <memory>
#include <cmath>

using namespace pycauset;

class SolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure we use AutoSolver (default)
        // We can force CPU or GPU if needed, but AutoSolver should handle it.
    }
};

TEST_F(SolverTest, DenseMatrixMultiply) {
    uint64_t n = 10;
    auto A = std::make_unique<DenseMatrix<double>>(n);
    auto B = std::make_unique<DenseMatrix<double>>(n);
    
    // A = Identity, B = Random
    for(uint64_t i=0; i<n; ++i) A->set(i, i, 1.0);
    for(uint64_t i=0; i<n; ++i) {
        for(uint64_t j=0; j<n; ++j) {
            B->set(i, j, (double)(i+j));
        }
    }
    
    auto C = A->multiply(*B);
    
    for(uint64_t i=0; i<n; ++i) {
        for(uint64_t j=0; j<n; ++j) {
            EXPECT_DOUBLE_EQ(C->get(i, j), (double)(i+j));
        }
    }
}

TEST_F(SolverTest, DenseMatrixInverse) {
    uint64_t n = 2;
    auto A = std::make_unique<DenseMatrix<double>>(n);
    // A = [[4, 7], [2, 6]]
    // Det = 24 - 14 = 10
    // Inv = [[0.6, -0.7], [-0.2, 0.4]]
    A->set(0, 0, 4.0); A->set(0, 1, 7.0);
    A->set(1, 0, 2.0); A->set(1, 1, 6.0);
    
    auto Inv = A->inverse();
    
    EXPECT_NEAR(Inv->get(0, 0), 0.6, 1e-9);
    EXPECT_NEAR(Inv->get(0, 1), -0.7, 1e-9);
    EXPECT_NEAR(Inv->get(1, 0), -0.2, 1e-9);
    EXPECT_NEAR(Inv->get(1, 1), 0.4, 1e-9);
}

TEST_F(SolverTest, BitMatrixMultiply) {
    uint64_t n = 4;
    auto A = std::make_unique<DenseBitMatrix>(n);
    auto B = std::make_unique<DenseBitMatrix>(n);
    
    // A = Identity
    for(uint64_t i=0; i<n; ++i) A->set(i, i, true);
    
    // B = All ones
    for(uint64_t i=0; i<n; ++i) 
        for(uint64_t j=0; j<n; ++j) 
            B->set(i, j, true);
            
    auto C = A->multiply(*B, ""); // Returns DenseMatrix<int32_t>
    
    for(uint64_t i=0; i<n; ++i) {
        for(uint64_t j=0; j<n; ++j) {
            EXPECT_EQ(C->get(i, j), 1); // 1 path of length 1
        }
    }
}

TEST_F(SolverTest, TriangularMatrixMultiply) {
    uint64_t n = 3;
    auto A = std::make_unique<TriangularMatrix<double>>(n, "", false); // Strictly Upper
    auto B = std::make_unique<TriangularMatrix<double>>(n, "", false);
    
    // A = [[0, 1, 2], [0, 0, 3], [0, 0, 0]]
    A->set(0, 1, 1.0); A->set(0, 2, 2.0);
    A->set(1, 2, 3.0);
    
    // B = [[0, 1, 1], [0, 0, 1], [0, 0, 0]]
    B->set(0, 1, 1.0); B->set(0, 2, 1.0);
    B->set(1, 2, 1.0);
    
    // C = A * B
    // c00 = 0
    // c01 = 0*1 + 1*0 + 2*0 = 0
    // c02 = 0*1 + 1*1 + 2*0 = 1
    // c12 = 0*1 + 0*1 + 3*0 = 0
    
    auto C = A->multiply(*B);
    
    EXPECT_DOUBLE_EQ(C->get(0, 2), 1.0);
    EXPECT_DOUBLE_EQ(C->get(0, 1), 0.0);
}

TEST_F(SolverTest, DiagonalMatrixMultiply) {
    // We don't have explicit multiply on DiagonalMatrix, but we can test via ComputeContext if we exposed it.
    // But DiagonalMatrix doesn't have a multiply method in the class.
    // So we can't test it easily unless we cast to MatrixBase and call solver directly, 
    // or if we added multiply to DiagonalMatrix (which we didn't).
    
    // However, we can test that CpuSolver handles it if we call it manually.
    
    uint64_t n = 3;
    auto A = std::make_unique<DiagonalMatrix<double>>(n);
    auto B = std::make_unique<DiagonalMatrix<double>>(n);
    auto C = std::make_unique<DiagonalMatrix<double>>(n);
    
    for(uint64_t i=0; i<n; ++i) A->set(i, i, 2.0);
    for(uint64_t i=0; i<n; ++i) B->set(i, i, 3.0);
    
    ComputeContext::instance().get_device()->matmul(*A, *B, *C);
    
    for(uint64_t i=0; i<n; ++i) {
        EXPECT_DOUBLE_EQ(C->get(i, i), 6.0);
    }
}

TEST_F(SolverTest, DiagonalMatrixInverse) {
    // We can test inverse via CpuSolver
    uint64_t n = 3;
    auto A = std::make_unique<DiagonalMatrix<double>>(n);
    auto Inv = std::make_unique<DiagonalMatrix<double>>(n);
    
    for(uint64_t i=0; i<n; ++i) A->set(i, i, 2.0);
    
    ComputeContext::instance().get_device()->inverse(*A, *Inv);
    
    for(uint64_t i=0; i<n; ++i) {
        EXPECT_DOUBLE_EQ(Inv->get(i, i), 0.5);
    }
}
