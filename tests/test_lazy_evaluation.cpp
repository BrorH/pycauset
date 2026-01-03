#include <gtest/gtest.h>
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/matrix/MatrixOps.hpp"

using namespace pycauset;

TEST(LazyEvaluationTest, Addition) {
    DenseMatrix<double> A(10, 10);
    DenseMatrix<double> B(10, 10);
    
    // Initialize
    for(uint64_t i=0; i<10; ++i) {
        for(uint64_t j=0; j<10; ++j) {
            A.set(i, j, 1.0);
            B.set(i, j, 2.0);
        }
    }

    // Lazy addition
    auto expr = A + B;
    
    // Verify expression type (compile-time check mostly, but we can check rows/cols)
    EXPECT_EQ(expr.rows(), 10);
    EXPECT_EQ(expr.cols(), 10);
    
    // Evaluate into C
    DenseMatrix<double> C(10, 10);
    C = expr;
    
    for(uint64_t i=0; i<10; ++i) {
        for(uint64_t j=0; j<10; ++j) {
            EXPECT_DOUBLE_EQ(C.get(i, j), 3.0);
        }
    }
}

TEST(LazyEvaluationTest, ChainedOperations) {
    DenseMatrix<double> A(5, 5);
    A.set(0, 0, 10.0);
    
    auto expr = (A + A) * 2.0;
    
    DenseMatrix<double> C(5, 5);
    C = expr;
    
    EXPECT_DOUBLE_EQ(C.get(0, 0), 40.0);
}
