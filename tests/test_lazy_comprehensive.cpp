#include <gtest/gtest.h>
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/matrix/MatrixOps.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>

using namespace pycauset;

class LazyComprehensiveTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup if needed
    }

    void TearDown() override {
        if (std::filesystem::exists("test_spill.tmp")) {
            std::filesystem::remove("test_spill.tmp");
        }
    }
};

// 1. Basic Arithmetic
TEST_F(LazyComprehensiveTest, BasicArithmetic) {
    DenseMatrix<double> A(5, 5);
    DenseMatrix<double> B(5, 5);
    A.fill(10.0);
    B.fill(5.0);

    DenseMatrix<double> C(5, 5);
    C = A + B; // 15
    EXPECT_DOUBLE_EQ(C.get(0, 0), 15.0);

    C = A - B; // 5
    EXPECT_DOUBLE_EQ(C.get(0, 0), 5.0);

    C = A * 2.0; // 20
    EXPECT_DOUBLE_EQ(C.get(0, 0), 20.0);

    C = A / 2.0; // 5
    EXPECT_DOUBLE_EQ(C.get(0, 0), 5.0);
}

// 2. Chained Operations (Precedence)
TEST_F(LazyComprehensiveTest, ChainedOperations) {
    DenseMatrix<double> A(2, 2);
    A.fill(2.0);
    
    DenseMatrix<double> B(2, 2);
    B.fill(3.0);

    DenseMatrix<double> C(2, 2);
    
    // (A + B) * 2 = (2+3)*2 = 10
    C = (A + B) * 2.0;
    EXPECT_DOUBLE_EQ(C.get(0, 0), 10.0);

    // A + B * 2 = 2 + (3*2) = 8
    C = A + B * 2.0;
    EXPECT_DOUBLE_EQ(C.get(0, 0), 8.0);
}

// 3. Unary Operations
TEST_F(LazyComprehensiveTest, UnaryOperations) {
    DenseMatrix<double> A(2, 2);
    A.fill(1.0);

    DenseMatrix<double> C(2, 2);
    C = -A;
    EXPECT_DOUBLE_EQ(C.get(0, 0), -1.0);

    // sin(0) = 0
    A.fill(0.0);
    C = pycauset::sin(A);
    EXPECT_DOUBLE_EQ(C.get(0, 0), 0.0);

    // exp(0) = 1
    C = pycauset::exp(A);
    EXPECT_DOUBLE_EQ(C.get(0, 0), 1.0);
}

// 4. Aliasing Safety
TEST_F(LazyComprehensiveTest, AliasingDetection) {
    DenseMatrix<double> A(2, 2);
    A.fill(1.0);

    // A = A + A should work now (via temporary fallback)
    A = A + A;
    EXPECT_DOUBLE_EQ(A.get(0, 0), 2.0);
    
    // A += A should also work
    A += A;
    EXPECT_DOUBLE_EQ(A.get(0, 0), 4.0);
}

// 5. Dimension Mismatch
TEST_F(LazyComprehensiveTest, DimensionMismatch) {
    DenseMatrix<double> A(2, 2);
    DenseMatrix<double> B(3, 3);
    DenseMatrix<double> C(2, 2);

    // Binary op mismatch
    EXPECT_THROW({
        auto expr = A + B;
    }, std::runtime_error);

    // Assignment mismatch
    DenseMatrix<double> D(3, 3);
    D.fill(1.0);
    EXPECT_THROW({
        C = D; // 2x2 = 3x3
    }, std::runtime_error);
}

// 6. Complex Expressions
TEST_F(LazyComprehensiveTest, ComplexExpression) {
    DenseMatrix<double> A(2, 2); A.fill(1.0);
    DenseMatrix<double> B(2, 2); B.fill(2.0);
    DenseMatrix<double> C(2, 2); C.fill(3.0);
    DenseMatrix<double> D(2, 2);

    // D = (A + B) * C - A
    // (1+2)*3 - 1 = 9 - 1 = 8
    D = (A + B) * 3.0 - A; 
    EXPECT_DOUBLE_EQ(D.get(0, 0), 8.0);
}

// 7. Large Matrix (Performance/Correctness check)
TEST_F(LazyComprehensiveTest, LargeMatrix) {
    // Not huge, but enough to verify loops work correctly
    uint64_t N = 100;
    DenseMatrix<double> A(N, N);
    DenseMatrix<double> B(N, N);
    
    for(uint64_t i=0; i<N; ++i) {
        for(uint64_t j=0; j<N; ++j) {
            A.set(i, j, (double)i);
            B.set(i, j, (double)j);
        }
    }

    DenseMatrix<double> C(N, N);
    C = A + B;

    for(uint64_t i=0; i<N; ++i) {
        for(uint64_t j=0; j<N; ++j) {
            EXPECT_DOUBLE_EQ(C.get(i, j), (double)(i + j));
        }
    }
}

// 7. Eager Matrix Multiplication
TEST_F(LazyComprehensiveTest, EagerMatMul) {
    DenseMatrix<double> A(2, 3);
    DenseMatrix<double> B(3, 2);
    
    // A = [[1, 2, 3], [4, 5, 6]]
    A.set_element_as_double(0, 0, 1.0); A.set_element_as_double(0, 1, 2.0); A.set_element_as_double(0, 2, 3.0);
    A.set_element_as_double(1, 0, 4.0); A.set_element_as_double(1, 1, 5.0); A.set_element_as_double(1, 2, 6.0);

    // B = [[1, 0], [0, 1], [1, 1]]
    B.set_element_as_double(0, 0, 1.0); B.set_element_as_double(0, 1, 0.0);
    B.set_element_as_double(1, 0, 0.0); B.set_element_as_double(1, 1, 1.0);
    B.set_element_as_double(2, 0, 1.0); B.set_element_as_double(2, 1, 1.0);

    // C = A * B
    // [[1*1 + 2*0 + 3*1, 1*0 + 2*1 + 3*1],
    //  [4*1 + 5*0 + 6*1, 4*0 + 5*1 + 6*1]]
    // = [[4, 5], [10, 11]]
    
    auto C_ptr = A * B;
    MatrixBase& C = *C_ptr;

    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 2);
    EXPECT_DOUBLE_EQ(C.get_element_as_double(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(C.get_element_as_double(0, 1), 5.0);
    EXPECT_DOUBLE_EQ(C.get_element_as_double(1, 0), 10.0);
    EXPECT_DOUBLE_EQ(C.get_element_as_double(1, 1), 11.0);
}

// 8. Spill to Disk
TEST_F(LazyComprehensiveTest, SpillToDisk) {
    DenseMatrix<double> A(10, 10);
    A.fill(3.14);

    std::string filename = "test_spill.tmp";
    if (std::filesystem::exists(filename)) {
        std::filesystem::remove(filename);
    }

    // Spill
    A.spill_to_disk(filename);

    // Verify file exists
    EXPECT_TRUE(std::filesystem::exists(filename));
    EXPECT_GT(std::filesystem::file_size(filename), 0);

    // Verify data is still accessible
    EXPECT_DOUBLE_EQ(A.get_element_as_double(0, 0), 3.14);
    EXPECT_DOUBLE_EQ(A.get_element_as_double(9, 9), 3.14);

    // Modify and verify persistence (optional, but good check)
    A.set_element_as_double(0, 0, 2.71);
    EXPECT_DOUBLE_EQ(A.get_element_as_double(0, 0), 2.71);
}
