#include <gtest/gtest.h>
#include "pycauset/matrix/TriangularBitMatrix.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/IdentityMatrix.hpp"
#include "pycauset/core/StorageUtils.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include <filesystem>
#include <fstream>
#include <cmath>

using namespace pycauset;

// Aliases for convenience
using FloatMatrix = DenseMatrix<double>;
using IntegerMatrix = DenseMatrix<int32_t>;
using TriangularFloatMatrix = TriangularMatrix<double>;
using TriangularIntegerMatrix = TriangularMatrix<int32_t>;

class MatrixTest : public ::testing::Test {
protected:
    std::string testFile = "test_matrix.bin";
    std::string resultFile = "result_matrix.bin";
    std::string auxFile = "aux_matrix.bin";

    void SetUp() override {
        cleanup();
    }

    void TearDown() override {
        cleanup();
    }

    void cleanup() {
        if (std::filesystem::exists(testFile)) std::filesystem::remove(testFile);
        if (std::filesystem::exists(resultFile)) std::filesystem::remove(resultFile);
        if (std::filesystem::exists(auxFile)) std::filesystem::remove(auxFile);
    }
};

// --- TriangularBitMatrix Tests ---

TEST_F(MatrixTest, TBM_Initialization) {
    TriangularBitMatrix mat(100, testFile);
    EXPECT_EQ(mat.size(), 100);
}

TEST_F(MatrixTest, TBM_SetAndGet) {
    int N = 10;
    TriangularBitMatrix mat(N, testFile);
    
    mat.set(0, 1, true);
    mat.set(1, 2, true);
    mat.set(0, 9, true);

    EXPECT_TRUE(mat.get(0, 1));
    EXPECT_TRUE(mat.get(1, 2));
    EXPECT_TRUE(mat.get(0, 9));
    
    EXPECT_FALSE(mat.get(0, 2));
    EXPECT_FALSE(mat.get(1, 3));
    
    // Test overwriting
    mat.set(0, 1, false);
    EXPECT_FALSE(mat.get(0, 1));
}

TEST_F(MatrixTest, TBM_MultiplicationSmall) {
    // A: 0->1, 1->2. (Path 0->1->2)
    // B: Same.
    // A*B should have (0, 2) = 1.
    
    TriangularBitMatrix mat(3, testFile);
    mat.set(0, 1, true);
    mat.set(1, 2, true);
    
    auto res = mat.multiply(mat, resultFile);
    
    // Result is TriangularIntegerMatrix
    EXPECT_EQ(res->get(0, 2), 1);
    EXPECT_EQ(res->get(0, 1), 0); // No path of length 2
}

TEST_F(MatrixTest, TBM_MultiplicationPaths) {
    // 0->1, 0->2, 1->3, 2->3
    // Paths from 0 to 3: 0->1->3 and 0->2->3. Total 2.
    
    TriangularBitMatrix mat(4, testFile);
    mat.set(0, 1, true);
    mat.set(0, 2, true);
    mat.set(1, 3, true);
    mat.set(2, 3, true);
    
    auto res = mat.multiply(mat, resultFile);
    
    EXPECT_EQ(res->get(0, 3), 2);
}

TEST_F(MatrixTest, TBM_UpperTriangularConstraint) {
    TriangularBitMatrix mat(10, testFile);
    EXPECT_THROW(mat.set(1, 1, true), std::invalid_argument);
    EXPECT_THROW(mat.set(2, 1, true), std::invalid_argument);
}

// --- DenseMatrix<double> (FloatMatrix) Tests ---

TEST_F(MatrixTest, FM_Initialization) {
    FloatMatrix mat(50, testFile);
    EXPECT_EQ(mat.size(), 50);
    for(int i=0; i<50; ++i)
        for(int j=0; j<50; ++j)
            EXPECT_EQ(mat.get(i, j), 0.0);
}

TEST_F(MatrixTest, FM_SetAndGet) {
    FloatMatrix mat(5, testFile);
    mat.set(0, 0, 1.5);
    mat.set(4, 4, -2.0);
    mat.set(2, 3, 3.14);

    EXPECT_DOUBLE_EQ(mat.get(0, 0), 1.5);
    EXPECT_DOUBLE_EQ(mat.get(4, 4), -2.0);
    EXPECT_DOUBLE_EQ(mat.get(2, 3), 3.14);
}

TEST_F(MatrixTest, FM_Multiplication) {
    // Identity * Identity = Identity
    std::string fileI = "test_I.bin";
    if (std::filesystem::exists(fileI)) std::filesystem::remove(fileI);
    
    {
        FloatMatrix I(3, fileI);
        I.set(0, 0, 1.0); I.set(1, 1, 1.0); I.set(2, 2, 1.0);

        auto res = I.multiply(I, resultFile);
        EXPECT_DOUBLE_EQ(res->get(0, 0), 1.0);
        EXPECT_DOUBLE_EQ(res->get(1, 1), 1.0);
        EXPECT_DOUBLE_EQ(res->get(2, 2), 1.0);
        EXPECT_DOUBLE_EQ(res->get(0, 1), 0.0);
    }
    
    if (std::filesystem::exists(fileI)) std::filesystem::remove(fileI);
    if (std::filesystem::exists(resultFile)) std::filesystem::remove(resultFile);

    // A = [[1, 2], [3, 4]]
    // B = [[2, 0], [1, 2]]
    // A*B = [[4, 4], [10, 8]]
    std::string fileA = "test_A.bin";
    if (std::filesystem::exists(fileA)) std::filesystem::remove(fileA);
    
    {
        FloatMatrix A(2, fileA);
        A.set(0, 0, 1.0); A.set(0, 1, 2.0);
        A.set(1, 0, 3.0); A.set(1, 1, 4.0);

        FloatMatrix B(2, auxFile);
        B.set(0, 0, 2.0); B.set(0, 1, 0.0);
        B.set(1, 0, 1.0); B.set(1, 1, 2.0);

        auto C = A.multiply(B, resultFile);
        EXPECT_DOUBLE_EQ(C->get(0, 0), 4.0);
        EXPECT_DOUBLE_EQ(C->get(0, 1), 4.0);
        EXPECT_DOUBLE_EQ(C->get(1, 0), 10.0);
        EXPECT_DOUBLE_EQ(C->get(1, 1), 8.0);
    }
    
    if (std::filesystem::exists(fileA)) std::filesystem::remove(fileA);
}

TEST_F(MatrixTest, FM_Inversion) {
    // A = [[4, 7], [2, 6]]
    // Det = 24 - 14 = 10
    // Inv = 1/10 * [[6, -7], [-2, 4]] = [[0.6, -0.7], [-0.2, 0.4]]
    
    FloatMatrix A(2, testFile);
    A.set(0, 0, 4.0); A.set(0, 1, 7.0);
    A.set(1, 0, 2.0); A.set(1, 1, 6.0);

    auto Inv = A.inverse(resultFile);
    
    EXPECT_NEAR(Inv->get(0, 0), 0.6, 1e-9);
    EXPECT_NEAR(Inv->get(0, 1), -0.7, 1e-9);
    EXPECT_NEAR(Inv->get(1, 0), -0.2, 1e-9);
    EXPECT_NEAR(Inv->get(1, 1), 0.4, 1e-9);
}

// --- DenseMatrix<int32_t> (IntegerMatrix) Tests ---

TEST_F(MatrixTest, IM_SetAndGet) {
    IntegerMatrix mat(5, testFile);
    mat.set(1, 1, 42);
    mat.set(3, 2, -10);
    
    EXPECT_EQ(mat.get(1, 1), 42);
    EXPECT_EQ(mat.get(3, 2), -10);
}

TEST_F(MatrixTest, IM_Multiplication) {
    // A = [[1, 1], [1, 1]]
    // A*A = [[2, 2], [2, 2]]
    IntegerMatrix A(2, testFile);
    A.set(0, 0, 1); A.set(0, 1, 1);
    A.set(1, 0, 1); A.set(1, 1, 1);
    
    auto res = A.multiply(A, resultFile);
    EXPECT_EQ(res->get(0, 0), 2);
    EXPECT_EQ(res->get(1, 1), 2);
}

// --- TriangularMatrix<double> (TriangularFloatMatrix) Tests ---

TEST_F(MatrixTest, TFM_SetAndGet) {
    TriangularFloatMatrix mat(5, testFile);
    mat.set(0, 1, 1.5);
    
    EXPECT_DOUBLE_EQ(mat.get(0, 1), 1.5);
    EXPECT_DOUBLE_EQ(mat.get(0, 0), 0.0); // Diagonal is 0
    EXPECT_DOUBLE_EQ(mat.get(1, 0), 0.0); // Lower is 0
    
    EXPECT_THROW(mat.set(1, 0, 1.0), std::invalid_argument);
}

// --- Mixed Operations & Persistence ---

TEST_F(MatrixTest, Persistence) {
    {
        TriangularBitMatrix mat(10, testFile);
        mat.set(0, 5, true);
        mat.close();
    }
    
    // Re-open
    // In the new system, the file contains raw data only.
    uint64_t file_size = std::filesystem::file_size(testFile);
    auto mapper = std::make_unique<MemoryMapper>(testFile, file_size, 0, false);
    TriangularBitMatrix loaded(10, std::move(mapper));
    
    EXPECT_TRUE(loaded.get(0, 5));
    EXPECT_FALSE(loaded.get(0, 4));
}

TEST_F(MatrixTest, ComputeKMatrix) {
    // Simple case: C = [[0, 1], [0, 0]] (1 node 0->1)
    // K = C(aI + C)^-1
    // a = 1.0
    // (I + C) = [[1, 1], [0, 1]]
    // (I + C)^-1 = [[1, -1], [0, 1]]
    // C * (I+C)^-1 = [[0, 1], [0, 0]] * [[1, -1], [0, 1]] = [[0, 1], [0, 0]]
    
    TriangularBitMatrix C(2, testFile);
    C.set(0, 1, true);
    
    compute_k_matrix(C, 1.0, resultFile, 1);
    
    // Load result K (TriangularFloatMatrix)
    uint64_t file_size = std::filesystem::file_size(resultFile);
    auto mapper = std::make_unique<MemoryMapper>(resultFile, file_size, 0, false);
    TriangularFloatMatrix K(2, std::move(mapper));
    
    EXPECT_NEAR(K.get(0, 1), 1.0, 1e-9);
}

// --- IdentityMatrix Tests ---

TEST_F(MatrixTest, Identity_SetAndGet) {
    IdentityMatrix<double> mat(5, testFile);
    EXPECT_DOUBLE_EQ(mat.get_element_as_double(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(mat.get_element_as_double(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(mat.get_element_as_double(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(mat.get_element_as_double(1, 0), 0.0);
    
    // Test scalar
    mat.set_scalar(2.0);
    EXPECT_DOUBLE_EQ(mat.get_element_as_double(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(mat.get_element_as_double(1, 1), 2.0);
}

TEST_F(MatrixTest, Identity_Multiplication) {
    IdentityMatrix<double> I1(3, testFile);
    IdentityMatrix<double> I2(3, auxFile);
    
    I1.set_scalar(2.0);
    I2.set_scalar(3.0);
    
    auto res = I1.multiply(I2, resultFile);
    EXPECT_DOUBLE_EQ(res->get_element_as_double(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(res->get_element_as_double(0, 1), 0.0);
    
    // Check if result is IdentityMatrix
    EXPECT_NE(dynamic_cast<IdentityMatrix<double>*>(res.get()), nullptr);
}

TEST_F(MatrixTest, Identity_Addition) {
    IdentityMatrix<double> I1(3, testFile);
    IdentityMatrix<double> I2(3, auxFile);
    
    I1.set_scalar(2.0);
    I2.set_scalar(3.0);
    
    auto res = pycauset::add(I1, I2, resultFile);
    EXPECT_DOUBLE_EQ(res->get_element_as_double(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(res->get_element_as_double(0, 1), 0.0);
    
    // Check if result is IdentityMatrix
    EXPECT_NE(dynamic_cast<IdentityMatrix<double>*>(res.get()), nullptr);
}

