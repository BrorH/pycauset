#include <gtest/gtest.h>
#include "CausalMatrix.hpp"
#include "IntegerMatrix.hpp"
#include <filesystem>

class CausalMatrixTest : public ::testing::Test {
protected:
    std::string testFile = "test_matrix.bin";
    std::string resultFile = "result_matrix.bin";

    void SetUp() override {
        // Clean up before test
        if (std::filesystem::exists(testFile)) std::filesystem::remove(testFile);
        if (std::filesystem::exists(resultFile)) std::filesystem::remove(resultFile);
    }

    void TearDown() override {
        // Clean up after test
        if (std::filesystem::exists(testFile)) std::filesystem::remove(testFile);
        if (std::filesystem::exists(resultFile)) std::filesystem::remove(resultFile);
    }
};

TEST_F(CausalMatrixTest, Initialization) {
    CausalMatrix mat(100, testFile);
    EXPECT_EQ(mat.size(), 100);
}

TEST_F(CausalMatrixTest, SetAndGet) {
    int N = 10;
    CausalMatrix mat(N, testFile);
    
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

TEST_F(CausalMatrixTest, MultiplicationSmall) {
    // A: 0->1, 1->2. (Path 0->1->2)
    // B: Same.
    // A*B should have (0, 2) = 1.
    
    CausalMatrix mat(3, testFile);
    mat.set(0, 1, true);
    mat.set(1, 2, true);
    
    auto res = mat.multiply(mat, resultFile);
    
    EXPECT_EQ(res->get(0, 2), 1);
    EXPECT_EQ(res->get(0, 1), 0); // No path of length 2
}

TEST_F(CausalMatrixTest, MultiplicationPaths) {
    // 0->1, 0->2, 1->3, 2->3
    // Paths from 0 to 3: 0->1->3 and 0->2->3. Total 2.
    
    CausalMatrix mat(4, testFile);
    mat.set(0, 1, true);
    mat.set(0, 2, true);
    mat.set(1, 3, true);
    mat.set(2, 3, true);
    
    auto res = mat.multiply(mat, resultFile);
    
    EXPECT_EQ(res->get(0, 3), 2);
}

TEST_F(CausalMatrixTest, UpperTriangularConstraint) {
    CausalMatrix mat(10, testFile);
    EXPECT_THROW(mat.set(1, 1, true), std::invalid_argument);
    EXPECT_THROW(mat.set(2, 1, true), std::invalid_argument);
}

TEST_F(CausalMatrixTest, Persistence) {
    {
        CausalMatrix mat(100, testFile);
        mat.set(50, 60, true);
    } // mat goes out of scope, file should be saved/closed

    {
        CausalMatrix mat2(100, testFile);
        EXPECT_TRUE(mat2.get(50, 60));
        EXPECT_FALSE(mat2.get(50, 61));
    }
}

TEST_F(CausalMatrixTest, LargeIndexCalculation) {
    uint64_t N = 1000; 
    CausalMatrix mat(N, testFile);
    mat.set(0, N-1, true);
    EXPECT_TRUE(mat.get(0, N-1));
    
    mat.set(N-2, N-1, true);
    EXPECT_TRUE(mat.get(N-2, N-1));
}
