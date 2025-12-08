#include <gtest/gtest.h>
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/core/PersistentObject.hpp"
#include <memory>

using namespace pycauset;

TEST(CoWTest, DenseMatrixLazyCopy) {
    uint64_t n = 10;
    auto mat1 = std::make_unique<DenseMatrix<double>>(n, ""); // RAM-backed
    mat1->set(0, 0, 1.0);

    // Clone
    auto mat2_base = mat1->clone();
    auto mat2 = std::unique_ptr<DenseMatrix<double>>(static_cast<DenseMatrix<double>*>(mat2_base.release()));

    // Check they share data
    EXPECT_EQ(mat1->data(), mat2->data());
    EXPECT_EQ(mat1->get(0, 0), 1.0);
    EXPECT_EQ(mat2->get(0, 0), 1.0);

    // Modify mat2
    mat2->set(0, 0, 2.0);

    // Check they no longer share data
    EXPECT_NE(mat1->data(), mat2->data());

    // Check values
    EXPECT_EQ(mat1->get(0, 0), 1.0); // Should be unchanged
    EXPECT_EQ(mat2->get(0, 0), 2.0); // Should be changed
}

TEST(CoWTest, DenseMatrixLazyCopyWriteOriginal) {
    uint64_t n = 10;
    auto mat1 = std::make_unique<DenseMatrix<double>>(n, "");
    mat1->set(0, 0, 1.0);

    auto mat2_base = mat1->clone();
    auto mat2 = std::unique_ptr<DenseMatrix<double>>(static_cast<DenseMatrix<double>*>(mat2_base.release()));

    EXPECT_EQ(mat1->data(), mat2->data());

    // Modify mat1
    mat1->set(0, 0, 3.0);

    EXPECT_NE(mat1->data(), mat2->data());
    EXPECT_EQ(mat1->get(0, 0), 3.0);
    EXPECT_EQ(mat2->get(0, 0), 1.0);
}
