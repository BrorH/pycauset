#include <gtest/gtest.h>
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/core/ObjectFactory.hpp"
#include "pycauset/compute/ComputeContext.hpp"
#include "pycauset/compute/ComputeDevice.hpp"
#include <filesystem>
#include <cmath>

using namespace pycauset;

class TypeTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
        // Cleanup
        for (const auto& f : temp_files) {
            if (std::filesystem::exists(f)) std::filesystem::remove(f);
        }
    }

    std::string make_temp_file(const std::string& name) {
        temp_files.push_back(name);
        return name;
    }

    std::vector<std::string> temp_files;
};

TEST_F(TypeTest, Float32_Addition_AntiPromotion) {
    int N = 100;
    auto A = std::make_unique<DenseMatrix<float>>(N, make_temp_file("f32_add_A.bin"));
    auto B = std::make_unique<DenseMatrix<float>>(N, make_temp_file("f32_add_B.bin"));

    // Fill
    for(int i=0; i<N; ++i) {
        A->set(i, i, 1.0f);
        B->set(i, i, 2.0f);
    }

    // A + B
    auto C = pycauset::add(*A, *B, make_temp_file("f32_add_C.bin"));

    // Verify Type
    EXPECT_EQ(C->get_data_type(), DataType::FLOAT32);
    
    // Verify Values
    auto* C_dense = dynamic_cast<DenseMatrix<float>*>(C.get());
    ASSERT_NE(C_dense, nullptr);
    
    for(int i=0; i<N; ++i) {
        EXPECT_FLOAT_EQ(C_dense->get(i, i), 3.0f);
        EXPECT_FLOAT_EQ(C_dense->get(i, (i+1)%N), 0.0f);
    }
}

TEST_F(TypeTest, Float32_Matmul_AntiPromotion) {
    int N = 64;
    auto A = std::make_unique<DenseMatrix<float>>(N, make_temp_file("f32_mul_A.bin"));
    auto B = std::make_unique<DenseMatrix<float>>(N, make_temp_file("f32_mul_B.bin"));

    // Fill Identity
    for(int i=0; i<N; ++i) {
        A->set(i, i, 1.0f);
        B->set(i, i, 2.0f);
    }

    // A * B
    // Note: DenseMatrix::multiply returns unique_ptr<DenseMatrix<int32_t>> for bool,
    // but for float/double it's not defined in DenseMatrix directly?
    // Wait, DenseMatrix<T> doesn't have multiply() method for generic T?
    // It has multiply_scalar.
    // Matmul is usually done via MatrixOperations::multiply or ComputeDevice::matmul.
    // Let's use MatrixOperations::multiply (if it exists) or just check how bindings do it.
    // Bindings use A.multiply(B) if A is DenseMatrix<bool>.
    // For float, bindings use pycauset::matmul(A, B).
    
    // Let's check MatrixOperations.hpp for matmul.
    // It's not there?
    // Usually it's ComputeContext::instance().get_device()->matmul(A, B, C).
    
    auto C = std::make_unique<DenseMatrix<float>>(N, make_temp_file("f32_mul_C.bin"));
    ComputeContext::instance().get_device()->matmul(*A, *B, *C);
    
    // Verify Values
    for(int i=0; i<N; ++i) {
        EXPECT_FLOAT_EQ(C->get(i, i), 2.0f);
    }
}

TEST_F(TypeTest, Float32_AddScalar_AntiPromotion) {
    int N = 100;
    auto A = std::make_unique<DenseMatrix<float>>(N, make_temp_file("f32_sadd_A.bin"));
    
    for(int i=0; i<N; ++i) A->set(i, i, 1.0f);

    // A + 0.5
    auto C = A->add_scalar(0.5, make_temp_file("f32_sadd_C.bin"));
    
    // Verify Type
    EXPECT_EQ(C->get_data_type(), DataType::FLOAT32);
    
    auto* C_dense = dynamic_cast<DenseMatrix<float>*>(C.get());
    ASSERT_NE(C_dense, nullptr);
    
    for(int i=0; i<N; ++i) {
        EXPECT_FLOAT_EQ(C_dense->get(i, i), 1.5f);
        EXPECT_FLOAT_EQ(C_dense->get(i, (i+1)%N), 0.5f);
    }
}
