#include <gtest/gtest.h>
#include "DenseBitMatrix.hpp"
#include "DenseMatrix.hpp"
#include "ComputeContext.hpp"
#include "AcceleratorConfig.hpp"
#include "ComputeDevice.hpp"
#include <filesystem>
#include <vector>
#include <random>

using namespace pycauset;

class GpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure GPU is active if available
        AcceleratorConfig config;
        config.device_id = 0;
        ComputeContext::instance().enable_gpu(config);
    }

    void TearDown() override {
    }
};

TEST_F(GpuTest, Float32_Multiplication) {
    if (!ComputeContext::instance().is_gpu_active()) {
        std::cout << "[SKIPPED] GPU not available" << std::endl;
        GTEST_SKIP() << "GPU not available";
    }
    std::cout << "[INFO] GPU is active." << std::endl;

    std::string fA = "gpu_f32_A.bin";
    std::string fB = "gpu_f32_B.bin";
    std::string fC = "gpu_f32_C.bin";
    
    if (std::filesystem::exists(fA)) std::filesystem::remove(fA);
    if (std::filesystem::exists(fB)) std::filesystem::remove(fB);
    if (std::filesystem::exists(fC)) std::filesystem::remove(fC);

    int N = 256;
    std::cout << "[INFO] Creating Float32 matrices..." << std::endl;
    DenseMatrix<float> A(N, fA);
    DenseMatrix<float> B(N, fB);
    DenseMatrix<float> C(N, fC);

    // Fill with identity
    std::cout << "[INFO] Filling matrices..." << std::endl;
    for(int i=0; i<N; ++i) {
        A.set(i, i, 1.0f);
        B.set(i, i, 2.0f);
    }

    std::cout << "[INFO] Running matmul..." << std::endl;
    try {
        ComputeContext::instance().get_device()->matmul(A, B, C);
    } catch (const std::exception& e) {
        std::cout << "[ERROR] matmul failed: " << e.what() << std::endl;
        throw;
    }

    std::cout << "[INFO] Verifying results..." << std::endl;
    for(int i=0; i<N; ++i) {
        EXPECT_FLOAT_EQ(C.get(i, i), 2.0f);
        EXPECT_FLOAT_EQ(C.get(i, (i+1)%N), 0.0f);
    }
    
    A.close(); B.close(); C.close();
    if (std::filesystem::exists(fA)) std::filesystem::remove(fA);
    if (std::filesystem::exists(fB)) std::filesystem::remove(fB);
    if (std::filesystem::exists(fC)) std::filesystem::remove(fC);
}

TEST_F(GpuTest, DenseBitMatrix_Multiplication) {
    if (!ComputeContext::instance().is_gpu_active()) {
        GTEST_SKIP() << "GPU not available";
    }
    std::cout << "[INFO] Starting BitMatrix Test..." << std::endl;

    std::string fA = "gpu_bit_A.bin";
    std::string fB = "gpu_bit_B.bin";
    std::string fC = "gpu_bit_C.bin";

    if (std::filesystem::exists(fA)) std::filesystem::remove(fA);
    if (std::filesystem::exists(fB)) std::filesystem::remove(fB);
    if (std::filesystem::exists(fC)) std::filesystem::remove(fC);

    int N = 256; // Multiple of 64 for easy checking
    DenseMatrix<bool> A(N, fA);
    DenseMatrix<bool> B(N, fB);
    DenseMatrix<int32_t> C(N, fC);

    // Set A to Identity
    for(int i=0; i<N; ++i) A.set(i, i, true);
    
    // Set B to have 1s on diagonal and super-diagonal
    for(int i=0; i<N; ++i) {
        B.set(i, i, true);
        if (i < N-1) B.set(i, i+1, true);
    }

    std::cout << "[INFO] Running BitMatrix matmul..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    ComputeContext::instance().get_device()->matmul(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "[INFO] Matmul done in " << diff.count() << "s" << std::endl;
    
    for(int i=0; i<N; ++i) {
        EXPECT_EQ(C.get(i, i), 1);
        if (i < N-1) EXPECT_EQ(C.get(i, i+1), 1);
        if (i > 0) EXPECT_EQ(C.get(i, i-1), 0);
    }

    A.close(); B.close(); C.close();
    if (std::filesystem::exists(fA)) std::filesystem::remove(fA);
    if (std::filesystem::exists(fB)) std::filesystem::remove(fB);
    if (std::filesystem::exists(fC)) std::filesystem::remove(fC);
}
