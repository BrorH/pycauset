#include <gtest/gtest.h>
#include "DenseMatrix.hpp"
#include "Float16.hpp"
#include "CudaDevice.hpp"
#include "AcceleratorConfig.hpp"
#include <filesystem>
#include <memory>

using namespace pycauset;

class GpuFloat16Test : public ::testing::Test {
protected:
    void SetUp() override {
        AcceleratorConfig config;
        config.device_id = 0;
        try {
            device = std::unique_ptr<ComputeDevice>(new CudaDevice(config));
        } catch (const std::exception& e) {
            GTEST_SKIP() << "CUDA not available: " << e.what();
        }
    }

    void TearDown() override {
        for (const auto& f : temp_files) {
            if (std::filesystem::exists(f)) std::filesystem::remove(f);
        }
    }

    std::string make_temp_file(const std::string& name) {
        temp_files.push_back(name);
        return name;
    }

    std::vector<std::string> temp_files;
    std::unique_ptr<ComputeDevice> device;
};

TEST_F(GpuFloat16Test, Matmul_Float16) {
    if (!device) return;

    int N = 64;
    auto A = std::make_unique<DenseMatrix<Float16>>(N, make_temp_file("f16_A.bin"));
    auto B = std::make_unique<DenseMatrix<Float16>>(N, make_temp_file("f16_B.bin"));
    auto C = std::make_unique<DenseMatrix<Float16>>(N, make_temp_file("f16_C.bin"));

    // Fill Identity * 2
    for(int i=0; i<N; ++i) {
        A->set(i, i, Float16(1.0f));
        B->set(i, i, Float16(2.0f));
    }

    device->matmul(*A, *B, *C);

    // Verify
    for(int i=0; i<N; ++i) {
        float val = static_cast<float>(C->get(i, i));
        EXPECT_NEAR(val, 2.0f, 1e-3f);
    }
}

TEST_F(GpuFloat16Test, Matmul_Float16_Streaming) {
    if (!device) return;

    int N = 1024;
    auto A = std::make_unique<DenseMatrix<Float16>>(N, make_temp_file("f16_str_A.bin"));
    auto B = std::make_unique<DenseMatrix<Float16>>(N, make_temp_file("f16_str_B.bin"));
    auto C = std::make_unique<DenseMatrix<Float16>>(N, make_temp_file("f16_str_C.bin"));

    // Fill Diagonal
    for(int i=0; i<N; ++i) {
        A->set(i, i, Float16(1.0f));
        B->set(i, i, Float16(2.0f));
    }

    device->matmul(*A, *B, *C);

    for(int i=0; i<N; ++i) {
        float val = static_cast<float>(C->get(i, i));
        EXPECT_NEAR(val, 2.0f, 1e-3f);
    }
}

TEST_F(GpuFloat16Test, Inverse_Float32) {
    if (!device) return;

    int N = 100;
    auto A = std::make_unique<DenseMatrix<float>>(N, make_temp_file("f32_inv_A.bin"));
    auto B = std::make_unique<DenseMatrix<float>>(N, make_temp_file("f32_inv_B.bin"));

    // Fill Identity
    for(int i=0; i<N; ++i) {
        A->set(i, i, 2.0f);
    }

    device->inverse(*A, *B);

    // Verify
    for(int i=0; i<N; ++i) {
        float val = static_cast<float>(B->get(i, i));
        EXPECT_NEAR(val, 0.5f, 1e-3f);
    }
}
