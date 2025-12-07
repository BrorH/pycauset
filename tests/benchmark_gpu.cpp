#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <filesystem>
#include "DenseMatrix.hpp"
#include "DenseBitMatrix.hpp"
#include "ComputeContext.hpp"
#include "AcceleratorConfig.hpp"
#include "ComputeDevice.hpp"

using namespace pycauset;

void benchmark_float32(int N) {
    std::cout << "\n--- Benchmarking Float32 Matrix Multiplication (N=" << N << ") ---" << std::endl;
    
    std::string fA = "bench_f32_A.bin";
    std::string fB = "bench_f32_B.bin";
    std::string fC = "bench_f32_C.bin";
    
    if (std::filesystem::exists(fA)) std::filesystem::remove(fA);
    if (std::filesystem::exists(fB)) std::filesystem::remove(fB);
    if (std::filesystem::exists(fC)) std::filesystem::remove(fC);

    DenseMatrix<float> A(N, fA);
    DenseMatrix<float> B(N, fB);
    DenseMatrix<float> C(N, fC);

    // Fill with non-zero values
    std::cout << "Filling matrices..." << std::endl;
    for(int i=0; i<N; ++i) {
        for(int j=0; j<N; ++j) {
            A.set(i, j, 1.0f);
            B.set(i, j, 1.0f);
        }
    }
    std::cout << "A(0,0) = " << A.get(0,0) << std::endl;
    
    // 1. CPU Benchmark (Force CPU)
    ComputeContext::instance().disable_gpu();
    std::cout << "Running CPU..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    ComputeContext::instance().get_device()->matmul(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "CPU Time: " << diff.count() << " s" << std::endl;
    
    // 2. GPU Benchmark
    AcceleratorConfig config;
    config.device_id = 0;
    ComputeContext::instance().enable_gpu(config);
    
    if (ComputeContext::instance().is_gpu_active()) {
        std::cout << "Running GPU..." << std::endl;
        auto start_gpu = std::chrono::high_resolution_clock::now();
        ComputeContext::instance().get_device()->matmul(A, B, C);
        auto end_gpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_gpu = end_gpu - start_gpu;
        std::cout << "GPU Time: " << diff_gpu.count() << " s" << std::endl;
    } else {
        std::cout << "GPU not available." << std::endl;
    }

    A.close(); B.close(); C.close();
    if (std::filesystem::exists(fA)) std::filesystem::remove(fA);
    if (std::filesystem::exists(fB)) std::filesystem::remove(fB);
    if (std::filesystem::exists(fC)) std::filesystem::remove(fC);
}

void benchmark_bitmatrix(int N) {
    std::cout << "\n--- Benchmarking BitMatrix Multiplication (N=" << N << ") ---" << std::endl;
    
    std::string fA = "bench_bit_A.bin";
    std::string fB = "bench_bit_B.bin";
    std::string fC = "bench_bit_C.bin";
    
    if (std::filesystem::exists(fA)) std::filesystem::remove(fA);
    if (std::filesystem::exists(fB)) std::filesystem::remove(fB);
    if (std::filesystem::exists(fC)) std::filesystem::remove(fC);

    DenseMatrix<bool> A(N, fA);
    DenseMatrix<bool> B(N, fB);
    DenseMatrix<int32_t> C(N, fC);

    // 1. CPU Benchmark
    ComputeContext::instance().disable_gpu();
    std::cout << "Running CPU..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    // CpuDevice::matmul for bool?
    // CpuDevice usually implements matmul for double.
    // For bool, it might fallback to generic?
    // Or we call A.multiply(B)?
    // A.multiply calls dispatch.
    // Let's call A.multiply(B, fC).
    // But wait, A.multiply returns a new matrix.
    // We want to use C.
    // Let's use ComputeContext::instance().get_device()->matmul(A, B, C).
    // Does CpuDevice support bool?
    // If not, we skip CPU.
    
    try {
        ComputeContext::instance().get_device()->matmul(A, B, C);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_cpu = end_cpu - start_cpu;
        std::cout << "CPU Time: " << diff_cpu.count() << " s" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "CPU Benchmark failed/unsupported: " << e.what() << std::endl;
    }

    // 2. GPU Benchmark
    AcceleratorConfig config;
    config.device_id = 0;
    ComputeContext::instance().enable_gpu(config);
    
    if (ComputeContext::instance().is_gpu_active()) {
        std::cout << "Running GPU..." << std::endl;
        auto start_gpu = std::chrono::high_resolution_clock::now();
        ComputeContext::instance().get_device()->matmul(A, B, C);
        auto end_gpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_gpu = end_gpu - start_gpu;
        std::cout << "GPU Time: " << diff_gpu.count() << " s" << std::endl;
    } else {
        std::cout << "GPU not available." << std::endl;
    }

    A.close(); B.close(); C.close();
    if (std::filesystem::exists(fA)) std::filesystem::remove(fA);
    if (std::filesystem::exists(fB)) std::filesystem::remove(fB);
    if (std::filesystem::exists(fC)) std::filesystem::remove(fC);
}

int main(int argc, char** argv) {
    std::cout << "PyCauset GPU Benchmarks" << std::endl;
    
    benchmark_float32(2048);
    benchmark_bitmatrix(4096);
    
    return 0;
}
