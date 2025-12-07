#include <gtest/gtest.h>
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/compute/ComputeContext.hpp"
#include "pycauset/core/SystemUtils.hpp"
#include <vector>
#include <chrono>
#include <iostream>
#include <cstring>

// Helper to check if CUDA is available
bool is_cuda_available() {
    return pycauset::ComputeContext::instance().is_gpu_active();
}

TEST(PinnedMemoryTest, BasicAllocation) {
    if (!is_cuda_available()) {
        std::cout << "[SKIPPED] CUDA not available for PinnedMemoryTest" << std::endl;
        return;
    }

    size_t size = 1024 * 1024 * 10; // 10 MB
    MemoryMapper mapper(":memory:", size);
    
    void* ptr = mapper.get_data();
    ASSERT_NE(ptr, nullptr);
    
    // Write data
    int* int_ptr = static_cast<int*>(ptr);
    for (size_t i = 0; i < size / sizeof(int); ++i) {
        int_ptr[i] = (int)i;
    }
    
    // Read data
    for (size_t i = 0; i < size / sizeof(int); ++i) {
        ASSERT_EQ(int_ptr[i], (int)i);
    }
}

TEST(PinnedMemoryTest, StressAllocation) {
    if (!is_cuda_available()) {
        std::cout << "[SKIPPED] CUDA not available for PinnedMemoryTest" << std::endl;
        return;
    }

    size_t size = 1024 * 1024 * 50; // 50 MB
    int iterations = 20;
    
    for (int k = 0; k < iterations; ++k) {
        MemoryMapper mapper(":memory:", size);
        void* ptr = mapper.get_data();
        ASSERT_NE(ptr, nullptr);
        
        // Touch memory to ensure it's backed
        std::memset(ptr, 0xFF, 1024); 
    }
}

TEST(PinnedMemoryTest, LargeAllocation) {
    if (!is_cuda_available()) {
        std::cout << "[SKIPPED] CUDA not available for PinnedMemoryTest" << std::endl;
        return;
    }

    // Try to allocate a significant chunk of RAM (e.g., 500MB)
    // This verifies that pinned memory allocation doesn't fail for larger sizes
    size_t size = 500 * 1024 * 1024; 
    
    // Check if we have enough RAM first to avoid OOM killing the test
    if (pycauset::SystemUtils::get_available_ram() < size * 2) {
        std::cout << "[SKIPPED] Not enough RAM for LargeAllocation test" << std::endl;
        return;
    }

    try {
        MemoryMapper mapper(":memory:", size);
        void* ptr = mapper.get_data();
        ASSERT_NE(ptr, nullptr);
        
        // Write to start and end
        char* char_ptr = static_cast<char*>(ptr);
        char_ptr[0] = 1;
        char_ptr[size - 1] = 1;
    } catch (const std::exception& e) {
        FAIL() << "Large allocation failed: " << e.what();
    }
}

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>

TEST(PinnedMemoryTest, PerformanceBenchmark) {
    if (!is_cuda_available()) {
        return;
    }

    size_t size = 1024 * 1024 * 100; // 100 MB
    size_t bytes = size;

    // 1. Pageable Memory (Standard malloc)
    void* pageable_ptr = std::malloc(bytes);
    std::memset(pageable_ptr, 1, bytes);
    
    void* d_ptr;
    cudaMalloc(&d_ptr, bytes);

    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_ptr, pageable_ptr, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double pageable_time = std::chrono::duration<double>(end - start).count();
    
    std::free(pageable_ptr);

    // 2. Pinned Memory (via MemoryMapper)
    MemoryMapper mapper(":memory:", bytes);
    void* pinned_ptr = mapper.get_data();
    std::memset(pinned_ptr, 1, bytes);

    start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_ptr, pinned_ptr, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double pinned_time = std::chrono::duration<double>(end - start).count();

    cudaFree(d_ptr);

    std::cout << "[Benchmark] 100MB Transfer H2D:" << std::endl;
    std::cout << "  Pageable: " << pageable_time << " s (" << (bytes/1e9)/pageable_time << " GB/s)" << std::endl;
    std::cout << "  Pinned:   " << pinned_time << " s (" << (bytes/1e9)/pinned_time << " GB/s)" << std::endl;
    std::cout << "  Speedup:  " << pageable_time / pinned_time << "x" << std::endl;

    // Pinned should be faster, but on some systems (e.g. WSL2 or unified memory) it might be close.
    // We don't assert strict speedup to avoid flaky tests on CI, but we print it.
}
#endif

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
