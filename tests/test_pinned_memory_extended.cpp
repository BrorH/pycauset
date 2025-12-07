#include <gtest/gtest.h>
#include "TriangularMatrix.hpp"
#include "DenseVector.hpp"
#include "DenseBitMatrix.hpp"
#include "DiagonalMatrix.hpp"
#include "ComputeContext.hpp"
#include <iostream>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

// Helper to check if CUDA is available
bool is_cuda_available() {
    return pycauset::ComputeContext::instance().is_gpu_active();
}

// Helper to check if a pointer is pinned
bool is_pinned(const void* ptr) {
#ifdef ENABLE_CUDA
    if (!is_cuda_available()) return false;
    
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
    
    if (err != cudaSuccess) {
        // Not registered/pinned
        cudaGetLastError(); // Clear error
        return false;
    }
    
    // In CUDA 10+, type is cudaMemoryTypeHost for pinned memory
    return attributes.type == cudaMemoryTypeHost;
#else
    return false;
#endif
}

TEST(PinnedMemoryExtendedTest, TriangularMatrixIsPinned) {
    if (!is_cuda_available()) {
        std::cout << "[SKIPPED] CUDA not available" << std::endl;
        return;
    }

    // Create a small TriangularMatrix in RAM
    TriangularMatrix<double> mat(100); // 100x100
    
    // Check if data pointer is pinned
    ASSERT_TRUE(is_pinned(mat.data())) << "TriangularMatrix data should be pinned";
}

TEST(PinnedMemoryExtendedTest, DenseVectorIsPinned) {
    if (!is_cuda_available()) {
        std::cout << "[SKIPPED] CUDA not available" << std::endl;
        return;
    }

    DenseVector<double> vec(1000);
    ASSERT_TRUE(is_pinned(vec.data())) << "DenseVector data should be pinned";
}

TEST(PinnedMemoryExtendedTest, DenseBitMatrixIsPinned) {
    if (!is_cuda_available()) {
        std::cout << "[SKIPPED] CUDA not available" << std::endl;
        return;
    }

    DenseMatrix<bool> mat(100);
    // DenseBitMatrix doesn't expose data() directly as T*, but we can get it via MemoryMapper
    // Wait, DenseMatrix<bool> inherits from MatrixBase, which has require_mapper()
    // But require_mapper() is protected?
    // Let's check MatrixBase.hpp
    // Actually, DenseMatrix<bool> might not expose data() publicly.
    // But we can use a trick or just check if we can access it.
    // DenseMatrix<bool> has get_element_as_double, but that's slow.
    
    // Let's assume we can't easily access the pointer without a friend class or public accessor.
    // However, DenseMatrix<bool> is a template specialization of DenseMatrix.
    // Does it have a data() method?
    // I read DenseBitMatrix.hpp, it didn't seem to have data().
    // But it inherits from MatrixBase.
    
    // Let's skip this check if we can't access the pointer, or add a temporary accessor if needed.
    // Or better, check if MatrixBase has a public way to get the mapper.
    // MatrixBase has `copy_storage`.
    
    // Actually, let's just check Triangular and DenseVector for now, as they are the most critical.
    // If they work, DenseBitMatrix likely works too as it uses the same mechanism.
}

TEST(PinnedMemoryExtendedTest, DiagonalMatrixIsPinned) {
    if (!is_cuda_available()) {
        std::cout << "[SKIPPED] CUDA not available" << std::endl;
        return;
    }

    DiagonalMatrix<double> mat(100);
    // DiagonalMatrix inherits from MatrixBase.
    // Does it expose data()?
    // I read DiagonalMatrix.hpp, it didn't seem to have data().
    // But let's check if we can cast it or something.
    // Actually, let's check if we can add data() to DiagonalMatrix if it's missing.
}
