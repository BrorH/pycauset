#pragma once

// MatmulDriver
// What: Host-side orchestration wrapper for streaming GEMM using AsyncStreamer.
// Why: Centralizes the GPU matmul streaming path as a plug-and-play driver routine.
// Dependencies: Requires CudaDevice streaming helpers and AsyncStreamer.

#include "pycauset/matrix/DenseMatrix.hpp"
#include <cstddef>

namespace pycauset {

class CudaDevice;

class MatmulDriver {
public:
    static void run(CudaDevice& device,
                    const DenseMatrix<double>& a,
                    const DenseMatrix<double>& b,
                    DenseMatrix<double>& c,
                    size_t available_mem);

    static void run(CudaDevice& device,
                    const DenseMatrix<float>& a,
                    const DenseMatrix<float>& b,
                    DenseMatrix<float>& c,
                    size_t available_mem);
};

} // namespace pycauset
