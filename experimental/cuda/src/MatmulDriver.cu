// MatmulDriver
// What: Host-side orchestration wrapper for streaming GEMM using AsyncStreamer.
// Why: Centralizes the GPU matmul streaming path as a plug-and-play driver routine.
// Dependencies: CudaDevice streaming helpers.

#include "MatmulDriver.hpp"
#include "CudaDevice.hpp"

namespace pycauset {

void MatmulDriver::run(CudaDevice& device,
                       const DenseMatrix<double>& a,
                       const DenseMatrix<double>& b,
                       DenseMatrix<double>& c,
                       size_t available_mem) {
    device.matmul_streaming(&a, &b, &c, available_mem);
}

void MatmulDriver::run(CudaDevice& device,
                       const DenseMatrix<float>& a,
                       const DenseMatrix<float>& b,
                       DenseMatrix<float>& c,
                       size_t available_mem) {
    device.matmul_streaming(&a, &b, &c, available_mem);
}

} // namespace pycauset
