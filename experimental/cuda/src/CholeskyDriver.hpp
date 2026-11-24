#pragma once

// CholeskyDriver
// What: GPU-backed Cholesky factorization driver for dense float matrices.
// Why: Provides a plug-and-play acceleration routine for SPD factorizations.
// Dependencies: CudaDevice (cuSOLVER/cuBLAS) and dense matrix types.

#include "pycauset/matrix/MatrixBase.hpp"

namespace pycauset {

class CudaDevice;

class CholeskyDriver {
public:
    static void run(CudaDevice& device, const MatrixBase& in, MatrixBase& out);
};

} // namespace pycauset
