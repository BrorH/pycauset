#pragma once

// ArnoldiDriver
// What: Host-side Arnoldi/Lanczos driver using GPU batch_gemv for matvecs.
// Why: Provides a plug-and-play GPU acceleration routine for top-k eigenvalues.
// Dependencies: CudaDevice batch_gemv and Eigen for Hessenberg eigensolve.

#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/vector/VectorBase.hpp"

namespace pycauset {

class CudaDevice;

class ArnoldiDriver {
public:
    static void run(CudaDevice& device, const MatrixBase& a, VectorBase& out, int k, int m, double tol);
};

} // namespace pycauset
