#include "pycauset/math/Eigen.hpp"

#include "pycauset/compute/ComputeContext.hpp"
#include "pycauset/math/ScalarOps.hpp"

namespace pycauset {

std::complex<double> determinant(const MatrixBase& matrix) {
    if (!is_plain(matrix)) return determinant(*write_out(matrix));  // kernels read storage (F-062)
    return ComputeContext::instance().get_device()->determinant(matrix);
}

} // namespace pycauset
