#include "pycauset/math/Eigen.hpp"

#include "pycauset/compute/ComputeContext.hpp"

namespace pycauset {

double trace(const MatrixBase& matrix) {
    return ComputeContext::instance().get_device()->trace(matrix);
}

double determinant(const MatrixBase& matrix) {
    return ComputeContext::instance().get_device()->determinant(matrix);
}

} // namespace pycauset


