#include "pycauset/math/Eigen.hpp"

#include "pycauset/compute/ComputeContext.hpp"

namespace pycauset {

double trace(const MatrixBase& matrix) {
    if (auto cached = matrix.get_cached_trace()) {
        return *cached;
    }

    double tr = ComputeContext::instance().get_device()->trace(matrix);
    matrix.set_cached_trace(tr);
    return tr;
}

double determinant(const MatrixBase& matrix) {
    if (auto cached = matrix.get_cached_determinant()) {
        return *cached;
    }

    double det = ComputeContext::instance().get_device()->determinant(matrix);
    matrix.set_cached_determinant(det);
    return det;
}

} // namespace pycauset


