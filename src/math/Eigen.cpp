#include "pycauset/math/Eigen.hpp"

#include <cmath>

#include "pycauset/compute/ComputeContext.hpp"
#include "pycauset/math/ScalarOps.hpp"

namespace pycauset {

static LuPivots pivots_of(const MatrixBase& matrix) {
    if (!is_plain(matrix)) return pivots_of(*write_out(matrix));  // kernels read storage (F-062)
    return ComputeContext::instance().get_device()->lu_pivots(matrix);
}

std::complex<double> determinant(const MatrixBase& matrix) {
    const LuPivots lu = pivots_of(matrix);
    std::complex<double> det = lu.odd_permutation ? -1.0 : 1.0;
    for (const auto& p : lu.pivots) det *= p;
    return det;
}

std::pair<std::complex<double>, double> slogdet(const MatrixBase& matrix) {
    // Sum of logs: no overflow where the determinant itself is beyond double range (F-070).
    const LuPivots lu = pivots_of(matrix);
    std::complex<double> sign = lu.odd_permutation ? -1.0 : 1.0;
    double logabs = 0.0;
    for (const auto& p : lu.pivots) {
        const double a = std::abs(p);
        if (a == 0.0) return {0.0, -INFINITY};
        sign *= p / a;
        logabs += std::log(a);
    }
    return {sign, logabs};
}

} // namespace pycauset
