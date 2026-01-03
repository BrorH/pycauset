#include "pycauset/matrix/MatrixOps.hpp"
#include "pycauset/math/LinearAlgebra.hpp"

namespace pycauset {

std::unique_ptr<MatrixBase> operator*(const MatrixBase& lhs, const MatrixBase& rhs) {
    return dispatch_matmul(lhs, rhs);
}

} // namespace pycauset
