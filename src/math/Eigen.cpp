#include "pycauset/math/DerivedCache.hpp"
#include "pycauset/math/Eigen.hpp"

#include <cmath>
#include <vector>

#include "pycauset/compute/ComputeContext.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/math/ScalarOps.hpp"
#include "pycauset/math/ViewRead.hpp"
#include "pycauset/math/ViewValue.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"

namespace pycauset {

// det / slogdet (NumPy's det, slogdet): the pivots of an LU factorisation. Structure rules:
//   identity:            s^n (one pivot value repeated), O(1) work
//   diagonal, triangular: the diagonal is the pivots (no factorisation), O(n) typed reads; a
//                        strict triangle (no stored diagonal) is singular
//   otherwise:           ?getrf in NumPy's LAPACK dtype (float16/float32 in single precision),
//                        after to_dense (the shared typed reader) for any other dtype, view or
//                        structure.
// Replaces a kernel that read every element through the virtual accessor into a float64 /
// complex128 buffer (every input was a generic, double-precision path).
static LuPivots pivots_of(const MatrixBase& a) {
    const uint64_t n = a.rows();
    if (n != a.cols()) throw std::invalid_argument("det: matrix must be square");
    const Structure st = a.get_structure();
    LuPivots out;
    if (st == Structure::IDENTITY) {
        out.pivots.assign(n, view_value(1.0, a.get_scalar(), a.is_conjugated()));
        return out;
    }
    if (st == Structure::TRIANGULAR && !static_cast<const TriangularMatrixBase&>(a).has_diagonal()) {
        out.pivots.assign(n, std::complex<double>(0.0, 0.0));
        return out;
    }
    if (st == Structure::DIAGONAL || st == Structure::TRIANGULAR) {
        out.pivots.resize(n);
        if (read_diagonal(a, out.pivots.data())) return out;
    }
    const DataType dt = lapack_dtype(a.get_data_type(), a.get_data_type());
    const bool ready = st == Structure::DENSE && a.get_data_type() == dt && is_plain(a);
    std::unique_ptr<MatrixBase> h = ready ? nullptr : to_dense(a, dt);
    return ComputeContext::instance().get_device()->lu_pivots(h ? *h : a);
}

static std::complex<double> determinant_compute(const MatrixBase& matrix) {
    const LuPivots lu = pivots_of(matrix);
    std::complex<double> det = lu.odd_permutation ? -1.0 : 1.0;
    for (const auto& p : lu.pivots) det *= p;
    return det;
}

std::complex<double> determinant(const MatrixBase& matrix) {
    return cached<std::complex<double>>(matrix, "det", "", [&] { return determinant_compute(matrix); });
}

static std::pair<std::complex<double>, double> slogdet_compute(const MatrixBase& matrix) {
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

std::pair<std::complex<double>, double> slogdet(const MatrixBase& matrix) {
    return cached<std::pair<std::complex<double>, double>>(matrix, "slogdet", "", [&] { return slogdet_compute(matrix); });
}

} // namespace pycauset
