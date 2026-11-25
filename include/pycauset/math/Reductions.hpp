#pragma once
// Whole-array reductions with NumPy's result kinds (FINDINGS F-051).
//
//   integer / bit input -> exact integer (128-bit accumulation; view factors applied with
//                          NumPy's element wrap; std::overflow_error beyond 127 bits)
//   real float input    -> double
//   complex input       -> complex<double>
//
// Floating-point sums are deterministic: each row is summed in order, then the row sums are
// added in row order, independent of the thread count.

#include <complex>
#include <cstdint>

namespace pycauset {

class MatrixBase;
class VectorBase;

struct ReductionValue {
    enum class Kind { Int, Real, Complex };
    Kind kind = Kind::Real;
    __int128 i = 0;
    double r = 0.0;
    std::complex<double> c{0.0, 0.0};
};

ReductionValue sum(const MatrixBase& m);
ReductionValue sum(const VectorBase& v);

}  // namespace pycauset
