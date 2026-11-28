// Comparisons for every dtype, structure and view (review F-128; protocol: Adding Operations).
//
//   entry:      compare(a, b, op)
//   result:     a dense bool matrix of the broadcast shape
//   normaliser: each operand read in its kind's wide dtype (to_dense, the shared typed reader;
//               no copy for a plain dense operand already in it): bool / signed / unsigned up to
//               32 bits -> int64, uint64 -> uint64, real floats -> float64, complex -> complex128
//   kernel:     one templated loop over the (row-broadcast) result, comparing exactly
#include "pycauset/math/Compare.hpp"

#include <complex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "pycauset/core/DebugTrace.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/math/ScalarOps.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"

namespace pycauset {
namespace {

DataType wide_dtype(DataType d) {
    if (is_complex_dtype(d)) return DataType::COMPLEX_FLOAT64;
    if (is_real_float_dtype(d)) return DataType::FLOAT64;
    if (d == DataType::UINT64) return DataType::UINT64;
    return DataType::INT64;  // bool and every other integer fit int64
}

// The operand as a plain dense matrix of `dt` (itself when it already is one).
std::unique_ptr<MatrixBase> read_as(const MatrixBase& m, DataType dt, const MatrixBase*& out) {
    if (m.get_data_type() == dt && m.get_structure() == Structure::DENSE && is_plain(m) &&
        m.rows() == m.base_rows() && m.cols() == m.base_cols()) {
        out = &m;
        return nullptr;
    }
    auto h = to_dense(m, dt);
    out = h.get();
    return h;
}

template <typename X, typename Y>
bool cmp(CompareOp op, X x, Y y) {
    if constexpr (std::is_integral_v<X> && std::is_integral_v<Y>) {
        switch (op) {  // exact for signed with unsigned (std::cmp_*)
            case CompareOp::Eq: return std::cmp_equal(x, y);
            case CompareOp::Ne: return std::cmp_not_equal(x, y);
            case CompareOp::Lt: return std::cmp_less(x, y);
            case CompareOp::Le: return std::cmp_less_equal(x, y);
            case CompareOp::Gt: return std::cmp_greater(x, y);
            case CompareOp::Ge: return std::cmp_greater_equal(x, y);
        }
        return false;
    } else if constexpr (std::is_same_v<X, std::complex<double>> || std::is_same_v<Y, std::complex<double>>) {
        const std::complex<double> u(x), v(y);
        // NumPy 2 orders complex numbers lexicographically (real part, then imaginary part).
        switch (op) {
            case CompareOp::Eq: return u == v;
            case CompareOp::Ne: return u != v;
            case CompareOp::Lt: return u.real() < v.real() || (u.real() == v.real() && u.imag() < v.imag());
            case CompareOp::Le: return u.real() < v.real() || (u.real() == v.real() && u.imag() <= v.imag());
            case CompareOp::Gt: return u.real() > v.real() || (u.real() == v.real() && u.imag() > v.imag());
            case CompareOp::Ge: return u.real() > v.real() || (u.real() == v.real() && u.imag() >= v.imag());
        }
        return false;
    } else {
        const double u = static_cast<double>(x), v = static_cast<double>(y);  // an integer with a float: float64, as NumPy
        switch (op) {
            case CompareOp::Eq: return u == v;
            case CompareOp::Ne: return u != v;
            case CompareOp::Lt: return u < v;
            case CompareOp::Le: return u <= v;
            case CompareOp::Gt: return u > v;
            case CompareOp::Ge: return u >= v;
        }
        return false;
    }
}

template <typename F>
auto with_wide(DataType dt, F&& f) {
    switch (dt) {
        case DataType::INT64: return f(int64_t{});
        case DataType::UINT64: return f(uint64_t{});
        case DataType::FLOAT64: return f(double{});
        default: return f(std::complex<double>{});
    }
}

uint64_t broadcast(uint64_t x, uint64_t y, const char* axis) {
    if (x == y || y == 1) return x;
    if (x == 1) return y;
    throw std::invalid_argument(std::string("comparison: operands could not be broadcast together (") + axis +
                                " " + std::to_string(x) + " vs " + std::to_string(y) + ")");
}

}  // namespace

std::unique_ptr<MatrixBase> compare(const MatrixBase& a, const MatrixBase& b, CompareOp op) {
    const uint64_t rows = broadcast(a.rows(), b.rows(), "rows");
    const uint64_t cols = broadcast(a.cols(), b.cols(), "cols");
    const MatrixBase* pa = nullptr;
    const MatrixBase* pb = nullptr;
    const auto ha = read_as(a, wide_dtype(a.get_data_type()), pa);
    const auto hb = read_as(b, wide_dtype(b.get_data_type()), pb);
    auto out = std::make_unique<DenseMatrix<bool>>(rows, cols, std::string());
    debug_trace::set_last("cpu.compare");
    with_wide(pa->get_data_type(), [&](auto xt) {
        using X = decltype(xt);
        with_wide(pb->get_data_type(), [&](auto yt) {
            using Y = decltype(yt);
            const X* x = static_cast<const DenseMatrix<X>&>(*pa).data();
            const Y* y = static_cast<const DenseMatrix<Y>&>(*pb).data();
            const uint64_t ac = pa->cols(), bc = pb->cols();
            const bool ar = pa->rows() == 1, br = pb->rows() == 1, acb = ac == 1, bcb = bc == 1;
            // Rows in parallel, written as packed words (a row of a bit matrix is whole words; the
            // new storage is zero). `set` would bump the storage version per element.
            uint64_t* words = out->data();
            const uint64_t stride = out->stride_bytes() / 8;
            ParallelFor(0, rows, [&](size_t i) {
                const X* xr = x + (ar ? 0 : i) * ac;
                const Y* yr = y + (br ? 0 : i) * bc;
                uint64_t* w = words + i * stride;
                for (uint64_t j = 0; j < cols; ++j) {
                    if (cmp(op, xr[acb ? 0 : j], yr[bcb ? 0 : j])) w[j / 64] |= uint64_t{1} << (j % 64);
                }
            });
            return 0;
        });
        return 0;
    });
    return out;
}

}  // namespace pycauset
