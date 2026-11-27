// Typed readers for every matrix class with the view applied once (review PLAN N.2).
// Kept in its own translation unit: one instantiation per (class, wide type) pair.
#include "pycauset/math/ViewRead.hpp"
#include "pycauset/math/ViewValue.hpp"
#include "pycauset/core/DTypeDispatch.hpp"

#include <string>
#include <type_traits>

#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/core/ScalarUtils.hpp"
#include "pycauset/matrix/ComplexFloat16Matrix.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/matrix/IdentityMatrix.hpp"
#include "pycauset/matrix/SymmetricMatrix.hpp"
#include "pycauset/matrix/TriangularBitMatrix.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"

namespace pycauset {
namespace {

// Logical value (i, j) of a as W, from the class's typed `get` (raw storage, transposition
// applied) and the view rule.
template <typename W, typename Src>
inline W logical(const Src& m, uint64_t i, uint64_t j, std::complex<double> s, bool conj, bool anti) {
    if (anti && i == j) return W(0);
    const auto raw = m.get(i, j);
    using In = std::decay_t<decltype(raw)>;
    if constexpr (std::is_same_v<W, int64_t> && std::is_integral_v<In> && !std::is_same_v<In, bool>) {
        // Integer storage into an integer result stays exact (no detour through double).
        if (scalar::is_exact_int_factor(s)) {
            const int64_t k = static_cast<int64_t>(s.real());
            return static_cast<int64_t>(k == 1 ? raw : scalar::wrap_scale(raw, k));
        }
        return static_cast<int64_t>(view_value(raw, s, conj).real());
    } else {
        const std::complex<double> z = view_value(raw, s, conj);
        if constexpr (std::is_same_v<W, std::complex<double>>) return z;
        else if constexpr (std::is_same_v<W, double>) return z.real();
        else return static_cast<int64_t>(z.real());
    }
}

// Rows [i0, i1) into dst (row-major, cols wide).
template <typename W, typename Src>
void fill_rows(const Src& m, const MatrixBase& a, bool anti, uint64_t i0, uint64_t i1, W* dst) {
    const std::complex<double> s = a.get_scalar();
    const bool conj = a.is_conjugated();
    const uint64_t cols = a.cols();
    ParallelFor(i0, i1, [&](size_t ii) {
        const uint64_t i = static_cast<uint64_t>(ii);
        W* row = dst + (i - i0) * cols;
        for (uint64_t j = 0; j < cols; ++j) row[j] = logical<W>(m, i, j, s, conj, anti);
    });
}

// Can the class's typed `get` read a? Dense `get` honours windows and transposition; the
// other classes' `get` (and the bit specialisation's) honour transposition, not windows:
// their windows read virtually. A lazy closure computes on read (virtual path).
bool typed_readable(const MatrixBase& a) {
    const Structure st = a.get_structure();
    if (a.is_subview() && (st != Structure::DENSE || a.get_data_type() == DataType::BIT)) return false;
    return dynamic_cast<const BitClosureMatrix*>(&a) == nullptr;
}

template <typename W>
bool rows_as(const MatrixBase& a, uint64_t i0, uint64_t i1, W* dst) {
    if (!typed_readable(a)) return false;
    visit_matrix(a, [&](const auto& typed) { fill_rows<W>(typed, a, a.get_structure() == Structure::ANTISYMMETRIC, i0, i1, dst); });
    return true;
}

template <typename W>
bool diagonal_as(const MatrixBase& a, W* dst) {
    if (!typed_readable(a)) return false;
    const std::complex<double> s = a.get_scalar();
    const bool conj = a.is_conjugated(), anti = a.get_structure() == Structure::ANTISYMMETRIC;
    const uint64_t n = std::min(a.rows(), a.cols());
    visit_matrix(a, [&](const auto& typed) {
        for (uint64_t i = 0; i < n; ++i) dst[i] = logical<W>(typed, i, i, s, conj, anti);
    });
    return true;
}

template <typename W>
std::unique_ptr<MatrixBase> read_as(const MatrixBase& a) {
    if (!typed_readable(a)) return nullptr;
    auto out = std::make_unique<DenseMatrix<W>>(a.rows(), a.cols(), std::string());
    rows_as<W>(a, 0, a.rows(), out->data());
    return out;
}

}  // namespace

std::unique_ptr<MatrixBase> read_wide(const MatrixBase& a, WideKind kind) {
    switch (kind) {
        case WideKind::Int64: return read_as<int64_t>(a);
        case WideKind::Float64: return read_as<double>(a);
        case WideKind::Complex128: return read_as<std::complex<double>>(a);
    }
    return nullptr;
}

#define PYCAUSET_ROWS(W)                                                                       \
    bool read_rows(const MatrixBase& a, uint64_t i0, uint64_t i1, W* out) { return rows_as<W>(a, i0, i1, out); } \
    bool read_diagonal(const MatrixBase& a, W* out) { return diagonal_as<W>(a, out); }
PYCAUSET_ROWS(int64_t)
PYCAUSET_ROWS(double)
PYCAUSET_ROWS(std::complex<double>)
#undef PYCAUSET_ROWS

}  // namespace pycauset
