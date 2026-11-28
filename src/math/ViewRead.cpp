// Typed readers for every matrix class with the view applied once (review PLAN N.2).
// Kept in its own translation unit: one instantiation per (class, wide type) pair.
#include "pycauset/math/ViewRead.hpp"
#include "pycauset/math/RawRows.hpp"
#include "pycauset/math/ViewValue.hpp"
#include "pycauset/core/DTypeDispatch.hpp"

#include <string>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <atomic>

#include "pycauset/core/MemoryMapper.hpp"
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

// Logical value of a stored value `raw` (the class's typed `get`: raw storage, transposition
// applied) as W, by the view rule.
template <typename W, typename In>
inline W logical_of(const In& raw, std::complex<double> s, bool conj) {
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

template <typename W, typename Src>
inline W logical(const Src& m, uint64_t i, uint64_t j, std::complex<double> s, bool conj, bool anti) {
    if (anti && i == j) return W(0);
    return logical_of<W>(m.get(i, j), s, conj);
}

// Rows [i0, i1) into dst (row-major, cols wide): stored values in bulk (math/RawRows.hpp)
// where the class has a bulk reader, then the view rule per element.
template <typename W, typename Src>
void fill_rows(const Src& m, const MatrixBase& a, bool anti, uint64_t i0, uint64_t i1, W* dst) {
    const std::complex<double> s = a.get_scalar();
    const bool conj = a.is_conjugated();
    const uint64_t cols = a.cols();
    using In = std::decay_t<decltype(m.get(0, 0))>;
    if constexpr (has_raw_rows_v<In, Src>) {
        // At least 32 tiles when there are rows for it: ParallelFor runs fewer items inline.
        const uint64_t TILE = std::clamp<uint64_t>((i1 - i0) / 32, 1, 32);
        const uint64_t tiles = (i1 - i0 + TILE - 1) / TILE;
        std::atomic<bool> ok{true};  // false for the same reason in every tile (a window, ...)
        ParallelFor(0, tiles, [&](size_t t) {
            const uint64_t r0 = i0 + t * TILE, r1 = std::min(i1, r0 + TILE);
            std::unique_ptr<In[]> raw(new In[(r1 - r0) * cols]);  // filled by raw_rows: no zeroing
            if (!raw_rows<In>(m, r0, r1, raw.get())) { ok = false; return; }
            W* out = dst + (r0 - i0) * cols;
            const In* __restrict rr = raw.get();
            const uint64_t len = (r1 - r0) * cols;
            for (uint64_t q = 0; q < len; ++q) out[q] = logical_of<W>(rr[q], s, conj);
            if (anti)
                for (uint64_t i = r0; i < r1 && i < cols; ++i) out[(i - r0) * cols + i] = W(0);
        });
        if (ok) return;
    }
    ParallelFor(i0, i1, [&](size_t ii) {
        const uint64_t i = static_cast<uint64_t>(ii);
        W* row = dst + (i - i0) * cols;
        for (uint64_t j = 0; j < cols; ++j) row[j] = logical<W>(m, i, j, s, conj, anti);
    });
}

// Can the class's typed `get` read a? Dense `get` honours windows and transposition; the
// other classes' `get` (and the bit specialisation's) honour transposition, not windows:
// their windows read virtually.
bool typed_readable(const MatrixBase& a) {
    const Structure st = a.get_structure();
    if (a.is_subview() && (st != Structure::DENSE || a.get_data_type() == DataType::BIT)) return false;
    return true;  // lazy closures read through their own typed get (and in bulk, RawRows.hpp)
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
    std::unique_ptr<DenseMatrix<W>> out;
    {
        UninitializedStorage u;  // rows_as writes every element
        out = std::make_unique<DenseMatrix<W>>(a.rows(), a.cols(), std::string());
    }
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
