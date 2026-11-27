#pragma once

// Bulk reads of stored values: rows [i0, i1) of m.get(i, j) (the class's typed accessor:
// transposition applied, view factor and conjugation NOT applied) into a row-major buffer,
// straight from the packed storage. The per-element `get` costs a bounds check, the
// triangle / mirror branches and an offset lookup per element; writing structured operands
// out through it made every structured element-wise op 5-10x slower than NumPy.
//
// The values are exactly those of m.get(i, j): callers apply the view rule
// (math/ViewValue.hpp) to them as before. Returns false for classes without a bulk reader
// (bit storage, complex_float16, identity, closures, windows of structured classes): callers
// then read element by element.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "pycauset/core/Float16.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/matrix/SymmetricMatrix.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"

namespace pycauset {

template <typename T, typename Src>
inline constexpr bool has_raw_rows_v =
    !std::is_same_v<T, bool> &&
    (std::is_same_v<Src, DenseMatrix<T>> || std::is_same_v<Src, TriangularMatrix<T>> ||
     std::is_same_v<Src, SymmetricMatrix<T>> || std::is_same_v<Src, DiagonalMatrix<T>>);

// dst[(i - i0) * cols + j] = f(col(j)[i - first(j)]) for j in [0, jn) and i in
// [max(i0, start(j)), i1), start(j) >= first(j): a
// column gather done 16 columns at a time, so each destination row is written in contiguous
// runs. Writing one column at a time down the rows (stride = a row, 4 KB for 1024 int32)
// mapped every store to the same cache set and ran at memory latency.
template <typename T, typename Col, typename First, typename Start, typename F>
void gather_columns(T* __restrict dst, uint64_t cols, uint64_t i0, uint64_t i1, uint64_t jn, Col col, First first,
                    Start start, F f) {
    constexpr uint64_t B = 16;
    const T* p[B];
    uint64_t lo[B], f0[B];
    for (uint64_t jb = 0; jb < jn; jb += B) {
        const uint64_t w = std::min(B, jn - jb);
        uint64_t lo_min = i1;
        for (uint64_t q = 0; q < w; ++q) {
            p[q] = col(jb + q);
            f0[q] = first(jb + q);
            lo[q] = std::max(i0, start(jb + q));
            lo_min = std::min(lo_min, lo[q]);
        }
        for (uint64_t i = lo_min; i < i1; ++i) {
            T* __restrict out = dst + (i - i0) * cols + jb;
            for (uint64_t q = 0; q < w; ++q)
                if (i >= lo[q]) out[q] = f(p[q][i - f0[q]]);
        }
    }
}

template <typename T, typename Src>
bool raw_rows(const Src& m, uint64_t i0, uint64_t i1, T* __restrict dst) {
    if constexpr (!has_raw_rows_v<T, Src>) {
        return false;
    } else {
        const uint64_t cols = m.cols();
        const uint64_t nr = i1 - i0;
        if constexpr (std::is_same_v<Src, DenseMatrix<T>>) {
            // Windows too: get() reads storage (row_offset + i, col_offset + j), swapped when
            // transposed.
            const uint64_t bc = m.base_cols();
            const T* base = m.data() + m.row_offset() * bc + m.col_offset();
            if (!m.is_transposed()) {
                for (uint64_t i = i0; i < i1; ++i) std::memcpy(dst + (i - i0) * cols, base + i * bc, cols * sizeof(T));
            } else {
                // Element (i, j) is base[j * bc + i]: storage row j read contiguously over i.
                auto zero = [](uint64_t) { return uint64_t{0}; };
                gather_columns(dst, cols, i0, i1, cols, [&](uint64_t j) { return base + j * bc; }, zero, zero,
                               [](T v) { return v; });
            }
            return true;
        } else if constexpr (std::is_same_v<Src, DiagonalMatrix<T>>) {
            if (m.get_structure() != Structure::DIAGONAL || m.is_subview()) return false;
            std::fill(dst, dst + nr * cols, T(0));
            const T* d = m.data();
            for (uint64_t i = i0; i < i1 && i < cols; ++i) dst[(i - i0) * cols + i] = d[i];
            return true;
        } else {
            if (m.is_subview()) return false;
            const auto* bytes = static_cast<const uint8_t*>(m.shared_mapper()->get_data());
            auto stored_row = [&](uint64_t i) { return reinterpret_cast<const T*>(bytes + m.get_row_offset(i)); };
            if constexpr (std::is_same_v<Src, TriangularMatrix<T>>) {
                const bool diag = m.has_diagonal();
                std::fill(dst, dst + nr * cols, T(0));
                if (!m.is_transposed()) {
                    // Row i stores columns [first, n).
                    for (uint64_t i = i0; i < i1; ++i) {
                        const uint64_t first = diag ? i : i + 1;
                        if (first < cols) std::memcpy(dst + (i - i0) * cols + first, stored_row(i), (cols - first) * sizeof(T));
                    }
                } else {
                    // Lower triangle: (i, j) = stored(j, i) for j <= i (j < i strict). Stored
                    // row j holds columns from its first; read it contiguously over i.
                    // (stored(j, i) is stored_row(j)[i - first(j)]).
                    auto first = [diag](uint64_t j) { return diag ? j : j + 1; };
                    gather_columns(dst, cols, i0, i1, std::min(i1, cols), stored_row, first, first,
                                   [](T v) { return v; });
                }
                return true;
            } else {
                // Symmetric family: (i, j) = stored(i, j) for j >= i, mirror(stored(j, i))
                // below (transposition is not applied by get() here either).
                for (uint64_t i = i0; i < i1; ++i)
                    std::memcpy(dst + (i - i0) * cols + i, stored_row(i), (cols - i) * sizeof(T));
                gather_columns(dst, cols, i0, i1, std::min(i1, cols), stored_row, [](uint64_t j) { return j; },
                               [](uint64_t j) { return j + 1; }, [&](T v) { return m.mirror(v); });
                return true;
            }
        }
    }
}

}  // namespace pycauset
