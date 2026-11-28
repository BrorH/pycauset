#pragma once

// Bulk reads of stored values: rows [i0, i1) of m.get(i, j) (the class's typed accessor:
// transposition applied, view factor and conjugation NOT applied) into a row-major buffer,
// straight from the packed storage. The per-element `get` costs a bounds check, the
// triangle / mirror branches and an offset lookup per element; writing structured operands
// out through it made every structured element-wise op 5-10x slower than NumPy.
//
// The values are those of m.get(i, j), except that the antisymmetric diagonal reads +0 (the
// logical value, F-078; get() returns what is stored): callers apply the view rule
// (math/ViewValue.hpp) to them as before. Returns false for classes without a bulk reader
// (bit storage, complex_float16, identity, closures, windows of structured classes): callers
// then read element by element.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "pycauset/core/Float16.hpp"
#include "pycauset/matrix/BitClosureMatrix.hpp"
#include "pycauset/matrix/ComplexFloat16Matrix.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/matrix/SymmetricMatrix.hpp"
#include "pycauset/matrix/TriangularBitMatrix.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"

namespace pycauset {

template <typename T, typename Src>
inline constexpr bool has_raw_rows_v =
    std::is_same_v<Src, BitClosureMatrix>
        ? std::is_same_v<T, int32_t>  // C + C^T / C - C^T over causal bits, read as int32
        : std::is_same_v<Src, ComplexFloat16Matrix>
        ? std::is_same_v<T, std::complex<double>>  // its get() widens the two float16 planes
        : std::is_same_v<T, bool>
        ? (std::is_same_v<Src, DenseMatrix<bool>> || std::is_same_v<Src, TriangularMatrix<bool>>)
        : (std::is_same_v<Src, DenseMatrix<T>> || std::is_same_v<Src, TriangularMatrix<T>> ||
           std::is_same_v<Src, SymmetricMatrix<T>> || std::is_same_v<Src, DiagonalMatrix<T>>);

// Bit storage: bits [b0, b0 + len) of a packed row (64 per word, bit k of word w is element
// 64 w + k) into bytes.
inline void unpack_bits(const uint64_t* __restrict words, uint64_t b0, uint64_t len, bool* __restrict out) {
    uint64_t k = 0;
    while (k < len) {
        const uint64_t b = b0 + k;
        uint64_t w = words[b / 64] >> (b % 64);
        const uint64_t take = std::min<uint64_t>(64 - b % 64, len - k);
        for (uint64_t q = 0; q < take; ++q, w >>= 1) out[k + q] = (w & 1u) != 0;
        k += take;
    }
}
inline bool bit_at(const uint64_t* words, uint64_t b) { return ((words[b / 64] >> (b % 64)) & 1u) != 0; }

// dst[(i - i0) * cols + j] = f(col(j)[i - first(j)]) for j in [0, jn) and i in
// [max(i0, start(j)), i1), start(j) >= first(j): a
// column gather done 16 columns at a time, so each destination row is written in contiguous
// runs. Writing one column at a time down the rows (stride = a row, 4 KB for 1024 int32)
// mapped every store to the same cache set and ran at memory latency.
// Dense case (every column starts at row 0, no mirror): a branch-free blocked transpose.
// Blocks of 8 columns x the tile's rows; the per-element start test of gather_columns below
// halved its speed.
template <typename T>
void gather_dense_columns(T* __restrict dst, uint64_t cols, uint64_t i0, uint64_t i1, const T* base, uint64_t bc) {
    constexpr uint64_t B = 8;  // 8 source rows a row apart: within L1 associativity (16 thrashed)
    for (uint64_t jb = 0; jb < cols; jb += B) {
        const uint64_t w = std::min(B, cols - jb);
        if (w == B) {
            const T* p[B];
            for (uint64_t q = 0; q < B; ++q) p[q] = base + (jb + q) * bc;
            for (uint64_t i = i0; i < i1; ++i) {
                T* __restrict out = dst + (i - i0) * cols + jb;
                for (uint64_t q = 0; q < B; ++q) out[q] = p[q][i];
            }
        } else {
            for (uint64_t q = 0; q < w; ++q) {
                const T* __restrict p = base + (jb + q) * bc;
                for (uint64_t i = i0; i < i1; ++i) dst[(i - i0) * cols + jb + q] = p[i];
            }
        }
    }
}

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
        if constexpr (std::is_same_v<Src, BitClosureMatrix>) {
            // (i, j) = C(i, j) + sign * C(j, i) from C's strict upper bits, as get() (no view).
            const auto& c = m.bits();
            if (m.is_materialized() || m.is_subview() || c.is_transposed() || c.has_diagonal()) return false;
            const auto* bytes = static_cast<const uint8_t*>(c.shared_mapper()->get_data());
            auto srow = [&](uint64_t r) { return reinterpret_cast<const uint64_t*>(bytes + c.get_row_offset(r)); };
            const T sg = static_cast<T>(m.sign());
            std::fill(dst, dst + nr * cols, T(0));
            for (uint64_t i = i0; i < i1; ++i) {  // upper part: row i's own bits (columns > i)
                const uint64_t len = c.row_bits(i);
                if (!len) continue;
                const uint64_t* w = srow(i);
                T* out = dst + (i - i0) * cols + (i + 1);
                for (uint64_t b = 0; b < len; ++b) out[b] = bit_at(w, b) ? T(1) : T(0);
            }
            for (uint64_t j = 0; j < std::min(i1, cols); ++j) {  // lower part: column i of row j
                const uint64_t* w = srow(j);
                for (uint64_t i = std::max(i0, j + 1); i < i1; ++i)
                    if (bit_at(w, i - (j + 1))) dst[(i - i0) * cols + j] = sg;
            }
            return true;
        } else if constexpr (std::is_same_v<Src, ComplexFloat16Matrix>) {
            // Split real / imaginary float16 planes (no windows: the class has none).
            const pycauset::float16_t* re = m.real_data();
            const pycauset::float16_t* im = m.imag_data();
            const uint64_t bc = m.base_cols();
            for (uint64_t i = i0; i < i1; ++i) {
                T* out = dst + (i - i0) * cols;
                if (!m.is_transposed()) {
                    const uint64_t b = i * bc;
                    for (uint64_t j = 0; j < cols; ++j)
                        out[j] = T(static_cast<double>(re[b + j]), static_cast<double>(im[b + j]));
                } else {
                    for (uint64_t j = 0; j < cols; ++j)
                        out[j] = T(static_cast<double>(re[j * bc + i]), static_cast<double>(im[j * bc + i]));
                }
            }
            return true;
        } else if constexpr (std::is_same_v<T, bool>) {
            // Bit storage (no windows: get() of the bit classes does not apply them).
            if (m.is_subview()) return false;
            const auto* bytes = static_cast<const uint8_t*>(m.shared_mapper()->get_data());
            if constexpr (std::is_same_v<Src, DenseMatrix<bool>>) {
                auto srow = [&](uint64_t r) { return reinterpret_cast<const uint64_t*>(bytes + r * m.stride_bytes()); };
                if (!m.is_transposed()) {
                    for (uint64_t i = i0; i < i1; ++i) unpack_bits(srow(i), 0, cols, dst + (i - i0) * cols);
                } else {  // (i, j) is bit i of storage row j
                    for (uint64_t j = 0; j < cols; ++j) {
                        const uint64_t* w = srow(j);
                        for (uint64_t i = i0; i < i1; ++i) dst[(i - i0) * cols + j] = bit_at(w, i);
                    }
                }
            } else {
                auto srow = [&](uint64_t r) { return reinterpret_cast<const uint64_t*>(bytes + m.get_row_offset(r)); };
                std::fill(dst, dst + nr * cols, false);
                if (!m.is_transposed()) {  // row i stores columns [first_col(i), n)
                    for (uint64_t i = i0; i < i1; ++i)
                        if (const uint64_t len = m.row_bits(i))
                            unpack_bits(srow(i), 0, len, dst + (i - i0) * cols + m.first_col(i));
                } else {  // (i, j) is stored(j, i) for i >= first_col(j)
                    for (uint64_t j = 0; j < std::min(i1, cols); ++j) {
                        const uint64_t f = m.first_col(j);
                        const uint64_t* w = srow(j);
                        for (uint64_t i = std::max(i0, f); i < i1; ++i) dst[(i - i0) * cols + j] = bit_at(w, i - f);
                    }
                }
            }
            return true;
        } else if constexpr (std::is_same_v<Src, DenseMatrix<T>>) {
            // Windows too: get() reads storage (row_offset + i, col_offset + j), swapped when
            // transposed.
            const uint64_t bc = m.base_cols();
            const T* base = m.data() + m.row_offset() * bc + m.col_offset();
            if (!m.is_transposed()) {
                for (uint64_t i = i0; i < i1; ++i) std::memcpy(dst + (i - i0) * cols, base + i * bc, cols * sizeof(T));
            } else {
                // Element (i, j) is base[j * bc + i]: storage row j read contiguously over i.
                gather_dense_columns(dst, cols, i0, i1, base, bc);
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
                // The antisymmetric diagonal is +0 whatever is stored (a packed scalar op can
                // store 0 * -1 = -0.0; F-078): the one place this differs from get().
                if (m.kind() == Structure::ANTISYMMETRIC)
                    for (uint64_t i = i0; i < i1; ++i) dst[(i - i0) * cols + i] = T(0);
                gather_columns(dst, cols, i0, i1, std::min(i1, cols), stored_row, [](uint64_t j) { return j; },
                               [](uint64_t j) { return j + 1; }, [&](T v) { return m.mirror(v); });
                return true;
            }
        }
    }
}

}  // namespace pycauset
