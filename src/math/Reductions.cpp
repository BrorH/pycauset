#include "pycauset/math/DerivedCache.hpp"
#include "pycauset/math/Reductions.hpp"

#include <algorithm>
#include <atomic>
#include <memory>
#include <optional>
#include <bit>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "pycauset/core/DTypeDispatch.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/core/ScalarUtils.hpp"
#include "pycauset/math/VectorOps.hpp"
#include "pycauset/math/RawRows.hpp"
#include "pycauset/math/ViewRead.hpp"
#include "pycauset/matrix/BitClosureMatrix.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/matrix/TriangularBitMatrix.hpp"
#include "pycauset/vector/VectorBase.hpp"

namespace pycauset {
namespace {

bool is_complex_dtype(DataType d) {
    return d == DataType::COMPLEX_FLOAT16 || d == DataType::COMPLEX_FLOAT32 || d == DataType::COMPLEX_FLOAT64;
}
bool is_float_dtype(DataType d) {
    return d == DataType::FLOAT16 || d == DataType::FLOAT32 || d == DataType::FLOAT64;
}

__int128 checked_add(__int128 a, __int128 b) {
    __int128 out;
    if (__builtin_add_overflow(a, b, &out)) throw std::overflow_error("sum: integer result does not fit 128 bits");
    return out;
}

// Sum of per-row partials, rows in parallel, combined in row order (deterministic).
template <typename Acc, typename RowFn>
Acc sum_rows(uint64_t rows, RowFn&& row_sum, Acc zero) {
    std::vector<Acc> partial(rows, zero);
    ParallelFor(0, rows, [&](size_t i) { partial[i] = row_sum(static_cast<uint64_t>(i)); });
    Acc total = zero;
    for (const Acc& p : partial) {
        if constexpr (std::is_same_v<Acc, __int128>) total = checked_add(total, p);
        else total += p;
    }
    return total;
}

// Number of set bits of a bit triangle (strict or with diagonal): row popcounts.
__int128 count_ones(const TriangularMatrix<bool>& c) {
    const uint64_t n = c.base_rows();
    const auto* base = reinterpret_cast<const char*>(c.data());
    return sum_rows<__int128>(n, [&](uint64_t i) -> __int128 {
        const uint64_t len = c.row_bits(i);  // bits in row i
        if (len == 0) return 0;
        const uint64_t* w = reinterpret_cast<const uint64_t*>(base + c.get_row_offset(i));
        const uint64_t words = (len + 63) / 64;
        uint64_t cnt = 0;
        for (uint64_t k = 0; k + 1 < words; ++k) cnt += std::popcount(w[k]);
        const uint64_t tail = len % 64;
        cnt += std::popcount(tail ? (w[words - 1] & ((1ULL << tail) - 1)) : w[words - 1]);
        return cnt;
    }, __int128(0));
}

__int128 count_ones(const DenseMatrix<bool>& m) {
    const uint64_t rows = m.base_rows(), cols = m.base_cols();
    const auto* base = reinterpret_cast<const char*>(m.data());
    return sum_rows<__int128>(rows, [&](uint64_t i) -> __int128 {
        const uint64_t* w = reinterpret_cast<const uint64_t*>(base + i * m.stride_bytes());
        const uint64_t words = (cols + 63) / 64;
        uint64_t cnt = 0;
        for (uint64_t k = 0; k < words; ++k) {
            const uint64_t tail = (k + 1 == words) ? cols % 64 : 0;
            cnt += std::popcount(tail ? (w[k] & ((1ULL << tail) - 1)) : w[k]);
        }
        return cnt;
    }, __int128(0));
}

// Sum of len stored integers, each scaled by k with NumPy's wrap. Small types with k == 1
// accumulate in int64 (|x| < 2^32 and len < 2^31: exact; it vectorizes), others in 128 bits.
template <typename T>
__int128 chunk_sum_int(const T* __restrict p, uint64_t len, int64_t k) {
    if (k == 1 && sizeof(T) <= 4 && len < (uint64_t{1} << 31)) {
        int64_t a0 = 0, a1 = 0;
        uint64_t i = 0;
        for (; i + 2 <= len; i += 2) { a0 += static_cast<int64_t>(p[i]); a1 += static_cast<int64_t>(p[i + 1]); }
        if (i < len) a0 += static_cast<int64_t>(p[i]);
        return static_cast<__int128>(a0) + a1;
    }
    __int128 acc = 0;
    for (uint64_t i = 0; i < len; ++i) acc += k == 1 ? static_cast<__int128>(p[i]) : scalar::wrap_scale<T>(p[i], k);
    return acc;  // len < 2^40 values below 2^64: no overflow
}

// Sum of len stored floats in double, four accumulators (one would serialise on the add).
template <typename T>
double chunk_sum_real(const T* __restrict p, uint64_t len) {
    double a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    uint64_t i = 0;
    for (; i + 4 <= len; i += 4) {
        a0 += static_cast<double>(p[i]); a1 += static_cast<double>(p[i + 1]);
        a2 += static_cast<double>(p[i + 2]); a3 += static_cast<double>(p[i + 3]);
    }
    for (; i < len; ++i) a0 += static_cast<double>(p[i]);
    return (a0 + a1) + (a2 + a3);
}

// The same with each value times the view factor k first, as NumPy sums the scaled elements:
// k * sum overflowed where the elements did not and hid an inf - inf (review 3.6 C-5).
template <typename T>
double chunk_sum_real_scaled(const T* __restrict p, uint64_t len, double k) {
    double a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    uint64_t i = 0;
    for (; i + 4 <= len; i += 4) {
        a0 += static_cast<double>(p[i]) * k; a1 += static_cast<double>(p[i + 1]) * k;
        a2 += static_cast<double>(p[i + 2]) * k; a3 += static_cast<double>(p[i + 3]) * k;
    }
    for (; i < len; ++i) a0 += static_cast<double>(p[i]) * k;
    return (a0 + a1) + (a2 + a3);
}

// Sum over every logical element of a dense / triangular / symmetric-family matrix of its
// stored values (view factor not applied): flat over the storage of a full dense matrix
// (element order is irrelevant to a sum: transposes too), else a tile of rows at a time through
// the bulk reader (math/RawRows.hpp). Chunk results combine in chunk order (deterministic).
// nullopt when the class has no bulk reader.
template <typename Acc, typename T, typename M, typename Chunk, typename Combine>
std::optional<Acc> bulk_sum(const M& typed, const MatrixBase& m, Chunk chunk, Combine combine) {
    if constexpr (!has_raw_rows_v<T, M>) {
        return std::nullopt;
    } else {
        const Structure st = m.get_structure();
        if (st == Structure::DIAGONAL || st == Structure::IDENTITY) return std::nullopt;
        const uint64_t rows = m.rows(), cols = m.cols();
        std::vector<Acc> part;
        if constexpr (std::is_same_v<M, DenseMatrix<T>>) {
            if (!m.is_subview() && rows * cols == m.base_rows() * m.base_cols()) {
                const T* p = typed.data();
                const uint64_t total = rows * cols;
                constexpr uint64_t B = uint64_t{1} << 14;
                part.assign((total + B - 1) / B, Acc{});
                ParallelFor(0, part.size(), [&](size_t c) {
                    const uint64_t b = c * B;
                    part[c] = chunk(p + b, std::min(total, b + B) - b);
                });
                Acc acc{};
                for (const Acc& x : part) acc = combine(acc, x);
                return acc;
            }
        }
        const uint64_t tile = std::clamp<uint64_t>(rows / 32, 1, 32);
        part.assign((rows + tile - 1) / tile, Acc{});
        std::atomic<bool> ok{true};
        ParallelFor(0, part.size(), [&](size_t t) {
            const uint64_t r0 = t * tile, r1 = std::min(rows, r0 + tile);
            std::unique_ptr<T[]> raw(new T[(r1 - r0) * cols]);
            if (!raw_rows<T>(typed, r0, r1, raw.get())) { ok = false; return; }
            part[t] = chunk(raw.get(), (r1 - r0) * cols);
        });
        if (!ok) return std::nullopt;
        Acc acc{};
        for (const Acc& x : part) acc = combine(acc, x);
        return acc;
    }
}

// int32 value k*x with NumPy's wrap (the closure's logical dtype).
__int128 wrap32(int64_t k, int32_t x) { return scalar::wrap_scale<int32_t>(x, k); }

// Elements to visit in row i: all columns, or just the diagonal for diagonal/identity.
template <typename F>
void for_row(const MatrixBase& m, uint64_t i, F&& f) {
    const Structure s = m.get_structure();
    if (s == Structure::DIAGONAL || s == Structure::IDENTITY) {
        if (i < m.cols()) f(i);
        return;
    }
    for (uint64_t j = 0; j < m.cols(); ++j) f(j);
}

// Real or complex sum through the typed row reader (ViewRead.hpp), a block of rows at a time;
// rows summed in order, row sums added in row order (deterministic). Diagonal and identity
// read their diagonal only. The virtual accessor is left for what the reader cannot read
// (windows of structured classes, lazy closures).
template <typename W>
W sum_values(const MatrixBase& m) {
    const uint64_t rows = m.rows(), cols = m.cols();
    const Structure s = m.get_structure();
    if (s == Structure::DIAGONAL || s == Structure::IDENTITY) {
        std::vector<W> d(std::min(rows, cols));
        if (read_diagonal(m, d.data())) {
            W acc{};
            for (const W& x : d) acc += x;
            return acc;
        }
    } else {
        const uint64_t block = std::max<uint64_t>(1, (uint64_t{1} << 18) / std::max<uint64_t>(cols, 1));
        std::vector<W> buf(std::min(block, rows) * cols);
        W total{};
        bool ok = true;
        for (uint64_t i0 = 0; i0 < rows && ok; i0 += block) {
            const uint64_t i1 = std::min(rows, i0 + block);
            if (!(ok = read_rows(m, i0, i1, buf.data()))) break;
            for (uint64_t i = i0; i < i1; ++i) {
                W acc{};
                const W* row = buf.data() + (i - i0) * cols;
                for (uint64_t j = 0; j < cols; ++j) acc += row[j];
                total += acc;
            }
        }
        if (ok) return total;
    }
    return sum_rows<W>(rows, [&](uint64_t i) {
        W acc{};
        for_row(m, i, [&](uint64_t j) {
            if constexpr (std::is_same_v<W, double>) acc += m.get_element_as_double(i, j);
            else acc += m.get_element_as_complex(i, j);
        });
        return acc;
    }, W{});
}

ReductionValue sum_generic(const MatrixBase& m) {
    ReductionValue out;
    if (is_complex_dtype(m.get_data_type()) || m.get_scalar().imag() != 0.0) {
        out.kind = ReductionValue::Kind::Complex;
        out.c = sum_values<std::complex<double>>(m);
    } else {
        out.kind = ReductionValue::Kind::Real;
        out.r = sum_values<double>(m);
    }
    return out;
}

}  // namespace

static ReductionValue sum_compute(const MatrixBase& m) {
    const DataType dt = m.get_data_type();
    const std::complex<double> f = m.get_scalar();
    const bool int_like = dt != DataType::BIT && !is_float_dtype(dt) && !is_complex_dtype(dt);
    ReductionValue out;

    if (m.rows() == 0 || m.cols() == 0) {
        out.kind = (int_like || dt == DataType::BIT) ? ReductionValue::Kind::Int
                 : is_complex_dtype(dt) ? ReductionValue::Kind::Complex : ReductionValue::Kind::Real;
        return out;
    }

    // C +- C^T closures: 1s at 2|C| places (sum) or +1 and -1 at |C| places each (difference).
    if (const auto* c = dynamic_cast<const BitClosureMatrix*>(&m); c && !c->is_materialized() &&
                                                                  scalar::is_exact_int_factor(f)) {
        const int64_t k = static_cast<int64_t>(f.real());
        const __int128 ones = count_ones(c->bits());
        out.kind = ReductionValue::Kind::Int;
        out.i = c->sign() > 0 ? 2 * ones * wrap32(k, 1) : ones * (wrap32(k, 1) + wrap32(k, -1));
        return out;
    }

    // Bit storage: popcount when the storage is the plain full matrix.
    if (dt == DataType::BIT && f == std::complex<double>(1.0, 0.0) && !m.is_subview() &&
        m.rows() * m.cols() == m.base_rows() * m.base_cols()) {
        if (const auto* t = dynamic_cast<const TriangularMatrix<bool>*>(&m)) {
            out.kind = ReductionValue::Kind::Int;
            out.i = count_ones(*t);
            return out;
        }
        if (const auto* d = dynamic_cast<const DenseMatrix<bool>*>(&m)) {
            out.kind = ReductionValue::Kind::Int;
            out.i = count_ones(*d);
            return out;
        }
    }

    // Integer / bit: exact, typed reads.
    if ((int_like && scalar::is_exact_int_factor(f)) ||  // conj of an integer is a no-op (3.6 C-6)
        (dt == DataType::BIT && f == std::complex<double>(1.0, 0.0))) {
        const int64_t k = int_like ? static_cast<int64_t>(f.real()) : 1;
        out.kind = ReductionValue::Kind::Int;
        out.i = visit_matrix(m, [&](const auto& typed) -> __int128 {
            using T = std::decay_t<decltype(typed.get(0, 0))>;
            if constexpr (!std::is_integral_v<T>) {
                throw std::logic_error("sum: integer path on a non-integer matrix");
            } else {
                if constexpr (!std::is_same_v<T, bool>) {
                    using M = std::decay_t<decltype(typed)>;
                    if (auto r = bulk_sum<__int128, T, M>(
                            typed, m, [k](const T* p, uint64_t len) { return chunk_sum_int<T>(p, len, k); },
                            [](__int128 a, __int128 b) { return checked_add(a, b); }))
                        return *r;
                }
                return sum_rows<__int128>(m.rows(), [&](uint64_t i) -> __int128 {
                    __int128 acc = 0;
                    for_row(m, i, [&](uint64_t j) {
                        if constexpr (std::is_same_v<T, bool>) acc += typed.get(i, j) ? 1 : 0;
                        else acc += scalar::wrap_scale<T>(typed.get(i, j), k);
                    });
                    return acc;
                }, __int128(0));
            }
        });
        return out;
    }

    // Real float32/float64 with a real factor: typed reads (contiguous rows when plain dense).
    if ((dt == DataType::FLOAT32 || dt == DataType::FLOAT64) && f.imag() == 0.0) {
        const double k = f.real();
        out.kind = ReductionValue::Kind::Real;
        out.r = visit_matrix(m, [&](const auto& typed) -> double {
            using M = std::decay_t<decltype(typed)>;
            using T = std::decay_t<decltype(typed.get(0, 0))>;
            if constexpr (!std::is_floating_point_v<T>) {
                throw std::logic_error("sum: float path on a non-float matrix");
            } else {
                if (auto r = bulk_sum<double, T, M>(
                        typed, m,
                        [k](const T* p, uint64_t len) {
                            return k == 1.0 ? chunk_sum_real<T>(p, len) : chunk_sum_real_scaled<T>(p, len, k);
                        },
                        [](double a, double b) { return a + b; }))
                    return *r;
                const uint64_t cols = m.cols();
                const bool contiguous = std::is_same_v<M, DenseMatrix<T>> && !m.is_transposed() &&
                                        !m.is_subview() && cols == m.base_cols();
                return sum_rows<double>(m.rows(), [&](uint64_t i) {
                    double acc = 0.0;
                    if constexpr (std::is_same_v<M, DenseMatrix<T>>) {
                        if (contiguous) {
                            const T* row = typed.data() + i * cols;
                            for (uint64_t j = 0; j < cols; ++j) acc += static_cast<double>(row[j]) * k;
                            return acc;
                        }
                    }
                    for_row(m, i, [&](uint64_t j) { acc += static_cast<double>(typed.get(i, j)) * k; });
                    return acc;
                }, 0.0);
            }
        });
        return out;
    }

    // float16 and complex: the virtual accessor applies every view.
    // Integer matrices with a non-integer factor are real-valued: float result.
    return sum_generic(m);
}

ReductionValue sum(const MatrixBase& m) {
    return cached<ReductionValue>(m, "sum", "", [&] { return sum_compute(m); });
}

static ReductionValue trace_compute(const MatrixBase& m) {
    const uint64_t n = std::min(m.rows(), m.cols());
    const DataType dt = m.get_data_type();
    const std::complex<double> f = m.get_scalar();
    const bool int_like = dt != DataType::BIT && !is_float_dtype(dt) && !is_complex_dtype(dt);
    ReductionValue out;
    if ((int_like && scalar::is_exact_int_factor(f)) ||  // conj of an integer is a no-op (3.6 C-6)
        (dt == DataType::BIT && f == std::complex<double>(1.0, 0.0))) {
        const int64_t k = int_like ? static_cast<int64_t>(f.real()) : 1;
        out.kind = ReductionValue::Kind::Int;
        out.i = visit_matrix(m, [&](const auto& typed) -> __int128 {
            using T = std::decay_t<decltype(typed.get(0, 0))>;
            if constexpr (!std::is_integral_v<T>) {
                throw std::logic_error("trace: integer path on a non-integer matrix");
            } else {
                __int128 acc = 0;
                for (uint64_t i = 0; i < n; ++i) {
                    if constexpr (std::is_same_v<T, bool>) acc += typed.get(i, i) ? 1 : 0;
                    else acc = checked_add(acc, scalar::wrap_scale<T>(typed.get(i, i), k));
                }
                return acc;
            }
        });
        return out;
    }
    // Real or complex: the typed diagonal reader (the virtual accessor only where it cannot read).
    auto diag_sum = [&](auto zero, auto virt) {
        using W = decltype(zero);
        std::vector<W> d(n);
        W acc = zero;
        if (read_diagonal(m, d.data())) {
            for (const W& x : d) acc += x;
        } else {
            for (uint64_t i = 0; i < n; ++i) acc += virt(i);
        }
        return acc;
    };
    if (is_complex_dtype(dt) || f.imag() != 0.0) {
        out.kind = ReductionValue::Kind::Complex;
        out.c = diag_sum(std::complex<double>(0.0, 0.0), [&](uint64_t i) { return m.get_element_as_complex(i, i); });
        return out;
    }
    out.kind = ReductionValue::Kind::Real;
    out.r = diag_sum(0.0, [&](uint64_t i) { return m.get_element_as_double(i, i); });
    return out;
}

ReductionValue trace(const MatrixBase& m) {
    return cached<ReductionValue>(m, "trace", "", [&] { return trace_compute(m); });
}

ReductionValue sum(const VectorBase& v) {
    auto row = as_row_matrix(v);
    return sum(*row);
}

}  // namespace pycauset
