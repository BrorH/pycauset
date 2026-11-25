#include "pycauset/math/Reductions.hpp"

#include <bit>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "pycauset/core/DTypeDispatch.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/core/ScalarUtils.hpp"
#include "pycauset/math/VectorOps.hpp"
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

// Number of set bits of a causal (strictly upper triangular) bit matrix: row popcounts.
__int128 count_ones(const TriangularMatrix<bool>& c) {
    const uint64_t n = c.base_rows();
    const auto* base = reinterpret_cast<const char*>(c.data());
    return sum_rows<__int128>(n, [&](uint64_t i) -> __int128 {
        const uint64_t len = n - 1 - i;  // bits in row i
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

ReductionValue sum_generic(const MatrixBase& m) {
    ReductionValue out;
    const uint64_t rows = m.rows();
    if (is_complex_dtype(m.get_data_type())) {
        out.kind = ReductionValue::Kind::Complex;
        out.c = sum_rows<std::complex<double>>(rows, [&](uint64_t i) {
            std::complex<double> acc{0.0, 0.0};
            for_row(m, i, [&](uint64_t j) { acc += m.get_element_as_complex(i, j); });
            return acc;
        }, std::complex<double>(0.0, 0.0));
    } else {
        out.kind = ReductionValue::Kind::Real;
        out.r = sum_rows<double>(rows, [&](uint64_t i) {
            double acc = 0.0;
            for_row(m, i, [&](uint64_t j) { acc += m.get_element_as_double(i, j); });
            return acc;
        }, 0.0);
    }
    return out;
}

}  // namespace

ReductionValue sum(const MatrixBase& m) {
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
    if (dt == DataType::BIT && f == std::complex<double>(1.0, 0.0) && !m.has_view_offset() &&
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
    if ((int_like && !m.is_conjugated() && scalar::is_exact_int_factor(f)) ||
        (dt == DataType::BIT && f == std::complex<double>(1.0, 0.0))) {
        const int64_t k = int_like ? static_cast<int64_t>(f.real()) : 1;
        out.kind = ReductionValue::Kind::Int;
        out.i = visit_matrix(m, [&](const auto& typed) -> __int128 {
            using T = std::decay_t<decltype(typed.get(0, 0))>;
            if constexpr (!std::is_integral_v<T>) {
                throw std::logic_error("sum: integer path on a non-integer matrix");
            } else {
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
                const uint64_t cols = m.cols();
                const bool contiguous = std::is_same_v<M, DenseMatrix<T>> && !m.is_transposed() &&
                                        !m.has_view_offset() && cols == m.base_cols();
                return k * sum_rows<double>(m.rows(), [&](uint64_t i) {
                    double acc = 0.0;
                    if constexpr (std::is_same_v<M, DenseMatrix<T>>) {
                        if (contiguous) {
                            const T* row = typed.data() + i * cols;
                            for (uint64_t j = 0; j < cols; ++j) acc += row[j];
                            return acc;
                        }
                    }
                    for_row(m, i, [&](uint64_t j) { acc += typed.get(i, j); });
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

ReductionValue sum(const VectorBase& v) {
    auto row = as_row_matrix(v);
    return sum(*row);
}

}  // namespace pycauset
