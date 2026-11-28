// Bitwise operations for every structure and view (review: Adding Operations protocol).
//
//   entry:      bitwise(a, b, op), bitwise_not(a)
//   result:     bool op bool -> bool; integers: NumPy's promotion (promotion::resolve Add)
//   normaliser: bool result -> plain DenseMatrix<bool> operands (write_out otherwise);
//               integer result -> DenseMatrix<r> operands (to_dense, the shared typed reader)
//   kernels:    words of packed bits (64 elements per operation); one flat loop per integer T
// Before: only dense bool matrices had & | ^ (every integer dtype and every structure raised),
// and ~ went through per-class methods.
#include "pycauset/math/Bitwise.hpp"

#include <algorithm>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "pycauset/core/DTypeDispatch.hpp"
#include "pycauset/core/Errors.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/core/PromotionResolver.hpp"
#include "pycauset/core/DebugTrace.hpp"
#include "pycauset/math/ScalarOps.hpp"
#include "pycauset/math/RawRows.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"

namespace pycauset {
namespace {


void require_bitwise_dtype(DataType d, const char* what) {
    if (d != DataType::BIT && !is_int_dtype(d)) {
        throw dtype_error(std::string(what) + " is defined for bool and integer dtypes (got " + dtype_name(d) + ")");
    }
}

// A plain bit matrix whose storage words can be combined directly (no view, full span).
bool plain_bits(const MatrixBase& m) {
    const auto* b = dynamic_cast<const DenseMatrix<bool>*>(&m);
    return b != nullptr && is_plain(m) && m.rows() == m.base_rows() && m.cols() == m.base_cols();
}

template <typename T>
bool plain_dense(const MatrixBase& m) {
    return dynamic_cast<const DenseMatrix<T>*>(&m) != nullptr && is_plain(m) && m.rows() == m.base_rows() &&
           m.cols() == m.base_cols();
}

template <typename T>
T apply_op(BitwiseOp op, T x, T y) {
    switch (op) {
        case BitwiseOp::And: return static_cast<T>(x & y);
        case BitwiseOp::Or: return static_cast<T>(x | y);
        default: return static_cast<T>(x ^ y);
    }
}

// An operand stored in T with no view, readable in bulk (math/RawRows.hpp): rows are read
// tile by tile, nothing is written out first. Returns false otherwise.
template <typename T>
bool bulk_readable(const MatrixBase& m) {
    if (m.get_scalar() != std::complex<double>(1.0, 0.0) || m.is_conjugated()) return false;
    bool ok = false;
    visit_matrix(m, [&](const auto& typed) {
        using M = std::decay_t<decltype(typed)>;
        using In = std::decay_t<decltype(typed.get(0, 0))>;
        if constexpr (std::is_same_v<In, T> && has_raw_rows_v<T, M>) ok = !m.is_subview() || std::is_same_v<M, DenseMatrix<T>>;
    });
    return ok;
}

template <typename T>
bool read_tile(const MatrixBase& m, uint64_t r0, uint64_t r1, T* dst) {
    bool ok = false;
    visit_matrix(m, [&](const auto& typed) {
        using M = std::decay_t<decltype(typed)>;
        using In = std::decay_t<decltype(typed.get(0, 0))>;
        if constexpr (std::is_same_v<In, T> && has_raw_rows_v<T, M>) ok = raw_rows<T>(typed, r0, r1, dst);
    });
    return ok;
}

template <typename T>
std::unique_ptr<MatrixBase> int_bitwise(const MatrixBase& a, const MatrixBase& b, BitwiseOp op, DataType r) {
    // Plain dense operands are read in place; other bulk-readable ones a tile of rows at a
    // time; the rest (other element types, views) written out once as before.
    const bool pa = plain_dense<T>(a), pb = plain_dense<T>(b);
    const bool ta = !pa && bulk_readable<T>(a), tb = !pb && bulk_readable<T>(b);
    std::unique_ptr<MatrixBase> ha = (pa || ta) ? nullptr : to_dense(a, r);
    std::unique_ptr<MatrixBase> hb = (pb || tb) ? nullptr : to_dense(b, r);
    const T* x = ta ? nullptr : static_cast<const DenseMatrix<T>&>(ha ? *ha : a).data();
    const T* y = tb ? nullptr : static_cast<const DenseMatrix<T>&>(hb ? *hb : b).data();
    std::unique_ptr<DenseMatrix<T>> out;
    {
        UninitializedStorage u;  // every element is written below
        out = std::make_unique<DenseMatrix<T>>(a.rows(), a.cols(), std::string());
    }
    T* z = out->data();
    const uint64_t rows = a.rows(), cols = a.cols();
    debug_trace::set_last("cpu.bitwise.int");
    // The operator chosen outside the loop: a switch per element kept it scalar.
    auto run = [&](auto f) {
        const uint64_t tile = std::clamp<uint64_t>(rows / 32, 1, 32);
        std::atomic<bool> ok{true};
        ParallelFor(0, (rows + tile - 1) / tile, [&](size_t t) {
            const uint64_t r0 = t * tile, r1 = std::min(rows, r0 + tile), len = (r1 - r0) * cols;
            std::unique_ptr<T[]> bx, by;
            const T* xs = x ? x + r0 * cols : nullptr;
            const T* ys = y ? y + r0 * cols : nullptr;
            if (!xs) { bx.reset(new T[len]); if (!read_tile<T>(a, r0, r1, bx.get())) { ok = false; return; } xs = bx.get(); }
            if (!ys) { by.reset(new T[len]); if (!read_tile<T>(b, r0, r1, by.get())) { ok = false; return; } ys = by.get(); }
            const T* __restrict xr = xs;
            const T* __restrict yr = ys;
            T* __restrict zr = z + r0 * cols;
            for (uint64_t i = 0; i < len; ++i) zr[i] = static_cast<T>(f(xr[i], yr[i]));
        });
        if (!ok) throw std::logic_error("bitwise: a bulk-readable operand could not be read");
    };
    switch (op) {
        case BitwiseOp::And: run([](T p, T q) { return p & q; }); break;
        case BitwiseOp::Or: run([](T p, T q) { return p | q; }); break;
        default: run([](T p, T q) { return p ^ q; }); break;
    }
    return out;
}

// m repeated to rows x cols along its length-1 axes (NumPy broadcasting), dense, m's dtype.
std::unique_ptr<MatrixBase> broadcast_copy(const MatrixBase& m, uint64_t rows, uint64_t cols) {
    const DataType dt = m.get_data_type();
    const auto h = dt == DataType::BIT ? write_out(m) : to_dense(m, dt);
    return visit_dtype(dt, [&](auto tag) -> std::unique_ptr<MatrixBase> {
        using T = typename decltype(tag)::type;
        if constexpr (std::is_integral_v<T>) {
            const auto& s = static_cast<const DenseMatrix<T>&>(*h);
            auto out = std::make_unique<DenseMatrix<T>>(rows, cols, std::string());
            const bool r1 = m.rows() == 1, c1 = m.cols() == 1;
            for (uint64_t i = 0; i < rows; ++i)
                for (uint64_t j = 0; j < cols; ++j) out->set(i, j, s.get(r1 ? 0 : i, c1 ? 0 : j));
            return out;
        } else {
            throw std::logic_error("bitwise broadcast: not an integer dtype");
        }
    });
}

}  // namespace

std::unique_ptr<MatrixBase> bitwise(const MatrixBase& a, const MatrixBase& b, BitwiseOp op) {
    const DataType da = a.get_data_type(), db = b.get_data_type();
    require_bitwise_dtype(da, "bitwise &, |, ^");
    require_bitwise_dtype(db, "bitwise &, |, ^");
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        // NumPy's broadcasting, as + - * / and the comparisons (review F-128): the operand with a
        // length-1 axis is repeated, then the equal-shape kernels run.
        auto dim = [](uint64_t x, uint64_t y) -> uint64_t {
            if (x == y || y == 1) return x;
            if (x == 1) return y;
            throw std::invalid_argument("operands could not be broadcast together (bitwise ops: shapes " +
                                        std::to_string(x) + " vs " + std::to_string(y) + ")");
        };
        const uint64_t rows = dim(a.rows(), b.rows()), cols = dim(a.cols(), b.cols());
        const auto ea = (a.rows() == rows && a.cols() == cols) ? nullptr : broadcast_copy(a, rows, cols);
        const auto eb = (b.rows() == rows && b.cols() == cols) ? nullptr : broadcast_copy(b, rows, cols);
        return bitwise(ea ? *ea : a, eb ? *eb : b, op);
    }

    if (da == DataType::BIT && db == DataType::BIT) {
        std::unique_ptr<MatrixBase> ha = plain_bits(a) ? nullptr : write_out(a);
        std::unique_ptr<MatrixBase> hb = plain_bits(b) ? nullptr : write_out(b);
        const auto& x = static_cast<const DenseMatrix<bool>&>(ha ? *ha : a);
        const auto& y = static_cast<const DenseMatrix<bool>&>(hb ? *hb : b);
        auto out = std::make_unique<DenseMatrix<bool>>(a.rows(), a.cols(), std::string());
        const uint64_t* px = x.data();
        const uint64_t* py = y.data();
        uint64_t* pz = out->data();
        // Same shape, same stride: padding bits of both are zero, so they stay zero for & | ^.
        const uint64_t words = x.stride_bytes() * x.rows() / 8;
        debug_trace::set_last("cpu.bitwise.bits");
        ParallelForRange(0, words, [&](size_t w0, size_t w1) {
            const uint64_t* __restrict xr = px;
            const uint64_t* __restrict yr = py;
            uint64_t* __restrict zr = pz;
            for (size_t w = w0; w < w1; ++w) zr[w] = apply_op<uint64_t>(op, xr[w], yr[w]);
        });
        return out;
    }

    // Signed with uint64: no integer dtype holds both (NumPy raises TypeError here too).
    auto is_signed = [](DataType d) {
        return d == DataType::INT8 || d == DataType::INT16 || d == DataType::INT32 || d == DataType::INT64;
    };
    if ((da == DataType::UINT64 && is_signed(db)) || (db == DataType::UINT64 && is_signed(da))) {
        throw dtype_error(std::string("bitwise ops: no common integer dtype for ") + dtype_name(da) + " and " +
                                dtype_name(db));
    }
    const DataType r = promotion::resolve(promotion::BinaryOp::Add, da, db).result_dtype;
    return visit_dtype(r, [&](auto tag) -> std::unique_ptr<MatrixBase> {
        using T = typename decltype(tag)::type;
        if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
            return int_bitwise<T>(a, b, op, r);
        } else {
            throw std::logic_error("bitwise: integer promotion gave " + std::string(dtype_name(r)));
        }
    });
}

std::unique_ptr<MatrixBase> bitwise_not(const MatrixBase& a) {
    const DataType d = a.get_data_type();
    require_bitwise_dtype(d, "~ (bitwise not)");
    if (d == DataType::BIT) {
        std::unique_ptr<MatrixBase> h = plain_bits(a) ? nullptr : write_out(a);
        debug_trace::set_last("cpu.bitwise.not_bits");
        return static_cast<const DenseMatrix<bool>&>(h ? *h : a).bitwise_not("");  // masks row padding
    }
    return visit_dtype(d, [&](auto tag) -> std::unique_ptr<MatrixBase> {
        using T = typename decltype(tag)::type;
        if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
            const bool p = plain_dense<T>(a), tiled = !p && bulk_readable<T>(a);
            std::unique_ptr<MatrixBase> h = (p || tiled) ? nullptr : to_dense(a, d);
            const T* x = tiled ? nullptr : static_cast<const DenseMatrix<T>&>(h ? *h : a).data();
            std::unique_ptr<DenseMatrix<T>> out;
            {
                UninitializedStorage u;  // every element is written below
                out = std::make_unique<DenseMatrix<T>>(a.rows(), a.cols(), std::string());
            }
            T* z = out->data();
            debug_trace::set_last("cpu.bitwise.not_int");
            const uint64_t rows = a.rows(), cols = a.cols();
            const uint64_t tile = std::clamp<uint64_t>(rows / 32, 1, 32);
            std::atomic<bool> ok{true};
            ParallelFor(0, (rows + tile - 1) / tile, [&](size_t t) {
                const uint64_t r0 = t * tile, r1 = std::min(rows, r0 + tile), len = (r1 - r0) * cols;
                std::unique_ptr<T[]> buf;
                const T* xs = x ? x + r0 * cols : nullptr;
                if (!xs) { buf.reset(new T[len]); if (!read_tile<T>(a, r0, r1, buf.get())) { ok = false; return; } xs = buf.get(); }
                const T* __restrict xr = xs;
                T* __restrict zr = z + r0 * cols;
                for (uint64_t i = 0; i < len; ++i) zr[i] = static_cast<T>(~xr[i]);
            });
            if (!ok) throw std::logic_error("bitwise_not: a bulk-readable operand could not be read");
            return out;
        } else {
            throw std::logic_error("bitwise_not: not an integer dtype");
        }
    });
}

// a + b (sign +1) or a - b (sign -1) for two bool matrices of equal shape: int32 counts (bit
// arithmetic counts, F-013), straight from the packed words (both operands as plain bits).
std::unique_ptr<MatrixBase> bit_counts(const MatrixBase& a, const MatrixBase& b, int sign) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) throw std::invalid_argument("bit_counts: equal shapes");
    std::unique_ptr<MatrixBase> ha = plain_bits(a) ? nullptr : write_out(a);
    std::unique_ptr<MatrixBase> hb = plain_bits(b) ? nullptr : write_out(b);
    const auto& x = static_cast<const DenseMatrix<bool>&>(ha ? *ha : a);
    const auto& y = static_cast<const DenseMatrix<bool>&>(hb ? *hb : b);
    const uint64_t rows = a.rows(), cols = a.cols(), stride = x.stride_bytes() / 8;
    auto out = std::make_unique<DenseMatrix<int32_t>>(rows, cols, std::string());
    int32_t* z = out->data();
    const uint64_t* px = x.data();
    const uint64_t* py = y.data();
    debug_trace::set_last("cpu.bitwise.counts");
    ParallelFor(0, rows, [&](size_t i) {
        const uint64_t* wx = px + i * stride;
        const uint64_t* wy = py + i * stride;
        int32_t* row = z + i * cols;
        for (uint64_t j0 = 0; j0 < cols; j0 += 64) {
            const uint64_t u = wx[j0 / 64], v = wy[j0 / 64];
            const uint64_t len = std::min<uint64_t>(64, cols - j0);
            for (uint64_t k = 0; k < len; ++k)
                row[j0 + k] = static_cast<int32_t>((u >> k) & 1u) + sign * static_cast<int32_t>((v >> k) & 1u);
        }
    });
    return out;
}

}  // namespace pycauset
