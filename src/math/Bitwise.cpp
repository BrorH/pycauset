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

#include <stdexcept>
#include <string>
#include <type_traits>

#include "pycauset/core/DTypeDispatch.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/core/PromotionResolver.hpp"
#include "pycauset/core/DebugTrace.hpp"
#include "pycauset/math/ScalarOps.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"

namespace pycauset {
namespace {

bool is_int_dtype(DataType d) { return is_integer_or_bit(d) && d != DataType::BIT; }

void require_bitwise_dtype(DataType d, const char* what) {
    if (d != DataType::BIT && !is_int_dtype(d)) {
        throw std::domain_error(std::string(what) + " is defined for bool and integer dtypes (got " + dtype_name(d) + ")");
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

template <typename T>
std::unique_ptr<MatrixBase> int_bitwise(const MatrixBase& a, const MatrixBase& b, BitwiseOp op, DataType r) {
    std::unique_ptr<MatrixBase> ha = plain_dense<T>(a) ? nullptr : to_dense(a, r);
    std::unique_ptr<MatrixBase> hb = plain_dense<T>(b) ? nullptr : to_dense(b, r);
    const T* x = static_cast<const DenseMatrix<T>&>(ha ? *ha : a).data();
    const T* y = static_cast<const DenseMatrix<T>&>(hb ? *hb : b).data();
    auto out = std::make_unique<DenseMatrix<T>>(a.rows(), a.cols(), std::string());
    T* z = out->data();
    const uint64_t total = a.rows() * a.cols();
    debug_trace::set_last("cpu.bitwise.int");
    ParallelFor(0, total, [&](size_t i) { z[i] = apply_op<T>(op, x[i], y[i]); });
    return out;
}

}  // namespace

std::unique_ptr<MatrixBase> bitwise(const MatrixBase& a, const MatrixBase& b, BitwiseOp op) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        throw std::invalid_argument("operands could not be broadcast together (bitwise ops need equal shapes)");
    }
    const DataType da = a.get_data_type(), db = b.get_data_type();
    require_bitwise_dtype(da, "bitwise &, |, ^");
    require_bitwise_dtype(db, "bitwise &, |, ^");

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
        ParallelFor(0, words, [&](size_t w) { pz[w] = apply_op<uint64_t>(op, px[w], py[w]); });
        return out;
    }

    // Signed with uint64: no integer dtype holds both (NumPy raises TypeError here too).
    auto is_signed = [](DataType d) {
        return d == DataType::INT8 || d == DataType::INT16 || d == DataType::INT32 || d == DataType::INT64;
    };
    if ((da == DataType::UINT64 && is_signed(db)) || (db == DataType::UINT64 && is_signed(da))) {
        throw std::domain_error(std::string("bitwise ops: no common integer dtype for ") + dtype_name(da) + " and " +
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
            std::unique_ptr<MatrixBase> h = plain_dense<T>(a) ? nullptr : to_dense(a, d);
            const T* x = static_cast<const DenseMatrix<T>&>(h ? *h : a).data();
            auto out = std::make_unique<DenseMatrix<T>>(a.rows(), a.cols(), std::string());
            T* z = out->data();
            debug_trace::set_last("cpu.bitwise.not_int");
            ParallelFor(0, a.rows() * a.cols(), [&](size_t i) { z[i] = static_cast<T>(~x[i]); });
            return out;
        } else {
            throw std::logic_error("bitwise_not: not an integer dtype");
        }
    });
}

}  // namespace pycauset
