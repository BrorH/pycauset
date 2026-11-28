// Matrix ⊕ scalar operations with NumPy result-dtype semantics. See ScalarOps.hpp.
#include "pycauset/core/DebugTrace.hpp"
#include "pycauset/math/ScalarOps.hpp"
#include "pycauset/math/Bitwise.hpp"
#include "pycauset/math/RawRows.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <atomic>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "pycauset/core/DTypeDispatch.hpp"
#include "pycauset/core/PromotionResolver.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/core/ObjectFactory.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include "pycauset/core/ScalarUtils.hpp"
#include "pycauset/matrix/ComplexFloat16Matrix.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/matrix/TriangularBitMatrix.hpp"
#include "pycauset/matrix/SymmetricMatrix.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/math/ViewRead.hpp"
#include "pycauset/math/VectorOps.hpp"
#include "pycauset/vector/VectorBase.hpp"
#include <cstring>
#include <utility>

namespace pycauset {

// ----------------------------------------------------------------- ScalarArg
ScalarArg ScalarArg::from_int64(int64_t v) {
    ScalarArg s;
    s.kind = Kind::Int;
    s.i = v;
    s.negative = v < 0;
    s.u = v < 0 ? 0 : static_cast<uint64_t>(v);
    return s;
}
ScalarArg ScalarArg::from_uint64(uint64_t v) {
    ScalarArg s;
    s.kind = Kind::Int;
    s.u = v;
    s.negative = false;
    s.i = v <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ? static_cast<int64_t>(v) : 0;
    return s;
}
ScalarArg ScalarArg::from_double(double v) {
    ScalarArg s;
    s.kind = Kind::Float;
    s.f = v;
    return s;
}
ScalarArg ScalarArg::from_complex(std::complex<double> v) {
    ScalarArg s;
    s.kind = Kind::Complex;
    s.c = v;
    return s;
}
std::complex<double> ScalarArg::as_complex() const {
    switch (kind) {
        case Kind::Int:
            return {negative ? static_cast<double>(i) : static_cast<double>(u), 0.0};
        case Kind::Float:
            return {f, 0.0};
        case Kind::Complex:
            return c;
    }
    return {0.0, 0.0};
}

// ------------------------------------------------------------ dtype helpers
namespace {

DataType complex_of(DataType real_float) {
    switch (real_float) {
        case DataType::FLOAT16: return DataType::COMPLEX_FLOAT16;
        case DataType::FLOAT32: return DataType::COMPLEX_FLOAT32;
        default: return DataType::COMPLEX_FLOAT64;
    }
}
int int_bits(DataType d) {
    switch (d) {
        case DataType::INT8: case DataType::UINT8: return 8;
        case DataType::INT16: case DataType::UINT16: return 16;
        case DataType::INT32: case DataType::UINT32: return 32;
        case DataType::INT64: case DataType::UINT64: return 64;
        default: return 0;
    }
}
bool int_fits(const ScalarArg& s, DataType d) {
    if (d == DataType::BIT) d = DataType::INT32;  // bit arithmetic counts in int32
    const int bits = int_bits(d);
    if (is_signed_int_dtype(d)) {
        const int64_t lo = bits == 64 ? std::numeric_limits<int64_t>::min() : -(int64_t(1) << (bits - 1));
        const int64_t hi = bits == 64 ? std::numeric_limits<int64_t>::max() : (int64_t(1) << (bits - 1)) - 1;
        if (s.negative) return s.i >= lo;
        return s.u <= static_cast<uint64_t>(hi);
    }
    if (s.negative) return false;
    return bits == 64 || s.u <= ((uint64_t(1) << bits) - 1);
}

}  // namespace


static DataType real_of(DataType c) {
    return c == DataType::COMPLEX_FLOAT16 ? DataType::FLOAT16
         : c == DataType::COMPLEX_FLOAT32 ? DataType::FLOAT32 : DataType::FLOAT64;
}

// The dtype a scalar takes part as (review 3.3 C-3: one promotion rule for matrix (+) matrix
// and matrix (+) scalar). A NumPy scalar is an operand of its own dtype (NumPy 2, NEP 50:
// "strong"). A Python scalar is weak: it takes the matrix's dtype within its kind; a Python
// float with an integer matrix is float64 and a Python complex with a real matrix is the
// complex of its precision (complex128 for integers), as in NumPy. A Python int with a bool
// matrix is int32, PyCauset's default integer (F-103).
static DataType scalar_operand_dtype(DataType a, const ScalarArg& s) {
    if (s.strong) return s.strong_dtype;
    switch (s.kind) {
        case ScalarArg::Kind::Int:
            return a == DataType::BIT ? DataType::INT32 : a;
        case ScalarArg::Kind::Float:
            return is_integer_or_bit(a) ? DataType::FLOAT64 : is_complex_dtype(a) ? real_of(a) : a;
        case ScalarArg::Kind::Complex:
            return is_complex_dtype(a) ? a : is_real_float_dtype(a) ? complex_of(a) : DataType::COMPLEX_FLOAT64;
    }
    return a;
}

DataType scalar_result_dtype(DataType a, ScalarOpKind op, const ScalarArg& s) {
    // The matrix rules (promotion::resolve) with the scalar as an operand: within-float
    // under-promotion applies to NumPy scalars too (a scalar never widens a matrix:
    // float16 M + np.float32(1) is float16), signed (+) uint64 raises (F-035b), and integer
    // division gives float32 for 8/16-bit operands (int8 M / 1 was float64).
    const DataType b = scalar_operand_dtype(a, s);
    using promotion::BinaryOp;
    BinaryOp bop = BinaryOp::Add;
    switch (op) {
        case ScalarOpKind::Add: bop = BinaryOp::Add; break;
        case ScalarOpKind::Sub: case ScalarOpKind::RSub: bop = BinaryOp::Subtract; break;
        case ScalarOpKind::Mul: bop = BinaryOp::ElementwiseMultiply; break;
        case ScalarOpKind::TrueDiv: case ScalarOpKind::RTrueDiv: bop = BinaryOp::Divide; break;
    }
    if (a == DataType::BIT && b == DataType::BIT && bop == BinaryOp::ElementwiseMultiply) return DataType::BIT;
    return promotion::resolve(bop, a, b).result_dtype;
}

// ----------------------------------------------------------------- kernels
namespace {

template <typename T>
struct dtype_of;
#define PYCAUSET_DTYPE_OF(T, D) template <> struct dtype_of<T> { static constexpr DataType value = D; };
PYCAUSET_DTYPE_OF(int8_t, DataType::INT8)
PYCAUSET_DTYPE_OF(int16_t, DataType::INT16)
PYCAUSET_DTYPE_OF(int32_t, DataType::INT32)
PYCAUSET_DTYPE_OF(int64_t, DataType::INT64)
PYCAUSET_DTYPE_OF(uint8_t, DataType::UINT8)
PYCAUSET_DTYPE_OF(uint16_t, DataType::UINT16)
PYCAUSET_DTYPE_OF(uint32_t, DataType::UINT32)
PYCAUSET_DTYPE_OF(uint64_t, DataType::UINT64)
PYCAUSET_DTYPE_OF(float, DataType::FLOAT32)
PYCAUSET_DTYPE_OF(double, DataType::FLOAT64)
PYCAUSET_DTYPE_OF(std::complex<float>, DataType::COMPLEX_FLOAT32)
PYCAUSET_DTYPE_OF(std::complex<double>, DataType::COMPLEX_FLOAT64)
#undef PYCAUSET_DTYPE_OF

// Does `x op s` map 0 to 0 (so a triangular input stays triangular)?
bool preserves_zero(ScalarOpKind op, const ScalarArg& s) {
    const std::complex<double> c = s.as_complex();
    const bool finite = std::isfinite(c.real()) && std::isfinite(c.imag());
    switch (op) {
        case ScalarOpKind::Mul: return finite;
        case ScalarOpKind::TrueDiv: return finite && c != std::complex<double>(0.0, 0.0);
        case ScalarOpKind::Add:
        case ScalarOpKind::Sub:
        case ScalarOpKind::RSub: return c == std::complex<double>(0.0, 0.0);
        case ScalarOpKind::RTrueDiv: return false;
    }
    return false;
}

// Scalar converted to the result element type R (range already checked for ints).
template <typename R>
R scalar_as(const ScalarArg& s) {
    if constexpr (std::is_integral_v<R>) {
        return s.negative ? static_cast<R>(s.i) : static_cast<R>(s.u);
    } else if constexpr (std::is_floating_point_v<R>) {
        return static_cast<R>(s.as_complex().real());
    } else {
        using V = typename R::value_type;
        const auto c = s.as_complex();
        return R(static_cast<V>(c.real()), static_cast<V>(c.imag()));
    }
}

// Forced inline: in apply_span the operator is a compile-time constant, and inlined the
// switch folds away; out of line it was a call per element (v * 2j for int16: 80% of the time).
#if defined(__GNUC__) || defined(__clang__)
#define PYCAUSET_FORCE_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define PYCAUSET_FORCE_INLINE __forceinline
#else
#define PYCAUSET_FORCE_INLINE inline
#endif

template <typename R>
PYCAUSET_FORCE_INLINE R apply(ScalarOpKind op, R x, R s) {
    if constexpr (std::is_integral_v<R>) {
        switch (op) {
            case ScalarOpKind::Add: return static_cast<R>(scalar::WrapPlus{}(x, s));
            case ScalarOpKind::Sub: return static_cast<R>(scalar::WrapMinus{}(x, s));
            case ScalarOpKind::RSub: return static_cast<R>(scalar::WrapMinus{}(s, x));
            case ScalarOpKind::Mul: return static_cast<R>(scalar::WrapMultiplies{}(x, s));
            default: throw std::logic_error("integer result for true division");
        }
    } else {
        switch (op) {
            case ScalarOpKind::Add: return x + s;
            case ScalarOpKind::Sub: return x - s;
            case ScalarOpKind::RSub: return s - x;
            // x * 1 is x. Not computed: complex (inf + 0j) * (1 + 0j) is inf + nan j, and
            // convert / dtype conversions are "multiply by 1" (F-113).
            case ScalarOpKind::Mul: return s == R(1) ? x : scalar::multiply(x, s);
            case ScalarOpKind::TrueDiv: return scalar::true_divide(x, s);
            case ScalarOpKind::RTrueDiv: return scalar::true_divide(s, x);
        }
    }
    return x;
}

// Convert an input element to the result type R.
template <typename R, typename In>
R convert_in(In v) {
    if constexpr (std::is_same_v<In, bool>) {
        return static_cast<R>(v ? 1 : 0);
    } else if constexpr (std::is_integral_v<R> || std::is_floating_point_v<R>) {
        return static_cast<R>(v);
    } else if constexpr (std::is_same_v<In, std::complex<float>> || std::is_same_v<In, std::complex<double>>) {
        return static_cast<R>(v);  // complex to complex
    } else {
        using V = typename R::value_type;
        return R(static_cast<V>(v), V(0));
    }
}

// Calls f(flat index, i, j) for every element of a rows x cols row-major result, in parallel
// over fixed blocks of the flat range (so a 1 x n vector-shaped result is parallel too).
template <typename F>
void for_each_block(uint64_t rows, uint64_t cols, F&& f) {
    const uint64_t total = rows * cols;
    if (total == 0) return;
    constexpr uint64_t kBlock = 1u << 14;
    ParallelFor(0, (total + kBlock - 1) / kBlock, [&](size_t b) {
        uint64_t idx = static_cast<uint64_t>(b) * kBlock;
        const uint64_t end = std::min(total, idx + kBlock);
        uint64_t i = idx / cols, j = idx % cols;
        for (; idx < end; ++idx) {
            f(idx, i, j);
            if (++j == cols) { j = 0; ++i; }
        }
    });
}

// The same blocks without the (i, j) bookkeeping: f(begin, end) runs a plain loop the
// compiler can vectorize (the per-element row / column branch above prevented that).
template <typename F>
void for_each_span(uint64_t total, F&& f) {
    if (total == 0) return;
    constexpr uint64_t kBlock = 1u << 14;
    ParallelFor(0, (total + kBlock - 1) / kBlock, [&](size_t b) {
        const uint64_t begin = static_cast<uint64_t>(b) * kBlock;
        f(begin, std::min(total, begin + kBlock));
    });
}

// Integer factor of an integer view (1 for plain matrices and for non-integer dtypes).
inline int64_t int_view_factor(const MatrixBase& a) {
    if (is_int_dtype(a.get_data_type()) && scalar::is_exact_int_factor(a.get_scalar())) {
        return static_cast<int64_t>(a.get_scalar().real());
    }
    return 1;
}

template <typename Src>
inline constexpr bool is_symmetric_src_v = false;
template <typename T>
inline constexpr bool is_symmetric_src_v<SymmetricMatrix<T>> = true;

// Typed source: reads element (i, j) of a concrete input class exactly.
template <typename In, typename Src>
struct TypedReader {
    const Src& m;
    int64_t k = 1;     // integer view factor (F-042), applied with NumPy's wrapping
    double f = 1.0;    // real float view factor (O(1) views of float32/float64 inputs)
    In operator()(uint64_t i, uint64_t j) const {
        if constexpr (std::is_integral_v<In> && !std::is_same_v<In, bool>) {
            return k == 1 ? m.get(i, j) : scalar::wrap_scale(m.get(i, j), k);
        } else if constexpr (std::is_floating_point_v<In>) {
            // Same rounding as the virtual accessor: stored value times the factor in double.
            return f == 1.0 ? m.get(i, j) : static_cast<In>(static_cast<double>(m.get(i, j)) * f);
        } else {
            return m.get(i, j);
        }
    }
};

// out[q] = apply(op, src[q] scaled by the integer factor k / real factor f, s) for q < len,
// with op, and whether there is a factor, fixed at compile time: a switch per element kept the
// loop scalar. Locals and restrict: int8 / uint8 are char types, which may alias anything, so
// through captures every store could change the operands (int8 + 7 was 0.1x NumPy).
template <typename R, typename In>
void apply_span(ScalarOpKind op, const In* __restrict src, R* __restrict out, uint64_t len, R sv, int64_t k,
                double f) {
    bool scaled = false;  // bit sources are only read with factor 1 (run_for_result's `plain`)
    if constexpr (std::is_integral_v<In> && !std::is_same_v<In, bool>) scaled = k != 1;
    else if constexpr (std::is_floating_point_v<In>) scaled = f != 1.0;
    auto loop = [&](auto op_c, auto scaled_c) {
        constexpr ScalarOpKind O = decltype(op_c)::value;
        const In* __restrict x = src;
        R* __restrict z = out;
        const R s_local = sv;
        const int64_t k_local = k;
        const double f_local = f;
        for (uint64_t q = 0; q < len; ++q) {
            In v = x[q];
            if constexpr (decltype(scaled_c)::value) {
                if constexpr (std::is_same_v<In, bool>) {
                } else if constexpr (std::is_integral_v<In>) v = scalar::wrap_scale(v, k_local);
                else if constexpr (std::is_floating_point_v<In>) v = static_cast<In>(static_cast<double>(v) * f_local);
            }
            z[q] = apply<R>(O, convert_in<R>(v), s_local);
        }
        (void)k_local;
        (void)f_local;
    };
    auto with_op = [&](auto scaled_c) {
        switch (op) {
            case ScalarOpKind::Add: loop(std::integral_constant<ScalarOpKind, ScalarOpKind::Add>{}, scaled_c); break;
            case ScalarOpKind::Sub: loop(std::integral_constant<ScalarOpKind, ScalarOpKind::Sub>{}, scaled_c); break;
            case ScalarOpKind::RSub: loop(std::integral_constant<ScalarOpKind, ScalarOpKind::RSub>{}, scaled_c); break;
            case ScalarOpKind::Mul: loop(std::integral_constant<ScalarOpKind, ScalarOpKind::Mul>{}, scaled_c); break;
            case ScalarOpKind::TrueDiv: loop(std::integral_constant<ScalarOpKind, ScalarOpKind::TrueDiv>{}, scaled_c); break;
            case ScalarOpKind::RTrueDiv: loop(std::integral_constant<ScalarOpKind, ScalarOpKind::RTrueDiv>{}, scaled_c); break;
        }
    };
    if (scaled) with_op(std::true_type{});
    else with_op(std::false_type{});
}

template <typename R, typename In, typename Src>
std::unique_ptr<MatrixBase> run(const Src& src, const MatrixBase& a, ScalarOpKind op, const ScalarArg& s,
                                bool triangular_in, bool has_diag, bool keep_structure) {
    const uint64_t rows = a.rows(), cols = a.cols();
    const R sv = scalar_as<R>(s);
    TypedReader<In, Src> read{src, int_view_factor(a), is_real_float_dtype(a.get_data_type()) ? a.get_scalar().real() : 1.0};

    // Triangular results for every real element type (each has a Python class since the
    // structure classes, F-032); complex results become dense.
    constexpr bool tri_capable = !std::is_same_v<R, std::complex<float>> && !std::is_same_v<R, std::complex<double>>;
    if constexpr (tri_capable) if (keep_structure && triangular_in && preserves_zero(op, s)) {
        auto out = std::make_unique<TriangularMatrix<R>>(rows, std::string(), has_diag);
        const bool lower = a.is_transposed();
        out->set_transposed(lower);
        // The result stores the same triangle as the input (a transposed input gives a
        // transposed result): stored row i in, stored row i out, through the typed span loop.
        // Element by element through get / set it was 80 ns per element (a 2000 x 2000 causal
        // matrix cast to int32 took 165 ms, 9x the product it fed; PLAN 6.5).
        if constexpr (std::is_same_v<Src, TriangularMatrix<In>> && !std::is_same_v<R, bool>) if (!a.is_subview()) {
            const uint64_t n = src.base_rows();
            const int64_t k = int_view_factor(a);
            const double f = is_real_float_dtype(a.get_data_type()) ? a.get_scalar().real() : 1.0;
            const char* in_base = static_cast<const char*>(src.require_mapper()->get_data());
            char* out_base = static_cast<char*>(out->require_mapper()->get_data());
            ParallelFor(0, n, [&](size_t i) {
                const uint64_t c0 = has_diag ? i : i + 1;
                if (c0 >= n) return;
                const uint64_t len = n - c0;
                R* orow = reinterpret_cast<R*>(out_base + out->get_row_offset(i));
                if constexpr (std::is_same_v<In, bool>) {
                    const uint64_t* w = reinterpret_cast<const uint64_t*>(in_base + src.get_row_offset(i));
                    std::vector<uint8_t> bits(len);
                    for (uint64_t t = 0; t < len; ++t) bits[t] = static_cast<uint8_t>((w[t / 64] >> (t % 64)) & 1u);
                    apply_span<R, bool>(op, reinterpret_cast<const bool*>(bits.data()), orow, len, sv, k, f);
                } else {
                    apply_span<R, In>(op, reinterpret_cast<const In*>(in_base + src.get_row_offset(i)), orow, len, sv, k, f);
                }
            });
            return out;
        }
        ParallelFor(0, rows, [&](size_t i) {
            const uint64_t j0 = lower ? 0 : (has_diag ? i : i + 1);
            const uint64_t j1 = lower ? (has_diag ? i + 1 : i) : cols;
            for (uint64_t j = j0; j < j1; ++j) {
                out->set(i, j, apply<R>(op, convert_in<R>(read(i, j)), sv));
            }
        });
        return out;
    }

    // Symmetric input: every scalar op keeps A == A^T; only zero-preserving ones keep
    // A == -A^T. Keeping the structure here makes the eager result the same class as the
    // lazy view (LAZY.md section 2; found by the PYCAUSET_LAZY=0 oracle run).
    if constexpr (is_symmetric_src_v<Src> && !std::is_same_v<R, std::complex<float>> &&
                  !std::is_same_v<R, std::complex<double>>) if (keep_structure) {
        const bool anti = a.get_structure() == Structure::ANTISYMMETRIC;
        if (!anti || preserves_zero(op, s)) {
            auto out = std::make_unique<SymmetricMatrix<R>>(rows, std::string(), anti ? Structure::ANTISYMMETRIC : Structure::SYMMETRIC);
            // Same element type and no view (a transposed antisymmetric matrix is negated):
            // the op over the packed storage, one element each (it went through get / set).
            if constexpr (std::is_same_v<R, In> && !std::is_same_v<In, bool>) {
                const size_t bytes = out->require_mapper()->get_data_size();
                if (!a.is_subview() && a.get_scalar() == std::complex<double>(1.0, 0.0) && !a.is_conjugated() &&
                    !(anti && a.is_transposed()) && src.require_mapper()->get_data_size() == bytes) {
                    const In* x = src.data();
                    R* z = out->data();
                    ParallelForRange(0, bytes / sizeof(R), [&](size_t k0, size_t k1) {
                        const In* __restrict xr = x;
                        R* __restrict zr = z;
                        const R s_local = sv;
                        for (size_t k = k0; k < k1; ++k) zr[k] = apply<R>(op, convert_in<R>(xr[k]), s_local);
                    });
                    return out;
                }
            }
            for (uint64_t i = 0; i < rows; ++i) {
                for (uint64_t j = anti ? i + 1 : i; j < cols; ++j) {
                    out->set(i, j, apply<R>(op, convert_in<R>(read(i, j)), sv));
                }
            }
            return out;
        }
    }

    std::unique_ptr<DenseMatrix<R>> out;
    {
        // Every element is written below; skip zeroing the storage (not for bit rows: padding).
        std::optional<UninitializedStorage> u;
        if constexpr (!std::is_same_v<R, bool>) u.emplace();
        out = std::make_unique<DenseMatrix<R>>(rows, cols, std::string());
    }
    R* dst = out->data();

    // Contiguous source, same result dtype (the common case: v + 1, 3 * v, -v): raw pointer,
    // op fixed at compile time. Same arithmetic as the general loop below. Limited to R == In
    // (plus the two common promotions) to keep the instantiation count (and compile memory) small.
    // Also for the two common promotions (to float64 and to complex128: int -> float, real ->
    // complex), which used the per-element general loop (3x NumPy's time for v * 2j).
    constexpr bool fast_pair = std::is_same_v<R, In> || std::is_same_v<R, double> || std::is_same_v<R, std::complex<double>>;
    if constexpr (std::is_same_v<Src, DenseMatrix<In>> && fast_pair && !std::is_same_v<In, bool>) {
        if (!a.is_transposed() && !a.is_subview() && cols == a.base_cols()) {
            const In* p = src.data();
            for_each_span(rows * cols, [&](uint64_t begin, uint64_t end) {
                apply_span<R, In>(op, p + begin, dst + begin, end - begin, sv, read.k, read.f);
            });
            return out;
        }
    }

    // Structured or transposed source: stored values in bulk, a tile of rows at a time
    // (math/RawRows.hpp), then the same per-element arithmetic as TypedReader. Writing a
    // structured operand out through get() per element made structured & | ^ ~ and / 5-10x
    // slower than NumPy.
    // (Bit sources for every result type: unpacking is the whole cost, and bit arithmetic
    // counts in int32, so R is never bool's own type.)
    if constexpr (has_raw_rows_v<In, Src> &&
                  (fast_pair || std::is_same_v<In, bool> || std::is_same_v<Src, ComplexFloat16Matrix>)) {
        // At least 32 tiles when there are rows for it: ParallelFor runs fewer items inline.
        const uint64_t TILE = std::clamp<uint64_t>(rows / 32, 1, 32);
        const uint64_t tiles = (rows + TILE - 1) / TILE;
        std::atomic<bool> ok{true};  // false for the same reason in every tile (a window, ...)
        ParallelFor(0, tiles, [&](size_t t) {
            const uint64_t r0 = t * TILE, r1 = std::min(rows, r0 + TILE);
            std::unique_ptr<In[]> raw(new In[(r1 - r0) * cols]);  // filled by raw_rows: no zeroing
            if (!raw_rows<In>(src, r0, r1, raw.get())) { ok = false; return; }
            apply_span<R, In>(op, raw.get(), dst + r0 * cols, (r1 - r0) * cols, sv, read.k, read.f);
        });
        if (ok) return out;
    }

    for_each_block(rows, cols, [&](uint64_t idx, uint64_t i, uint64_t j) {
        dst[idx] = apply<R>(op, convert_in<R>(read(i, j)), sv);
    });
    return out;
}

// Generic (exact for |x| < 2^53): read through the virtual complex accessor, which applies
// views. Only for what has no typed reader below (windows of structured classes, lazy and
// closure classes); the atlas counts these calls as the `generic` path.
template <typename R>
inline R from_logical(std::complex<double> v) {
    if constexpr (std::is_integral_v<R>) {
        return static_cast<R>(static_cast<int64_t>(v.real()));
    } else if constexpr (std::is_floating_point_v<R>) {
        return static_cast<R>(v.real());
    } else {
        using V = typename R::value_type;
        return R(static_cast<V>(v.real()), static_cast<V>(v.imag()));
    }
}

template <typename R>
std::unique_ptr<MatrixBase> run_virtual(const MatrixBase& a, ScalarOpKind op, const ScalarArg& s) {
    const uint64_t rows = a.rows(), cols = a.cols();
    const R sv = scalar_as<R>(s);
    std::unique_ptr<DenseMatrix<R>> out;
    {
        // Every element is written below; skip zeroing the storage (not for bit rows: padding).
        std::optional<UninitializedStorage> u;
        if constexpr (!std::is_same_v<R, bool>) u.emplace();
        out = std::make_unique<DenseMatrix<R>>(rows, cols, std::string());
    }
    R* dst = out->data();
    for_each_block(rows, cols, [&](uint64_t idx, uint64_t i, uint64_t j) {
        dst[idx] = apply<R>(op, from_logical<R>(a.get_element_as_complex(i, j)), sv);
    });
    return out;
}

// Every view of every class as a dense R matrix (no structure kept). The typed reader of
// ViewRead.cpp brings the operand to a plain wide matrix (int64 / float64 / complex128) with
// the view applied; the typed dense path narrows it to R. Only what has no typed reader
// (windows of structured classes, lazy and closure classes) reads through the virtual accessor.
template <typename R, typename W>
std::unique_ptr<MatrixBase> narrow_wide(const MatrixBase& wide, ScalarOpKind op, const ScalarArg& s) {
    const W* src = static_cast<const DenseMatrix<W>&>(wide).data();
    const uint64_t rows = wide.rows(), cols = wide.cols();
    const R sv = scalar_as<R>(s);
    std::unique_ptr<DenseMatrix<R>> out;
    {
        // Every element is written below; skip zeroing the storage (not for bit rows: padding).
        std::optional<UninitializedStorage> u;
        if constexpr (!std::is_same_v<R, bool>) u.emplace();
        out = std::make_unique<DenseMatrix<R>>(rows, cols, std::string());
    }
    R* dst = out->data();
    // complex128 wide values go to complex results only (run_generic picks W by R's kind), so
    // convert_in is the plain conversion here as well.
    for_each_span(rows * cols, [&](uint64_t begin, uint64_t end) {
        apply_span<R, W>(op, src + begin, dst + begin, end - begin, sv, 1, 1.0);
    });
    return out;
}

template <typename R>
std::unique_ptr<MatrixBase> run_generic(const MatrixBase& a, ScalarOpKind op, const ScalarArg& s) {
    if constexpr (std::is_integral_v<R>) {
        if (auto w = read_wide(a, WideKind::Int64)) return narrow_wide<R, int64_t>(*w, op, s);
    } else if constexpr (std::is_floating_point_v<R>) {
        if (auto w = read_wide(a, WideKind::Float64)) return narrow_wide<R, double>(*w, op, s);
    } else {
        if (auto w = read_wide(a, WideKind::Complex128)) return narrow_wide<R, std::complex<double>>(*w, op, s);
    }
    return run_virtual<R>(a, op, s);
}

// Dispatch on the concrete input class for a fixed result type R.
template <typename R>
std::unique_ptr<MatrixBase> run_for_result(const MatrixBase& a, ScalarOpKind op, const ScalarArg& s,
                                           bool keep_structure) {
    // Typed paths read the factor themselves: an exact integer factor for integer inputs, a
    // real factor for float32/float64 inputs (conjugation is a no-op on real values). Other
    // views (complex factors, bit with a factor) use the generic accessor.
    const std::complex<double> sc = a.get_scalar();
    const DataType dt = a.get_data_type();
    const bool plain = sc == std::complex<double>(1.0, 0.0) ? (!a.is_conjugated() || !is_complex_dtype(dt))
                     : is_int_dtype(dt) ? (!a.is_conjugated() && scalar::is_exact_int_factor(sc))
                     : (is_real_float_dtype(dt) && sc.imag() == 0.0);
#define PYCAUSET_TRY_DENSE(T)                                                              \
    if (auto* m = dynamic_cast<const DenseMatrix<T>*>(&a); m && plain)                    \
        return run<R, T>(*m, a, op, s, false, false, keep_structure);
#define PYCAUSET_TRY_TRI(T)                                                                \
    if (auto* m = dynamic_cast<const TriangularMatrix<T>*>(&a); m && plain)               \
        return run<R, T>(*m, a, op, s, true, m->has_diagonal(), keep_structure);
    PYCAUSET_TRY_DENSE(bool)
    PYCAUSET_TRY_DENSE(int8_t) PYCAUSET_TRY_DENSE(int16_t) PYCAUSET_TRY_DENSE(int32_t) PYCAUSET_TRY_DENSE(int64_t)
    PYCAUSET_TRY_DENSE(uint8_t) PYCAUSET_TRY_DENSE(uint16_t) PYCAUSET_TRY_DENSE(uint32_t) PYCAUSET_TRY_DENSE(uint64_t)
    PYCAUSET_TRY_DENSE(float) PYCAUSET_TRY_DENSE(double)
    if constexpr (std::is_same_v<R, std::complex<float>> || std::is_same_v<R, std::complex<double>>) {
        // Complex inputs (complex results only): typed and contiguous instead of a full copy
        // through the wide reader (c + 1j was 5 ms for 1024 x 1024).
        PYCAUSET_TRY_DENSE(std::complex<float>) PYCAUSET_TRY_DENSE(std::complex<double>)
        // complex_float16 storage (two float16 planes), read as complex128 in bulk.
        if (auto* m = dynamic_cast<const ComplexFloat16Matrix*>(&a); m && sc == std::complex<double>(1.0, 0.0) &&
                                                                      !a.is_conjugated())
            return run<R, std::complex<double>>(*m, a, op, s, false, false, keep_structure);
        // Structured complex inputs: the typed bulk-tile path (they went through a full
        // complex128 copy and a per-element operator switch: 0.25x NumPy).
        PYCAUSET_TRY_TRI(std::complex<float>) PYCAUSET_TRY_TRI(std::complex<double>)
        if (auto* m = dynamic_cast<const SymmetricMatrix<std::complex<float>>*>(&a); m && plain)
            return run<R, std::complex<float>>(*m, a, op, s, false, false, keep_structure);
        if (auto* m = dynamic_cast<const SymmetricMatrix<std::complex<double>>*>(&a); m && plain)
            return run<R, std::complex<double>>(*m, a, op, s, false, false, keep_structure);
    }
    PYCAUSET_TRY_TRI(bool)
    PYCAUSET_TRY_TRI(int8_t) PYCAUSET_TRY_TRI(int16_t) PYCAUSET_TRY_TRI(int32_t) PYCAUSET_TRY_TRI(int64_t)
    PYCAUSET_TRY_TRI(uint8_t) PYCAUSET_TRY_TRI(uint16_t) PYCAUSET_TRY_TRI(uint32_t) PYCAUSET_TRY_TRI(uint64_t)
    PYCAUSET_TRY_TRI(float) PYCAUSET_TRY_TRI(double)
#define PYCAUSET_TRY_SYM(T)                                                                \
    if (auto* m = dynamic_cast<const SymmetricMatrix<T>*>(&a); m && plain && is_int_dtype(a.get_data_type())) \
        return run<R, T>(*m, a, op, s, false, false, keep_structure);
    PYCAUSET_TRY_SYM(int8_t) PYCAUSET_TRY_SYM(int16_t) PYCAUSET_TRY_SYM(int32_t) PYCAUSET_TRY_SYM(int64_t)
    PYCAUSET_TRY_SYM(uint8_t) PYCAUSET_TRY_SYM(uint16_t) PYCAUSET_TRY_SYM(uint32_t) PYCAUSET_TRY_SYM(uint64_t)
#undef PYCAUSET_TRY_SYM
#undef PYCAUSET_TRY_DENSE
#undef PYCAUSET_TRY_TRI
    return run_generic<R>(a, op, s);
}

std::unique_ptr<MatrixBase> run_exact(const MatrixBase& a, DataType out, ScalarOpKind op, const ScalarArg& s,
                                      bool keep_structure = true) {
    switch (out) {
        case DataType::INT8: return run_for_result<int8_t>(a, op, s, keep_structure);
        case DataType::INT16: return run_for_result<int16_t>(a, op, s, keep_structure);
        case DataType::INT32: return run_for_result<int32_t>(a, op, s, keep_structure);
        case DataType::INT64: return run_for_result<int64_t>(a, op, s, keep_structure);
        case DataType::UINT8: return run_for_result<uint8_t>(a, op, s, keep_structure);
        case DataType::UINT16: return run_for_result<uint16_t>(a, op, s, keep_structure);
        case DataType::UINT32: return run_for_result<uint32_t>(a, op, s, keep_structure);
        case DataType::UINT64: return run_for_result<uint64_t>(a, op, s, keep_structure);
        case DataType::FLOAT32: return run_for_result<float>(a, op, s, keep_structure);
        case DataType::FLOAT64: return run_for_result<double>(a, op, s, keep_structure);
        case DataType::COMPLEX_FLOAT32: return run_for_result<std::complex<float>>(a, op, s, keep_structure);
        case DataType::COMPLEX_FLOAT64: return run_for_result<std::complex<double>>(a, op, s, keep_structure);
        default:
            throw std::invalid_argument("scalar operation: unsupported result dtype " + std::string(dtype_name(out)));
    }
}

}  // namespace

// Shared-storage view of an integer matrix scaled by k (nullptr when not representable).
std::unique_ptr<MatrixBase> int_scale_view(const MatrixBase& a, int64_t k) {
    if (!lazy_enabled().load()) return nullptr;
    const Structure st = a.get_structure();
    if ((st != Structure::DENSE && st != Structure::TRIANGULAR && st != Structure::SYMMETRIC &&
         st != Structure::ANTISYMMETRIC) || a.is_conjugated()) return nullptr;
    int64_t f = 0;
    if (!scalar::fold_int_factors(a.get_scalar(), k, f)) return nullptr;
    auto c = a.clone();
    std::unique_ptr<MatrixBase> view(static_cast<MatrixBase*>(c.release()));
    view->set_scalar(static_cast<double>(f));
    return view;
}

// ------------------------------------------------------------- entry points
std::unique_ptr<MatrixBase> scalar_op(const MatrixBase& a, ScalarOpKind op, const ScalarArg& s) {
    const DataType in = a.get_data_type();
    if (s.beyond_64_bits && is_integer_or_bit(in)) {
        throw std::overflow_error("Python integer out of bounds for " + std::string(dtype_name(in == DataType::BIT ? DataType::INT64 : in)));
    }
    const DataType out = scalar_result_dtype(in, op, s);

    if (is_integer_or_bit(in) && s.kind == ScalarArg::Kind::Int && is_int_dtype(out) && !int_fits(s, out)) {
        throw std::overflow_error("Python integer out of bounds for " + std::string(dtype_name(out)) +
                                  " (scalar does not fit the matrix dtype)");
    }

    // bool * bool scalar (the only bool result): a copy of the bits or all False.
    if (out == DataType::BIT) {
        if (s.as_complex() != std::complex<double>(0.0, 0.0)) return bitwise(a, a, BitwiseOp::And);
        return ObjectFactory::create_matrix(a.rows(), a.cols(), DataType::BIT, Structure::DENSE, "");
    }

    // Integer input, integer factor, dtype kept: an O(1) view sharing storage (LAZY.md L1,
    // F-042). Dense and triangular only (their readers apply integer factors); the folded
    // factor must stay exact, otherwise compute eagerly.
    if (op == ScalarOpKind::Mul && is_int_dtype(in) && out == in && s.kind == ScalarArg::Kind::Int) {
        const int64_t k = s.negative ? s.i : static_cast<int64_t>(s.u);  // mod 2^64 is exact here
        if (auto view = int_scale_view(a, k)) return view;
    }

    // Float/complex input whose dtype is kept: the classes' O(1) scalar-view paths.
    // Structure rule: a (skew-)Hermitian matrix times a non-real factor is neither, so its
    // class cannot be kept (H * 1j is skew-Hermitian, H * (1+1j) has no structure); those
    // compute eagerly into a dense result (review 3.2 C-4). The class's view factor stays real.
    const Structure st = a.get_structure();
    const bool hermitian_kind = st == Structure::HERMITIAN || st == Structure::SKEW_HERMITIAN;
    if (!is_integer_or_bit(in) && out == in && op != ScalarOpKind::RTrueDiv &&
        !(hermitian_kind && s.as_complex().imag() != 0.0)) {
        const std::complex<double> c = s.as_complex();
        switch (op) {
            case ScalarOpKind::Mul:
                return s.kind == ScalarArg::Kind::Complex ? a.multiply_scalar(c) : a.multiply_scalar(c.real());
            case ScalarOpKind::TrueDiv: {
                // A / s is the view A * (1/s) only when 1/s is finite: a subnormal s overflows
                // the reciprocal ([1e-10, 0] / 1e-310 gave [inf, nan], NumPy [1e300, 0]; review
                // 3.6 C-4). Otherwise divide element by element below.
                // And only when 1/s is exact, i.e. s a (real) power of two: x * (1/s) is otherwise
                // not the rounded quotient (49 / 49 gave 0.9999999999999999; a third of the
                // elements of A / 3 were an ulp off NumPy; review 3.2 C-14).
                if (c.imag() != 0.0) break;
                int e = 0;
                if (std::frexp(c.real(), &e) != 0.5 && std::frexp(c.real(), &e) != -0.5) break;
                const double r = 1.0 / c.real();
                if (!std::isfinite(r)) break;
                return a.multiply_scalar(r);
            }
            default:
                break;  // + and - compute below, through the one typed path (F-086)
        }
    }
    // Half-precision results: compute in float64 / complex128 (exact for these operands and
    // scalars' sums and products) and round once.
    if (out == DataType::FLOAT16 || out == DataType::COMPLEX_FLOAT16) {
        const DataType wide = out == DataType::FLOAT16 ? DataType::FLOAT64 : DataType::COMPLEX_FLOAT64;
        return convert(*run_exact(a, wide, op, s), out, Keep::Structure);
    }
    return run_exact(a, out, op, s);
}

std::unique_ptr<MatrixBase> negate(const MatrixBase& a) {
    if (a.get_data_type() == DataType::BIT) {
        throw std::invalid_argument("negation of a bit/bool matrix is not supported (as in NumPy); use bitwise_not");
    }
    if (is_int_dtype(a.get_data_type())) {
        // -A is the view factor -1 (unsigned negation wraps like NumPy: same arithmetic).
        if (auto view = int_scale_view(a, -1)) return view;
        // Unsigned negation wraps (NumPy): multiply by -1 in the unsigned type.
        ScalarArg minus_one = ScalarArg::from_int64(-1);
        if (is_unsigned_int_dtype(a.get_data_type())) {
            return run_exact(a, a.get_data_type(), ScalarOpKind::RSub, ScalarArg::from_int64(0));
        }
        return run_exact(a, a.get_data_type(), ScalarOpKind::Mul, minus_one);
    }
    return a.multiply_scalar(-1.0);
}

bool is_plain(const MatrixBase& a) {
    return !a.is_transposed() && !a.is_subview() && a.get_scalar() == std::complex<double>(1.0, 0.0) &&
           !(a.is_conjugated() && is_complex_dtype(a.get_data_type()));
}

// The one conversion (review 3.6 C-16: materialize, write_out, cast_to and to_dense were four
// bodies of this). Multiplying by one in the target type is exact; half-precision targets go
// through float64 / complex128 and round once (always dense); a bool target only from bool.
std::unique_ptr<MatrixBase> convert(const MatrixBase& a, DataType dt, Keep keep) {
    const uint64_t rows = a.rows(), cols = a.cols();
    if (dt == DataType::BIT) {
        if (a.get_data_type() != DataType::BIT)
            throw std::invalid_argument("convert: no conversion to bool (compare with != 0 instead)");
        if (keep == Keep::Structure) return run_exact(a, dt, ScalarOpKind::Mul, ScalarArg::from_int64(1));
        // Packed bits from the typed row reader (a block of rows at a time); the virtual
        // accessor only where the reader cannot read (structured windows, closures).
        auto out = std::make_unique<DenseMatrix<bool>>(rows, cols, std::string());
        uint64_t* words = out->data();
        const uint64_t stride = out->stride_bytes() / 8;
        const uint64_t block = std::max<uint64_t>(1, (uint64_t{1} << 18) / std::max<uint64_t>(cols, 1));
        std::vector<int64_t> buf(std::min(block, rows) * cols);
        bool typed = true;
        for (uint64_t i0 = 0; i0 < rows && typed; i0 += block) {
            const uint64_t i1 = std::min(rows, i0 + block);
            if (!(typed = read_rows(a, i0, i1, buf.data()))) break;
            ParallelFor(i0, i1, [&](size_t i) {
                const int64_t* v = buf.data() + (i - i0) * cols;
                uint64_t* w = words + i * stride;
                for (uint64_t j = 0; j < cols; ++j)
                    if (v[j] != 0) w[j / 64] |= uint64_t{1} << (j % 64);
            });
        }
        if (typed) return out;
        for (uint64_t i = 0; i < rows; ++i) {
            for (uint64_t j = 0; j < cols; ++j) out->set(i, j, a.get_element_as_double(i, j) != 0.0);
        }
        return out;
    }
    // Half-precision targets: the typed float64 / complex128 conversion (exact for every
    // source the old per-element path read exactly), then one rounding to half precision.
    if (dt == DataType::FLOAT16) {
        auto wide = run_exact(a, DataType::FLOAT64, ScalarOpKind::Mul, ScalarArg::from_int64(1), false);
        const double* src = static_cast<const DenseMatrix<double>&>(*wide).data();
        auto out = std::make_unique<DenseMatrix<pycauset::float16_t>>(rows, cols, std::string());
        pycauset::float16_t* dst = out->data();
        ParallelFor(0, rows, [&](size_t i) {
            for (uint64_t j = 0; j < cols; ++j) dst[i * cols + j] = pycauset::float16_t(src[i * cols + j]);
        });
        return out;
    }
    if (dt == DataType::COMPLEX_FLOAT16) {
        // A plain dense complex64 / complex128 source (the usual one: a complex32 operation
        // computes in complex64) converts straight into the two planes, without the
        // complex128 copy.
        auto direct = [&](auto tag) -> std::unique_ptr<MatrixBase> {
            using C = typename decltype(tag)::type;
            const auto* d = dynamic_cast<const DenseMatrix<C>*>(&a);
            if (!d || !is_plain(a) || a.is_subview() || a.cols() != a.base_cols()) return nullptr;
            const C* src = d->data();
            std::unique_ptr<ComplexFloat16Matrix> out;
            {
                UninitializedStorage u;  // both planes are written below
                out = std::make_unique<ComplexFloat16Matrix>(rows, cols, std::string());
            }
            pycauset::float16_t* re = out->real_data();
            pycauset::float16_t* im = out->imag_data();
            ParallelForRange(0, rows * cols, [&](size_t k0, size_t k1) {
                for (size_t k = k0; k < k1; ++k) {
                    re[k] = pycauset::float16_t(static_cast<double>(src[k].real()));
                    im[k] = pycauset::float16_t(static_cast<double>(src[k].imag()));
                }
            });
            return out;
        };
        if (auto r = direct(type_tag<std::complex<float>>{})) return r;
        if (auto r = direct(type_tag<std::complex<double>>{})) return r;
        auto wide = run_exact(a, DataType::COMPLEX_FLOAT64, ScalarOpKind::Mul, ScalarArg::from_int64(1), false);
        const std::complex<double>* src = static_cast<const DenseMatrix<std::complex<double>>&>(*wide).data();
        auto out = std::make_unique<ComplexFloat16Matrix>(rows, cols, std::string());
        pycauset::float16_t* re = out->real_data();
        pycauset::float16_t* im = out->imag_data();
        ParallelFor(0, rows, [&](size_t i) {
            for (uint64_t j = 0; j < cols; ++j) {
                re[i * cols + j] = pycauset::float16_t(src[i * cols + j].real());
                im[i * cols + j] = pycauset::float16_t(src[i * cols + j].imag());
            }
        });
        return out;
    }
    // Multiplying by one in the target type is an exact conversion (ints, floats, complex).
    return run_exact(a, dt, ScalarOpKind::Mul, ScalarArg::from_int64(1), keep == Keep::Structure);
}

std::unique_ptr<MatrixBase> write_out(const MatrixBase& a) {
    debug_trace::set_last("write_out");
    return convert(a, a.get_data_type(), Keep::Dense);
}


namespace {

// Logical rows of a, a block at a time: f(i0, i1, const W* rows) with rows [i0, i1) row-major
// (a.cols() wide), typed and view-applied (read_rows). Where the typed reader cannot read
// (windows of structured classes, closures) the values are written out once into a plain
// dense copy, which it can.
template <typename W, typename F>
bool for_row_blocks(const MatrixBase& a, F&& f) {
    const uint64_t rows = a.rows(), cols = std::max<uint64_t>(a.cols(), 1);
    const uint64_t block = std::max<uint64_t>(1, (uint64_t{1} << 17) / cols);
    std::vector<W> buf(std::min(block, std::max<uint64_t>(rows, 1)) * cols);
    if (rows > 0 && !read_rows(a, 0, std::min(block, rows), buf.data())) {
        auto d = write_out(a);
        return for_row_blocks<W>(*d, std::forward<F>(f));
    }
    for (uint64_t i0 = 0; i0 < rows; i0 += block) {
        const uint64_t i1 = std::min(rows, i0 + block);
        if (i0 > 0 && !read_rows(a, i0, i1, buf.data())) throw std::runtime_error("row reader failed mid-matrix");
        if (!f(i0, i1, static_cast<const W*>(buf.data()))) return false;
    }
    return true;
}

// The same blocks of a and of a's transpose side by side: f(i0, i1, rows, cols_of_a).
template <typename W, typename F>
bool for_row_and_column_blocks(const MatrixBase& a, F&& f) {
    const uint64_t n = a.rows();
    const uint64_t block = std::max<uint64_t>(1, (uint64_t{1} << 17) / std::max<uint64_t>(n, 1));
    std::vector<W> r(std::min(block, std::max<uint64_t>(n, 1)) * n), c(r.size());
    auto t = a.transposed_clone();
    if (n > 0 && !(read_rows(a, 0, 1, r.data()) && read_rows(*t, 0, 1, c.data()))) {
        auto d = write_out(a);
        return for_row_and_column_blocks<W>(*d, std::forward<F>(f));
    }
    for (uint64_t i0 = 0; i0 < n; i0 += block) {
        const uint64_t i1 = std::min(n, i0 + block);
        if (!read_rows(a, i0, i1, r.data()) || !read_rows(*t, i0, i1, c.data()))
            throw std::runtime_error("row reader failed mid-matrix");
        if (!f(i0, i1, static_cast<const W*>(r.data()), static_cast<const W*>(c.data()))) return false;
    }
    return true;
}

// NumPy's isclose(x, y, rtol, atol) (equal_nan=False): |x - y| <= atol + rtol * |y| for
// finite operands, x == y otherwise (inf matches the same inf; NaN matches nothing).
template <typename W>
bool numpy_isclose(const W& x, const W& y, double rtol, double atol) {
    auto finite = [](const W& v) {
        if constexpr (std::is_same_v<W, std::complex<double>>) return std::isfinite(v.real()) && std::isfinite(v.imag());
        else return std::isfinite(v);
    };
    if (!finite(x) || !finite(y)) return x == y;
    return std::abs(x - y) <= atol + rtol * std::abs(y);
}

int int_width(DataType dt) {
    switch (dt) {
        case DataType::INT8: case DataType::UINT8: return 8;
        case DataType::INT16: case DataType::UINT16: return 16;
        case DataType::INT32: case DataType::UINT32: return 32;
        default: return 64;
    }
}

}  // namespace

bool has_structure(const MatrixBase& a, Structure s, double rtol, double atol) {
    if (s != Structure::SYMMETRIC && s != Structure::ANTISYMMETRIC && s != Structure::HERMITIAN)
        throw std::invalid_argument("has_structure: symmetric, antisymmetric or hermitian");
    const DataType dt = a.get_data_type();
    if (s == Structure::ANTISYMMETRIC && (dt == DataType::BIT || is_unsigned_int_dtype(dt)))
        throw std::invalid_argument(std::string("antisymmetric matrices need a signed dtype (got ") + dtype_name(dt) + ")");
    detail::require_square(s, a.rows(), a.cols());
    debug_trace::set_last("has_structure");
    const uint64_t n = a.rows();
    std::atomic<bool> ok{true};
    if (is_integer_or_bit(dt)) {
        // Exact, in the dtype's own wrapping arithmetic (NumPy's -A of int8 -128 is -128).
        const uint64_t mask = int_width(dt) == 64 ? ~uint64_t{0} : (uint64_t{1} << int_width(dt)) - 1;
        const bool anti = s == Structure::ANTISYMMETRIC;
        return for_row_and_column_blocks<int64_t>(a, [&](uint64_t i0, uint64_t i1, const int64_t* r, const int64_t* c) {
            ParallelFor(i0, i1, [&](size_t i) {
                const int64_t* x = r + (i - i0) * n;
                const int64_t* y = c + (i - i0) * n;  // y[j] = a(j, i)
                for (uint64_t j = 0; j < n && ok.load(std::memory_order_relaxed); ++j) {
                    const bool good = anti ? ((static_cast<uint64_t>(x[j]) + static_cast<uint64_t>(y[j])) & mask) == 0 &&
                                                 (j != i || x[j] == 0)
                                           : x[j] == y[j];
                    if (!good) ok.store(false, std::memory_order_relaxed);
                }
            });
            return ok.load();
        });
    }
    auto run = [&](auto tag) {
        using W = typename decltype(tag)::type;
        return for_row_and_column_blocks<W>(a, [&](uint64_t i0, uint64_t i1, const W* r, const W* c) {
            ParallelFor(i0, i1, [&](size_t i) {
                const W* x = r + (i - i0) * n;
                const W* y = c + (i - i0) * n;
                for (uint64_t j = 0; j < n && ok.load(std::memory_order_relaxed); ++j) {
                    bool good;
                    if (s == Structure::SYMMETRIC) {
                        good = numpy_isclose(x[j], y[j], rtol, atol);
                    } else if (s == Structure::ANTISYMMETRIC) {
                        good = numpy_isclose(x[j], W(-y[j]), rtol, atol) &&
                               (j != i || numpy_isclose(x[j], W(0), rtol, atol));
                    } else {
                        if constexpr (std::is_same_v<W, std::complex<double>>) good = numpy_isclose(x[j], std::conj(y[j]), rtol, atol);
                        else good = numpy_isclose(x[j], y[j], rtol, atol);
                    }
                    if (!good) ok.store(false, std::memory_order_relaxed);
                }
            });
            return ok.load();
        });
    };
    return is_complex_dtype(dt) ? run(type_tag<std::complex<double>>{}) : run(type_tag<double>{});
}

std::pair<bool, bool> causal_input_check(const MatrixBase& a) {
    const DataType dt = a.get_data_type();
    const uint64_t cols = a.cols();
    std::atomic<bool> nonbinary{false}, lower{false};
    debug_trace::set_last("causal_input_check");
    auto scan = [&](auto tag) {
        using W = typename decltype(tag)::type;
        for_row_blocks<W>(a, [&](uint64_t i0, uint64_t i1, const W* rows) {
            ParallelFor(i0, i1, [&](size_t i) {
                const W* v = rows + (i - i0) * cols;
                bool nb = false, lo = false;
                for (uint64_t j = 0; j < cols; ++j) {
                    const bool zero = v[j] == W(0);
                    if (!zero && !(v[j] == W(1))) nb = true;
                    if (!zero && j <= i) lo = true;
                }
                if (nb) nonbinary.store(true, std::memory_order_relaxed);
                if (lo) lower.store(true, std::memory_order_relaxed);
            });
            return true;
        });
    };
    if (is_integer_or_bit(dt)) scan(type_tag<int64_t>{});
    else if (is_complex_dtype(dt)) scan(type_tag<std::complex<double>>{});
    else scan(type_tag<double>{});
    return {nonbinary.load(), lower.load()};
}

// The stored region of a (view applied) copied into a new matrix of structure s in a's dtype.
// Exact for every dtype: one typed read into a plain dense copy (none when a is plain dense
// already), then whole stored rows copied (memcpy; bit rows shifted into place). Entries outside
// the stored region are not read (the caller vouches for them, e.g. pc.symmetric validates its
// input first). The symmetric family's diagonal goes through `set` for its checks.
std::unique_ptr<MatrixBase> pack_as(const MatrixBase& a, Structure s, bool has_diagonal) {
    MatrixOptions opts;
    opts.has_diagonal = has_diagonal;
    const DataType dt = a.get_data_type();
    const uint64_t n = a.rows(), cols = a.cols();
    if (s != Structure::DENSE) detail::require_square(s, a.rows(), a.cols());
    auto out = make_matrix(s, dt, a.rows(), a.cols(), "", opts);
    if (s == Structure::IDENTITY) return out;
    std::unique_ptr<MatrixBase> copy;
    const bool plain_dense = is_plain(a) && a.get_structure() == Structure::DENSE;
    if (!plain_dense) copy = write_out(a);
    const MatrixBase& d = plain_dense ? a : *copy;
    const bool diag = s == Structure::TRIANGULAR && opts.has_diagonal;
    auto stored = [&](uint64_t i, uint64_t j) {
        switch (s) {
            case Structure::TRIANGULAR: return diag ? j >= i : j > i;
            case Structure::DIAGONAL: return i == j;
            case Structure::ANTISYMMETRIC: return j > i;
            case Structure::SYMMETRIC: case Structure::HERMITIAN: case Structure::SKEW_HERMITIAN: return j >= i;
            default: return true;
        }
    };
    debug_trace::set_last("pack_as");
    visit_dtype(dt, [&](auto tag) {
        using T = typename decltype(tag)::type;
        const auto* srcp = dynamic_cast<const typename DenseMatrixOf<T>::type*>(&d);
        std::unique_ptr<MatrixBase> fallback;
        if (!srcp) {  // a dense class outside the catalogue's (should not happen): write it out
            fallback = write_out(d);
            srcp = &static_cast<const typename DenseMatrixOf<T>::type&>(*fallback);
        }
        const auto& src = *srcp;
        visit_structure(s, [&](auto st) {
            constexpr Structure S = decltype(st)::value;
            if constexpr (offered<S, T>()) {
                auto& dst = static_cast<typename ClassOf<S, T>::type&>(*out);
                using E = structured_element_t<T>;
                constexpr bool packed_rows = S == Structure::TRIANGULAR || S == Structure::SYMMETRIC ||
                                             S == Structure::ANTISYMMETRIC || S == Structure::HERMITIAN ||
                                             S == Structure::SKEW_HERMITIAN;
                if constexpr (S == Structure::DIAGONAL) {
                    for (uint64_t i = 0; i < n; ++i) dst.set_diagonal(i, E(src.get(i, i)));
                } else if constexpr (S == Structure::TRIANGULAR && std::is_same_v<T, bool>) {
                    // Bit rows: row i of the triangle holds columns first_col(i).. from bit 0.
                    const uint64_t* sw = src.data();
                    const uint64_t sstride = src.stride_bytes() / 8;
                    const uint64_t src_words = (cols + 63) / 64;
                    uint64_t* dw = dst.data();
                    ParallelFor(0, n, [&](size_t i) {
                        const uint64_t first = dst.first_col(i), bits = dst.row_bits(i);
                        const uint64_t* row = sw + i * sstride;
                        uint64_t* out_row = dw + dst.get_row_offset(i) / 8;
                        for (uint64_t w = 0; w * 64 < bits; ++w) {
                            const uint64_t b = first + w * 64, k = b / 64, sh = b % 64;
                            uint64_t v = row[k] >> sh;
                            if (sh && k + 1 < src_words) v |= row[k + 1] << (64 - sh);
                            const uint64_t left = bits - w * 64;
                            if (left < 64) v &= (uint64_t{1} << left) - 1;
                            out_row[w] = v;
                        }
                    });
                } else if constexpr (packed_rows && !std::is_same_v<T, bool> &&
                                     !std::is_same_v<T, complex_float16_marker>) {
                    const T* sp = src.data();
                    char* base = reinterpret_cast<char*>(dst.data());
                    ParallelFor(0, n, [&](size_t i) {
                        const uint64_t first = (S == Structure::TRIANGULAR && !diag) ? i + 1 : i;
                        if (first >= cols) return;
                        std::memcpy(base + dst.get_row_offset(i), sp + i * cols + first, (cols - first) * sizeof(T));
                    });
                    if constexpr (S != Structure::TRIANGULAR) {
                        for (uint64_t i = 0; i < n; ++i) dst.set(i, i, src.get(i, i));  // the diagonal checks
                    }
                } else {
                    for (uint64_t i = 0; i < n; ++i)
                        for (uint64_t j = 0; j < cols; ++j)
                            if (stored(i, j)) dst.set(i, j, E(src.get(i, j)));
                }
            }
        });
    });
    return out;
}

std::unique_ptr<MatrixBase> diagonal_from_vector(const VectorBase& v) {
    auto row = write_out(*as_row_matrix(v));
    const uint64_t n = row->cols();
    const DataType dt = row->get_data_type();
    auto out = make_matrix(Structure::DIAGONAL, dt, n, n, "", MatrixOptions{});
    debug_trace::set_last("diagonal_from_vector");
    visit_dtype(dt, [&](auto tag) {
        using T = typename decltype(tag)::type;
        const auto& src = static_cast<const typename DenseMatrixOf<T>::type&>(*row);
        auto& dst = static_cast<typename ClassOf<Structure::DIAGONAL, T>::type&>(*out);
        using E = structured_element_t<T>;
        for (uint64_t i = 0; i < n; ++i) dst.set_diagonal(i, E(src.get(0, i)));
    });
    return out;
}


}  // namespace pycauset

