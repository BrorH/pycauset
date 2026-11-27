// Matrix ⊕ scalar operations with NumPy result-dtype semantics. See ScalarOps.hpp.
#include "pycauset/core/DebugTrace.hpp"
#include "pycauset/math/ScalarOps.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "pycauset/core/DTypeDispatch.hpp"
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

bool is_signed_int(DataType d) {
    return d == DataType::INT8 || d == DataType::INT16 || d == DataType::INT32 || d == DataType::INT64;
}
bool is_unsigned_int(DataType d) {
    return d == DataType::UINT8 || d == DataType::UINT16 || d == DataType::UINT32 || d == DataType::UINT64;
}
bool is_int(DataType d) { return is_signed_int(d) || is_unsigned_int(d); }
bool is_complex(DataType d) {
    return d == DataType::COMPLEX_FLOAT16 || d == DataType::COMPLEX_FLOAT32 || d == DataType::COMPLEX_FLOAT64;
}
bool is_real_float(DataType d) {
    return d == DataType::FLOAT16 || d == DataType::FLOAT32 || d == DataType::FLOAT64;
}
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
DataType int_dtype(bool is_signed, int bits) {
    switch (bits) {
        case 8: return is_signed ? DataType::INT8 : DataType::UINT8;
        case 16: return is_signed ? DataType::INT16 : DataType::UINT16;
        case 32: return is_signed ? DataType::INT32 : DataType::UINT32;
        default: return is_signed ? DataType::INT64 : DataType::UINT64;
    }
}
// Strong (NumPy-scalar) integer promotion of two integer dtypes, NumPy rules.
DataType promote_ints(DataType a, DataType b) {
    const bool sa = is_signed_int(a), sb = is_signed_int(b);
    const int ba = int_bits(a), bb = int_bits(b);
    if (sa == sb) return int_dtype(sa, std::max(ba, bb));
    const int b_signed = sa ? ba : bb, b_unsigned = sa ? bb : ba;
    if (b_signed > b_unsigned) return int_dtype(true, b_signed);
    if (b_unsigned < 64) return int_dtype(true, b_unsigned * 2);
    return DataType::FLOAT64;  // int64 with uint64 -> float64 (NumPy)
}

bool int_fits(const ScalarArg& s, DataType d) {
    if (d == DataType::BIT) d = DataType::INT32;  // bit arithmetic counts in int32
    const int bits = int_bits(d);
    if (is_signed_int(d)) {
        const int64_t lo = bits == 64 ? std::numeric_limits<int64_t>::min() : -(int64_t(1) << (bits - 1));
        const int64_t hi = bits == 64 ? std::numeric_limits<int64_t>::max() : (int64_t(1) << (bits - 1)) - 1;
        if (s.negative) return s.i >= lo;
        return s.u <= static_cast<uint64_t>(hi);
    }
    if (s.negative) return false;
    return bits == 64 || s.u <= ((uint64_t(1) << bits) - 1);
}

}  // namespace

bool is_integer_or_bit(DataType dt) { return dt == DataType::BIT || is_int(dt); }

// Smallest real float that can hold every value of an integer/bit dtype (NumPy rule).
static DataType float_for_int(DataType d) {
    if (d == DataType::BIT) return DataType::FLOAT16;
    const int bits = int_bits(d);
    return bits <= 8 ? DataType::FLOAT16 : bits <= 16 ? DataType::FLOAT32 : DataType::FLOAT64;
}
static int float_rank(DataType d) {
    return d == DataType::FLOAT16 ? 0 : d == DataType::FLOAT32 ? 1 : 2;
}
static DataType max_float(DataType x, DataType y) { return float_rank(x) >= float_rank(y) ? x : y; }
static DataType real_of(DataType c) {
    return c == DataType::COMPLEX_FLOAT16 ? DataType::FLOAT16
         : c == DataType::COMPLEX_FLOAT32 ? DataType::FLOAT32 : DataType::FLOAT64;
}

DataType scalar_result_dtype(DataType a, ScalarOpKind op, const ScalarArg& s) {
    const bool division = op == ScalarOpKind::TrueDiv || op == ScalarOpKind::RTrueDiv;
    const bool strong_float = s.strong && is_real_float(s.strong_dtype);
    const bool strong_int = s.strong && is_int(s.strong_dtype);

    if (is_integer_or_bit(a)) {
        if (s.kind == ScalarArg::Kind::Complex) {
            // Strong complex64 with a small int stays complex64; otherwise complex128.
            if (s.strong && s.strong_dtype == DataType::COMPLEX_FLOAT32 && float_rank(float_for_int(a)) <= 1) {
                return DataType::COMPLEX_FLOAT32;
            }
            return DataType::COMPLEX_FLOAT64;
        }
        if (strong_float) return max_float(float_for_int(a), s.strong_dtype);
        if (s.kind == ScalarArg::Kind::Float || division) {
            // Weak Python float (or true division): float64, NumPy's default float.
            if (division && strong_int) {
                const DataType base = a == DataType::BIT ? DataType::INT32 : a;
                return max_float(DataType::FLOAT64, float_for_int(promote_ints(base, s.strong_dtype)));
            }
            return DataType::FLOAT64;
        }
        const DataType base = a == DataType::BIT ? DataType::INT32 : a;
        if (strong_int) {
            // bit with a strong int scalar: NumPy gives the scalar's dtype.
            return a == DataType::BIT ? s.strong_dtype : promote_ints(base, s.strong_dtype);
        }
        return base;
    }

    // Float or complex matrix.
    DataType real = is_complex(a) ? real_of(a) : a;
    if (strong_float) real = max_float(real, s.strong_dtype);
    if (strong_int) real = max_float(real, float_for_int(s.strong_dtype));
    const bool complex_result = is_complex(a) || s.kind == ScalarArg::Kind::Complex;
    if (complex_result && s.strong && is_complex(s.strong_dtype)) real = max_float(real, real_of(s.strong_dtype));
    return complex_result ? complex_of(real) : real;
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

template <typename R>
R apply(ScalarOpKind op, R x, R s) {
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
            // to_dense / dtype conversions are "multiply by 1" (F-113).
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
    if (is_int(a.get_data_type()) && scalar::is_exact_int_factor(a.get_scalar())) {
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

template <typename R, typename In, typename Src>
std::unique_ptr<MatrixBase> run(const Src& src, const MatrixBase& a, ScalarOpKind op, const ScalarArg& s,
                                bool triangular_in, bool has_diag, bool keep_structure) {
    const uint64_t rows = a.rows(), cols = a.cols();
    const R sv = scalar_as<R>(s);
    TypedReader<In, Src> read{src, int_view_factor(a), is_real_float(a.get_data_type()) ? a.get_scalar().real() : 1.0};

    // Triangular results for every real element type (each has a Python class since the
    // structure classes, F-032); complex results become dense.
    constexpr bool tri_capable = !std::is_same_v<R, std::complex<float>> && !std::is_same_v<R, std::complex<double>>;
    if constexpr (tri_capable) if (keep_structure && triangular_in && preserves_zero(op, s)) {
        auto out = std::make_unique<TriangularMatrix<R>>(rows, std::string(), has_diag);
        const bool lower = a.is_transposed();
        out->set_transposed(lower);
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
                const size_t bytes = out->shared_mapper()->get_data_size();
                if (!a.is_subview() && a.get_scalar() == std::complex<double>(1.0, 0.0) && !a.is_conjugated() &&
                    !(anti && a.is_transposed()) && src.shared_mapper()->get_data_size() == bytes) {
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
            bool scaled = false;
            if constexpr (std::is_integral_v<In>) scaled = read.k != 1;
            else if constexpr (std::is_floating_point_v<In>) scaled = read.f != 1.0;
            auto loop = [&](auto op_c) {
                constexpr ScalarOpKind O = decltype(op_c)::value;
                auto run = [&](auto scaled_c) {
                    for_each_span(rows * cols, [&](uint64_t begin, uint64_t end) {
                        // Locals and restrict: int8 / uint8 are char types, which may alias
                        // anything, so through the captures every store could change the
                        // operands and the loop stayed scalar (int8 + 7 at 0.1x NumPy).
                        const In* __restrict src = p + begin;
                        R* __restrict out = dst + begin;
                        const R s_local = sv;
                        const int64_t k_local = read.k;
                        const double f_local = read.f;
                        const uint64_t len = end - begin;
                        for (uint64_t q = 0; q < len; ++q) {
                            In x = src[q];
                            if constexpr (decltype(scaled_c)::value) {
                                if constexpr (std::is_integral_v<In>) x = scalar::wrap_scale(x, k_local);
                                else if constexpr (std::is_floating_point_v<In>) x = static_cast<In>(static_cast<double>(x) * f_local);
                            }
                            out[q] = apply<R>(O, convert_in<R>(x), s_local);
                        }
                        (void)k_local;
                        (void)f_local;
                    });
                };
                if (scaled) run(std::true_type{});
                else run(std::false_type{});
            };
            switch (op) {
                case ScalarOpKind::Add: loop(std::integral_constant<ScalarOpKind, ScalarOpKind::Add>{}); break;
                case ScalarOpKind::Sub: loop(std::integral_constant<ScalarOpKind, ScalarOpKind::Sub>{}); break;
                case ScalarOpKind::RSub: loop(std::integral_constant<ScalarOpKind, ScalarOpKind::RSub>{}); break;
                case ScalarOpKind::Mul: loop(std::integral_constant<ScalarOpKind, ScalarOpKind::Mul>{}); break;
                case ScalarOpKind::TrueDiv: loop(std::integral_constant<ScalarOpKind, ScalarOpKind::TrueDiv>{}); break;
                case ScalarOpKind::RTrueDiv: loop(std::integral_constant<ScalarOpKind, ScalarOpKind::RTrueDiv>{}); break;
            }
            return out;
        }
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
    for_each_block(rows, cols, [&](uint64_t idx, uint64_t, uint64_t) {
        R v;
        if constexpr (std::is_same_v<W, std::complex<double>>) v = from_logical<R>(src[idx]);
        else v = convert_in<R>(src[idx]);
        dst[idx] = apply<R>(op, v, sv);
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
    const bool plain = sc == std::complex<double>(1.0, 0.0) ? (!a.is_conjugated() || !is_complex(dt))
                     : is_int(dt) ? (!a.is_conjugated() && scalar::is_exact_int_factor(sc))
                     : (is_real_float(dt) && sc.imag() == 0.0);
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
    }
    PYCAUSET_TRY_TRI(bool)
    PYCAUSET_TRY_TRI(int8_t) PYCAUSET_TRY_TRI(int16_t) PYCAUSET_TRY_TRI(int32_t) PYCAUSET_TRY_TRI(int64_t)
    PYCAUSET_TRY_TRI(uint8_t) PYCAUSET_TRY_TRI(uint16_t) PYCAUSET_TRY_TRI(uint32_t) PYCAUSET_TRY_TRI(uint64_t)
    PYCAUSET_TRY_TRI(float) PYCAUSET_TRY_TRI(double)
#define PYCAUSET_TRY_SYM(T)                                                                \
    if (auto* m = dynamic_cast<const SymmetricMatrix<T>*>(&a); m && plain && is_int(a.get_data_type())) \
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
    const DataType out = scalar_result_dtype(in, op, s);

    if (is_integer_or_bit(in) && s.kind == ScalarArg::Kind::Int && is_int(out) && !int_fits(s, out)) {
        throw std::overflow_error("Python integer out of bounds for " + std::string(dtype_name(out)) +
                                  " (scalar does not fit the matrix dtype)");
    }

    // Integer input, integer factor, dtype kept: an O(1) view sharing storage (LAZY.md L1,
    // F-042). Dense and triangular only (their readers apply integer factors); the folded
    // factor must stay exact, otherwise compute eagerly.
    if (op == ScalarOpKind::Mul && is_int(in) && out == in && s.kind == ScalarArg::Kind::Int) {
        const int64_t k = s.negative ? s.i : static_cast<int64_t>(s.u);  // mod 2^64 is exact here
        if (auto view = int_scale_view(a, k)) return view;
    }

    // Float/complex input whose dtype is kept: the classes' O(1) scalar-view paths.
    if (!is_integer_or_bit(in) && out == in && op != ScalarOpKind::RTrueDiv) {
        const std::complex<double> c = s.as_complex();
        switch (op) {
            case ScalarOpKind::Mul:
                return s.kind == ScalarArg::Kind::Complex ? a.multiply_scalar(c) : a.multiply_scalar(c.real());
            case ScalarOpKind::TrueDiv:
                return s.kind == ScalarArg::Kind::Complex ? a.multiply_scalar(1.0 / c) : a.multiply_scalar(1.0 / c.real());
            default:
                break;  // + and - compute below, through the one typed path (F-086)
        }
    }
    // Half-precision results: compute in float64 / complex128 (exact for these operands and
    // scalars' sums and products) and round once.
    if (out == DataType::FLOAT16 || out == DataType::COMPLEX_FLOAT16) {
        const DataType wide = out == DataType::FLOAT16 ? DataType::FLOAT64 : DataType::COMPLEX_FLOAT64;
        return cast_to(*run_exact(a, wide, op, s), out);
    }
    return run_exact(a, out, op, s);
}

std::unique_ptr<MatrixBase> materialize(const MatrixBase& a) {
    // Values through the exact view-aware reader into fresh storage (factor 1).
    return run_exact(a, a.get_data_type(), ScalarOpKind::Add, ScalarArg::from_int64(0));
}

std::unique_ptr<MatrixBase> negate(const MatrixBase& a) {
    if (a.get_data_type() == DataType::BIT) {
        throw std::invalid_argument("negation of a bit/bool matrix is not supported (as in NumPy); use bitwise_not");
    }
    if (is_int(a.get_data_type())) {
        // -A is the view factor -1 (unsigned negation wraps like NumPy: same arithmetic).
        if (auto view = int_scale_view(a, -1)) return view;
        // Unsigned negation wraps (NumPy): multiply by -1 in the unsigned type.
        ScalarArg minus_one = ScalarArg::from_int64(-1);
        if (is_unsigned_int(a.get_data_type())) {
            return run_exact(a, a.get_data_type(), ScalarOpKind::RSub, ScalarArg::from_int64(0));
        }
        return run_exact(a, a.get_data_type(), ScalarOpKind::Mul, minus_one);
    }
    return a.multiply_scalar(-1.0);
}

bool is_plain(const MatrixBase& a) {
    return !a.is_transposed() && !a.is_subview() && a.get_scalar() == std::complex<double>(1.0, 0.0) &&
           !(a.is_conjugated() && is_complex(a.get_data_type()));
}

std::unique_ptr<MatrixBase> write_out(const MatrixBase& a) {
    debug_trace::set_last("write_out");
    const DataType dt = a.get_data_type();
    const uint64_t rows = a.rows(), cols = a.cols();
    if (dt == DataType::BIT) {
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
    if (dt == DataType::FLOAT16 || dt == DataType::COMPLEX_FLOAT16) return cast_to(a, dt);  // dense
    return run_exact(a, dt, ScalarOpKind::Mul, ScalarArg::from_int64(1), /*keep_structure=*/false);
}

std::unique_ptr<MatrixBase> cast_to(const MatrixBase& a, DataType dt) {
    const uint64_t rows = a.rows(), cols = a.cols();
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
    return run_exact(a, dt, ScalarOpKind::Mul, ScalarArg::from_int64(1));
}

// The stored region of a (view applied) copied into a new matrix of structure s in a's dtype.
// Exact for every dtype: one typed read of a plain dense copy, one typed write per stored
// entry. Entries outside the stored region are not read (the caller vouches for them, e.g.
// pc.symmetric validates its input first).
std::unique_ptr<MatrixBase> pack_as(const MatrixBase& a, Structure s, bool has_diagonal) {
    MatrixOptions opts;
    opts.has_diagonal = has_diagonal;
    const DataType dt = a.get_data_type();
    const uint64_t n = a.rows();
    if (s != Structure::DENSE) detail::require_square(s, a.rows(), a.cols());
    auto out = make_matrix(s, dt, a.rows(), a.cols(), "", opts);
    if (s == Structure::IDENTITY) return out;
    auto d = write_out(a);
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
        const auto& src = static_cast<const typename DenseMatrixOf<T>::type&>(*d);
        visit_structure(s, [&](auto st) {
            constexpr Structure S = decltype(st)::value;
            if constexpr (offered<S, T>()) {
                auto& dst = static_cast<typename ClassOf<S, T>::type&>(*out);
                using E = structured_element_t<T>;
                for (uint64_t i = 0; i < n; ++i)
                    for (uint64_t j = 0; j < a.cols(); ++j)
                        if (stored(i, j)) {
                            if constexpr (S == Structure::DIAGONAL) dst.set_diagonal(i, E(src.get(i, j)));
                            else dst.set(i, j, E(src.get(i, j)));
                        }
            }
        });
    });
    return out;
}

std::unique_ptr<MatrixBase> to_dense(const MatrixBase& a, DataType dt) {
    if (dt == DataType::FLOAT16 || dt == DataType::COMPLEX_FLOAT16) return cast_to(a, dt);  // dense
    if (dt == DataType::BIT) throw std::invalid_argument("to_dense: bit target");
    return run_exact(a, dt, ScalarOpKind::Mul, ScalarArg::from_int64(1), /*keep_structure=*/false);
}

}  // namespace pycauset

