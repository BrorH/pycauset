// Matrix ⊕ scalar operations with NumPy result-dtype semantics. See ScalarOps.hpp.
#include "pycauset/math/ScalarOps.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

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

std::string dtype_name(DataType d) {
    switch (d) {
        case DataType::BIT: return "bit";
        case DataType::INT8: return "int8";
        case DataType::INT16: return "int16";
        case DataType::INT32: return "int32";
        case DataType::INT64: return "int64";
        case DataType::UINT8: return "uint8";
        case DataType::UINT16: return "uint16";
        case DataType::UINT32: return "uint32";
        case DataType::UINT64: return "uint64";
        case DataType::FLOAT16: return "float16";
        case DataType::FLOAT32: return "float32";
        case DataType::FLOAT64: return "float64";
        case DataType::COMPLEX_FLOAT16: return "complex_float16";
        case DataType::COMPLEX_FLOAT32: return "complex64";
        case DataType::COMPLEX_FLOAT64: return "complex128";
        default: return "unknown";
    }
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
            case ScalarOpKind::Mul: return x * s;
            case ScalarOpKind::TrueDiv: return x / s;
            case ScalarOpKind::RTrueDiv: return s / x;
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
    } else {
        using V = typename R::value_type;
        return R(static_cast<V>(v), V(0));
    }
}

// Integer factor of an integer view (1 for plain matrices and for non-integer dtypes).
inline int64_t int_view_factor(const MatrixBase& a) {
    if (is_int(a.get_data_type()) && scalar::is_exact_int_factor(a.get_scalar())) {
        return static_cast<int64_t>(a.get_scalar().real());
    }
    return 1;
}

// Typed source: reads element (i, j) of a concrete input class exactly.
template <typename In, typename Src>
struct TypedReader {
    const Src& m;
    int64_t k = 1;  // integer view factor (F-042), applied with NumPy's wrapping
    In operator()(uint64_t i, uint64_t j) const {
        if constexpr (std::is_integral_v<In> && !std::is_same_v<In, bool>) {
            return k == 1 ? m.get(i, j) : scalar::wrap_scale(m.get(i, j), k);
        } else {
            return m.get(i, j);
        }
    }
};

template <typename R, typename In, typename Src>
std::unique_ptr<MatrixBase> run(const Src& src, const MatrixBase& a, ScalarOpKind op, const ScalarArg& s,
                                bool triangular_in, bool has_diag) {
    const uint64_t rows = a.rows(), cols = a.cols();
    const R sv = scalar_as<R>(s);
    TypedReader<In, Src> read{src, int_view_factor(a)};

    // Triangular results only for element types that have a Python class
    // (TriangularIntegerMatrix = int32, TriangularFloatMatrix = float64); anything else
    // would reach Python as a bare MatrixBase (F-032). Other dtypes become dense.
    constexpr bool tri_capable = std::is_same_v<R, int32_t> || std::is_same_v<R, double>;
    if constexpr (tri_capable) if (triangular_in && preserves_zero(op, s)) {
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

    auto out = std::make_unique<DenseMatrix<R>>(rows, cols, std::string());
    R* dst = out->data();
    ParallelFor(0, rows, [&](size_t i) {
        for (uint64_t j = 0; j < cols; ++j) {
            dst[i * cols + j] = apply<R>(op, convert_in<R>(read(i, j)), sv);
        }
    });
    return out;
}

// Generic (exact for |x| < 2^53) fallback for other structures: diagonal, identity,
// symmetric, ... read through the virtual complex accessor, which applies views.
template <typename R>
std::unique_ptr<MatrixBase> run_generic(const MatrixBase& a, ScalarOpKind op, const ScalarArg& s) {
    struct GenericReader {
        const MatrixBase& m;
        R operator()(uint64_t i, uint64_t j) const {
            const std::complex<double> v = m.get_element_as_complex(i, j);
            if constexpr (std::is_integral_v<R>) {
                return static_cast<R>(static_cast<int64_t>(v.real()));
            } else if constexpr (std::is_floating_point_v<R>) {
                return static_cast<R>(v.real());
            } else {
                using V = typename R::value_type;
                return R(static_cast<V>(v.real()), static_cast<V>(v.imag()));
            }
        }
    } read{a};
    const uint64_t rows = a.rows(), cols = a.cols();
    const R sv = scalar_as<R>(s);
    auto out = std::make_unique<DenseMatrix<R>>(rows, cols, std::string());
    R* dst = out->data();
    ParallelFor(0, rows, [&](size_t i) {
        for (uint64_t j = 0; j < cols; ++j) dst[i * cols + j] = apply<R>(op, read(i, j), sv);
    });
    return out;
}

// Dispatch on the concrete input class for a fixed result type R.
template <typename R>
std::unique_ptr<MatrixBase> run_for_result(const MatrixBase& a, ScalarOpKind op, const ScalarArg& s) {
    // Exact typed paths only when the input carries no lazy scalar (integers never do
    // after F-001; a float input may, and is then read through the generic accessor).
    const bool plain = !a.is_conjugated() &&
                       (a.get_scalar() == std::complex<double>(1.0, 0.0) ||
                        (is_int(a.get_data_type()) && scalar::is_exact_int_factor(a.get_scalar())));
#define PYCAUSET_TRY_DENSE(T)                                                              \
    if (auto* m = dynamic_cast<const DenseMatrix<T>*>(&a); m && plain)                    \
        return run<R, T>(*m, a, op, s, false, false);
#define PYCAUSET_TRY_TRI(T)                                                                \
    if (auto* m = dynamic_cast<const TriangularMatrix<T>*>(&a); m && plain)               \
        return run<R, T>(*m, a, op, s, true, m->has_diagonal());
    PYCAUSET_TRY_DENSE(bool)
    PYCAUSET_TRY_DENSE(int8_t) PYCAUSET_TRY_DENSE(int16_t) PYCAUSET_TRY_DENSE(int32_t) PYCAUSET_TRY_DENSE(int64_t)
    PYCAUSET_TRY_DENSE(uint8_t) PYCAUSET_TRY_DENSE(uint16_t) PYCAUSET_TRY_DENSE(uint32_t) PYCAUSET_TRY_DENSE(uint64_t)
    PYCAUSET_TRY_DENSE(float) PYCAUSET_TRY_DENSE(double)
    PYCAUSET_TRY_TRI(bool)
    PYCAUSET_TRY_TRI(int8_t) PYCAUSET_TRY_TRI(int16_t) PYCAUSET_TRY_TRI(int32_t) PYCAUSET_TRY_TRI(int64_t)
    PYCAUSET_TRY_TRI(uint8_t) PYCAUSET_TRY_TRI(uint16_t) PYCAUSET_TRY_TRI(uint32_t) PYCAUSET_TRY_TRI(uint64_t)
    PYCAUSET_TRY_TRI(float) PYCAUSET_TRY_TRI(double)
#define PYCAUSET_TRY_SYM(T)                                                                \
    if (auto* m = dynamic_cast<const SymmetricMatrix<T>*>(&a); m && plain && is_int(a.get_data_type())) \
        return run<R, T>(*m, a, op, s, false, false);
    PYCAUSET_TRY_SYM(int8_t) PYCAUSET_TRY_SYM(int16_t) PYCAUSET_TRY_SYM(int32_t) PYCAUSET_TRY_SYM(int64_t)
    PYCAUSET_TRY_SYM(uint8_t) PYCAUSET_TRY_SYM(uint16_t) PYCAUSET_TRY_SYM(uint32_t) PYCAUSET_TRY_SYM(uint64_t)
#undef PYCAUSET_TRY_SYM
#undef PYCAUSET_TRY_DENSE
#undef PYCAUSET_TRY_TRI
    return run_generic<R>(a, op, s);
}

std::unique_ptr<MatrixBase> run_exact(const MatrixBase& a, DataType out, ScalarOpKind op, const ScalarArg& s) {
    switch (out) {
        case DataType::INT8: return run_for_result<int8_t>(a, op, s);
        case DataType::INT16: return run_for_result<int16_t>(a, op, s);
        case DataType::INT32: return run_for_result<int32_t>(a, op, s);
        case DataType::INT64: return run_for_result<int64_t>(a, op, s);
        case DataType::UINT8: return run_for_result<uint8_t>(a, op, s);
        case DataType::UINT16: return run_for_result<uint16_t>(a, op, s);
        case DataType::UINT32: return run_for_result<uint32_t>(a, op, s);
        case DataType::UINT64: return run_for_result<uint64_t>(a, op, s);
        case DataType::FLOAT32: return run_for_result<float>(a, op, s);
        case DataType::FLOAT64: return run_for_result<double>(a, op, s);
        case DataType::COMPLEX_FLOAT32: return run_for_result<std::complex<float>>(a, op, s);
        case DataType::COMPLEX_FLOAT64: return run_for_result<std::complex<double>>(a, op, s);
        default:
            throw std::invalid_argument("scalar operation: unsupported result dtype " + dtype_name(out));
    }
}

}  // namespace

// Shared-storage view of an integer matrix scaled by k (nullptr when not representable).
std::unique_ptr<MatrixBase> int_scale_view(const MatrixBase& a, int64_t k) {
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
        throw std::overflow_error("Python integer out of bounds for " + dtype_name(out) +
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
            case ScalarOpKind::Add:
                if (s.kind != ScalarArg::Kind::Complex) return a.add_scalar(c.real());
                break;
            case ScalarOpKind::Sub:
                if (s.kind != ScalarArg::Kind::Complex) return a.add_scalar(-c.real());
                break;
            case ScalarOpKind::RSub:
                if (s.kind != ScalarArg::Kind::Complex) return a.multiply_scalar(-1.0)->add_scalar(c.real());
                break;
            default:
                break;
        }
    }
    if (in == DataType::COMPLEX_FLOAT16 || out == DataType::COMPLEX_FLOAT16 || in == DataType::FLOAT16 ||
        out == DataType::FLOAT16) {
        throw std::invalid_argument("scalar operation: float16/complex_float16 inputs only support same-dtype "
                                    "real scalar arithmetic (dtype " + dtype_name(in) + " -> " + dtype_name(out) + ")");
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

std::unique_ptr<MatrixBase> cast_to(const MatrixBase& a, DataType dt) {
    const uint64_t rows = a.rows(), cols = a.cols();
    if (dt == DataType::FLOAT16) {
        auto out = std::make_unique<DenseMatrix<pycauset::float16_t>>(rows, cols, std::string());
        pycauset::float16_t* dst = out->data();
        ParallelFor(0, rows, [&](size_t i) {
            for (uint64_t j = 0; j < cols; ++j) dst[i * cols + j] = pycauset::float16_t(a.get_element_as_double(i, j));
        });
        return out;
    }
    if (dt == DataType::COMPLEX_FLOAT16) {
        auto out = std::make_unique<ComplexFloat16Matrix>(rows, cols, std::string());
        for (uint64_t i = 0; i < rows; ++i) {
            for (uint64_t j = 0; j < cols; ++j) out->set(i, j, a.get_element_as_complex(i, j));
        }
        return out;
    }
    // Multiplying by one in the target type is an exact conversion (ints, floats, complex).
    return run_exact(a, dt, ScalarOpKind::Mul, ScalarArg::from_int64(1));
}

}  // namespace pycauset

