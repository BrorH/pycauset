#pragma once

#include "pycauset/core/Float16.hpp"

#include <complex>
#include <type_traits>

namespace pycauset::scalar {

inline std::complex<double> to_complex_double(double v) { return {v, 0.0}; }
inline std::complex<double> to_complex_double(float v) { return {static_cast<double>(v), 0.0}; }
inline std::complex<double> to_complex_double(pycauset::float16_t v) { return {static_cast<double>(v), 0.0}; }
inline std::complex<double> to_complex_double(int32_t v) { return {static_cast<double>(v), 0.0}; }
inline std::complex<double> to_complex_double(int16_t v) { return {static_cast<double>(v), 0.0}; }
inline std::complex<double> to_complex_double(bool v) { return {v ? 1.0 : 0.0, 0.0}; }

template <typename T, typename = std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>>
inline std::complex<double> to_complex_double(T v) {
    return {static_cast<double>(v), 0.0};
}

template <typename T>
inline std::complex<double> to_complex_double(const std::complex<T>& v) {
    return {static_cast<double>(v.real()), static_cast<double>(v.imag())};
}

template <typename T>
inline T from_double(double v) {
    if constexpr (std::is_same_v<T, pycauset::float16_t>) {
        return pycauset::float16_t(v);
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return std::complex<float>(static_cast<float>(v), 0.0f);
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        return std::complex<double>(v, 0.0);
    } else {
        return static_cast<T>(v);
    }
}

template <typename T>
inline T from_complex_double(const std::complex<double>& z) {
    if constexpr (std::is_same_v<T, pycauset::float16_t>) {
        return pycauset::float16_t(z.real());
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return std::complex<float>(static_cast<float>(z.real()), static_cast<float>(z.imag()));
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        return z;
    } else {
        return static_cast<T>(z.real());
    }
}


// ---------------------------------------------------------------------------
// Wrapping arithmetic (F-006). Integer + - * use two's-complement wraparound,
// matching NumPy and the documented PyCauset policy, but WITHOUT signed-overflow
// undefined behaviour: the operation is carried out in the unsigned type of the
// (promoted) result, where wraparound is defined, then converted back (well
// defined modulo 2^N since C++20). This also covers small types promoted to int
// (e.g. uint16 * uint16). Floating-point and complex operands are unaffected.
// ---------------------------------------------------------------------------
namespace detail {
template <typename A, typename B, typename Op>
constexpr auto wrapping_apply(A a, B b, Op op) {
    using R = decltype(op(a, b));
    if constexpr (std::is_integral_v<R> && !std::is_same_v<R, bool>) {
        using U = std::make_unsigned_t<R>;
        return static_cast<R>(op(static_cast<U>(static_cast<R>(a)), static_cast<U>(static_cast<R>(b))));
    } else {
        return op(a, b);
    }
}
}  // namespace detail

struct WrapPlus {
    template <typename A, typename B>
    constexpr auto operator()(A a, B b) const { return detail::wrapping_apply(a, b, [](auto x, auto y) { return x + y; }); }
};
struct WrapMinus {
    template <typename A, typename B>
    constexpr auto operator()(A a, B b) const { return detail::wrapping_apply(a, b, [](auto x, auto y) { return x - y; }); }
};
struct WrapMultiplies {
    template <typename A, typename B>
    constexpr auto operator()(A a, B b) const { return detail::wrapping_apply(a, b, [](auto x, auto y) { return x * y; }); }
};

// --- Integer scale factors on views (LAZY.md L1, F-042) ---------------------------------
// An integer view stores its factor k in the object's complex<double> scalar slot. It is
// only ever an integer with |k| <= 2^53, so the double is exact.
inline constexpr double kMaxExactIntFactor = 9007199254740992.0;  // 2^53

inline bool is_exact_int_factor(std::complex<double> s) {
    return s.imag() == 0.0 && s.real() == static_cast<double>(static_cast<int64_t>(s.real())) &&
           s.real() <= kMaxExactIntFactor && s.real() >= -kMaxExactIntFactor;
}

// x * k with NumPy's wrapping semantics for integer type U (mod 2^bits(U)).
template <typename U>
constexpr U wrap_scale(U x, int64_t k) {
    if constexpr (std::is_same_v<U, bool> || !std::is_integral_v<U>) {
        return x;  // only integer types carry integer view factors
    } else {
        return static_cast<U>(static_cast<uint64_t>(static_cast<int64_t>(x)) * static_cast<uint64_t>(k));
    }
}

// Folds two integer factors (mod 2^64). Returns false when the result is not exactly
// representable as a view factor; the caller then computes eagerly.
inline bool fold_int_factors(std::complex<double> a, int64_t k, int64_t& out) {
    if (!is_exact_int_factor(a)) return false;
    const int64_t f = static_cast<int64_t>(static_cast<uint64_t>(static_cast<int64_t>(a.real())) * static_cast<uint64_t>(k));
    if (static_cast<double>(f) > kMaxExactIntFactor || static_cast<double>(f) < -kMaxExactIntFactor) return false;
    out = f;
    return true;
}

} // namespace pycauset::scalar

#include <atomic>
namespace pycauset {
// Global switch for lazy results (views, closures). Off = everything computed eagerly; the
// test oracle (LAZY.md section 10) runs expressions both ways and compares.
inline std::atomic<bool>& lazy_enabled() {
    static std::atomic<bool> flag{true};
    return flag;
}
}  // namespace pycauset
