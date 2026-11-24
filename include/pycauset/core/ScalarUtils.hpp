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

} // namespace pycauset::scalar
