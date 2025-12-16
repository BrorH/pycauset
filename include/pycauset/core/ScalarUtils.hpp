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

} // namespace pycauset::scalar
