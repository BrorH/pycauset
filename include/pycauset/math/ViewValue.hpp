#pragma once

#include <complex>
#include <cstdint>
#include <type_traits>

#include "pycauset/core/Float16.hpp"
#include "pycauset/core/ScalarUtils.hpp"

namespace pycauset {

// The one definition of an element's logical value (review PLAN N.2; see ViewRead.hpp):
//
//     value = conj?(raw * factor)
//
// Integer storage with an integer factor wraps like NumPy (F-042); real storage with a real
// factor multiplies in real arithmetic (0.0 * (-1 - 0j) must stay -0.0, as NumPy scales a
// real array). The conjugation applies to the scaled value: a conj view with a complex
// factor is NOT conj(raw) * factor (StridedReader did that: m.conj() * 2j + 0 gave -4-2j for
// 4+2j, F-086). Every typed reader calls this; do not re-derive it.
template <typename T>
inline std::complex<double> view_value(const T& raw, std::complex<double> s, bool conj) {
    std::complex<double> z;
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
        if (s == std::complex<double>(1.0, 0.0)) z = {static_cast<double>(raw), 0.0};
        else if (scalar::is_exact_int_factor(s))
            z = {static_cast<double>(scalar::wrap_scale(raw, static_cast<int64_t>(s.real()))), 0.0};
        else z = static_cast<double>(raw) * s;
    } else if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, bool> ||
                         std::is_same_v<T, pycauset::float16_t>) {
        const double r = scalar::to_complex_double(raw).real();
        z = s.imag() == 0.0 ? std::complex<double>(r * s.real(), 0.0) : r * s;
    } else {
        z = scalar::to_complex_double(raw);
        if (s != std::complex<double>(1.0, 0.0)) z *= s;
    }
    return conj ? std::conj(z) : z;
}

}  // namespace pycauset
