#pragma once

#include <complex>
#include <cstdint>
#include <type_traits>

#include "pycauset/core/Float16.hpp"
#include "pycauset/core/ScalarUtils.hpp"

namespace pycauset {

// Complex element types of matrix storage (std::complex and the interleaved complex_half).
template <typename T>
inline constexpr bool is_complex_storage_v = std::is_same_v<T, std::complex<float>> ||
                                             std::is_same_v<T, std::complex<double>> ||
                                             std::is_same_v<T, complex_half>;

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

// Two view factors combined (A * s1 then * s2, the factors of a product's operands). Real
// factors multiply in real arithmetic: as complex numbers, (1 + 0i) * (inf + 0i) is inf + nan i,
// so `A * inf` of a real matrix got a complex factor (review 3.6 C-4).
inline std::complex<double> compose_factors(std::complex<double> a, std::complex<double> b) {
    if (a.imag() == 0.0 && b.imag() == 0.0) return {a.real() * b.real(), 0.0};
    return a * b;
}

}  // namespace pycauset
