#pragma once
#include <complex>

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace pycauset {

namespace detail {

inline uint32_t float_to_bits(float v) {
    uint32_t out;
    std::memcpy(&out, &v, sizeof(out));
    return out;
}

inline float bits_to_float(uint32_t v) {
    float out;
    std::memcpy(&out, &v, sizeof(out));
    return out;
}

inline uint16_t float_to_half_bits(float f) {
    const uint32_t x = float_to_bits(f);

    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exp = (x >> 23) & 0xFFu;
    const uint32_t mant = x & 0x7FFFFFu;

    // NaN / Inf
    if (exp == 0xFFu) {
        if (mant == 0) {
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
        // Preserve some mantissa bits for NaN payload.
        uint16_t nan_mant = static_cast<uint16_t>(mant >> 13);
        if (nan_mant == 0) nan_mant = 1;
        return static_cast<uint16_t>(sign | 0x7C00u | nan_mant);
    }

    // Re-bias exponent from float32 (127) to float16 (15)
    const int32_t exp16 = static_cast<int32_t>(exp) - 127 + 15;

    // Subnormal / underflow
    if (exp16 <= 0) {
        // Too small => signed zero
        if (exp16 < -10) {
            return static_cast<uint16_t>(sign);
        }

        // Subnormal: implicit leading 1 + mantissa
        uint32_t mant32 = mant | 0x800000u;
        const int32_t shift = 14 - exp16; // 1..24

        uint32_t half_mant = mant32 >> shift;

        // Round to nearest-even
        const uint32_t round_bit = 1u << (shift - 1);
        const uint32_t round_mask = round_bit - 1u;
        const uint32_t round_src = mant32;
        const bool round = (round_src & round_bit) && ((round_src & round_mask) || (half_mant & 1u));
        if (round) {
            ++half_mant;
        }

        return static_cast<uint16_t>(sign | static_cast<uint16_t>(half_mant));
    }

    // Overflow => Inf
    if (exp16 >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    // Normalized
    uint32_t half_exp = static_cast<uint32_t>(exp16);
    uint32_t half_mant = mant >> 13;

    // Round to nearest-even using lower 13 bits
    const uint32_t round_src = mant;
    const uint32_t round_bit = 1u << 12;
    const uint32_t round_mask = round_bit - 1u;
    const bool round = (round_src & round_bit) && ((round_src & round_mask) || (half_mant & 1u));
    if (round) {
        ++half_mant;
        if (half_mant == 0x400u) {
            // Mantissa overflow
            half_mant = 0;
            ++half_exp;
            if (half_exp >= 31u) {
                return static_cast<uint16_t>(sign | 0x7C00u);
            }
        }
    }

    return static_cast<uint16_t>(sign | (half_exp << 10) | (half_mant & 0x3FFu));
}

inline float half_bits_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;

    uint32_t out;

    if (exp == 0) {
        if (mant == 0) {
            out = sign;
            return bits_to_float(out);
        }

        // Subnormal: normalize
        exp = 1;
        while ((mant & 0x400u) == 0) {
            mant <<= 1;
            --exp;
        }
        mant &= 0x3FFu;

        const uint32_t exp32 = (exp + (127 - 15)) << 23;
        const uint32_t mant32 = mant << 13;
        out = sign | exp32 | mant32;
        return bits_to_float(out);
    }

    if (exp == 31) {
        // Inf / NaN
        const uint32_t exp32 = 0xFFu << 23;
        const uint32_t mant32 = mant << 13;
        out = sign | exp32 | mant32;
        return bits_to_float(out);
    }

    const uint32_t exp32 = (exp + (127 - 15)) << 23;
    const uint32_t mant32 = mant << 13;
    out = sign | exp32 | mant32;
    return bits_to_float(out);
}

} // namespace detail

struct float16_t {
    uint16_t bits{0};

    float16_t() = default;
    explicit constexpr float16_t(uint16_t raw_bits) : bits(raw_bits) {}

    // Avoid ambiguous conversions for integer literals (e.g. static_cast<float16_t>(0)).
    float16_t(int v) : bits(detail::float_to_half_bits(static_cast<float>(v))) {}
    float16_t(unsigned int v) : bits(detail::float_to_half_bits(static_cast<float>(v))) {}

    float16_t(float v) : bits(detail::float_to_half_bits(v)) {}
    float16_t(double v) : bits(detail::float_to_half_bits(static_cast<float>(v))) {}

    operator float() const { return detail::half_bits_to_float(bits); }
};

static_assert(sizeof(float16_t) == sizeof(uint16_t), "float16_t must be exactly 2 bytes");
static_assert(std::is_trivially_copyable_v<float16_t>, "float16_t must be trivially copyable");

inline float16_t operator+(float16_t a, float16_t b) { return float16_t(static_cast<float>(a) + static_cast<float>(b)); }
inline float16_t operator-(float16_t a, float16_t b) { return float16_t(static_cast<float>(a) - static_cast<float>(b)); }
inline float16_t operator*(float16_t a, float16_t b) { return float16_t(static_cast<float>(a) * static_cast<float>(b)); }
inline float16_t operator/(float16_t a, float16_t b) { return float16_t(static_cast<float>(a) / static_cast<float>(b)); }

inline bool operator==(float16_t a, float16_t b) { return static_cast<float>(a) == static_cast<float>(b); }
inline bool operator!=(float16_t a, float16_t b) { return !(a == b); }

// Interleaved complex float16 element: the storage of the structured complex_float16 classes
// (TriangularMatrix<complex_half>, SymmetricMatrix<complex_half>, ...; review N.5.5). Dense
// complex_float16 keeps its split planes (ComplexFloat16Matrix). Values convert through
// std::complex; arithmetic happens in the readers and kernels, not here.
struct complex_half {
    float16_t re{}, im{};

    complex_half() = default;
    complex_half(int v) : re(v), im(0) {}  // T(0), T(1)
    complex_half(double v) : re(v), im(0) {}
    complex_half(float16_t r, float16_t i) : re(r), im(i) {}
    template <typename V>
    complex_half(const std::complex<V>& z) : re(static_cast<double>(z.real())), im(static_cast<double>(z.imag())) {}

    float real() const { return re; }
    float imag() const { return im; }
    operator std::complex<float>() const { return {static_cast<float>(re), static_cast<float>(im)}; }
    operator std::complex<double>() const { return {static_cast<double>(static_cast<float>(re)), static_cast<double>(static_cast<float>(im))}; }
};

static_assert(sizeof(complex_half) == 2 * sizeof(uint16_t), "complex_half must be exactly 4 bytes");

inline complex_half operator-(complex_half a) {
    return {float16_t(-static_cast<float>(a.re)), float16_t(-static_cast<float>(a.im))};
}
inline bool operator==(complex_half a, complex_half b) { return a.re == b.re && a.im == b.im; }
inline bool operator!=(complex_half a, complex_half b) { return !(a == b); }
inline complex_half conj(complex_half a) { return {a.re, float16_t(-static_cast<float>(a.im))}; }

} // namespace pycauset
