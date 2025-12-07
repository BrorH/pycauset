#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>

namespace pycauset {

// Simple Float16 wrapper for storage
struct Float16 {
    uint16_t data;

    Float16() : data(0) {}
    // Explicitly handle integer types to avoid ambiguity
    Float16(int i) { data = float_to_half(static_cast<float>(i)); }
    Float16(long long i) { data = float_to_half(static_cast<float>(i)); }
    Float16(size_t i) { data = float_to_half(static_cast<float>(i)); }
    
    // Keep these
    Float16(uint16_t v) : data(v) {}
    Float16(float f) {
        data = float_to_half(f);
    }
    Float16(double d) {
        data = float_to_half(static_cast<float>(d));
    }

    // Only provide conversion to float to avoid ambiguity
    operator float() const {
        return half_to_float(data);
    }
    
    // Explicit operators to handle accumulation in DenseMatrix
    Float16& operator+=(const Float16& other) {
        *this = Float16(static_cast<float>(*this) + static_cast<float>(other));
        return *this;
    }
    Float16& operator-=(const Float16& other) {
        *this = Float16(static_cast<float>(*this) - static_cast<float>(other));
        return *this;
    }
    Float16& operator*=(const Float16& other) {
        *this = Float16(static_cast<float>(*this) * static_cast<float>(other));
        return *this;
    }
    Float16& operator/=(const Float16& other) {
        *this = Float16(static_cast<float>(*this) / static_cast<float>(other));
        return *this;
    }

    // Comparison
    bool operator==(const Float16& other) const { return data == other.data; }
    bool operator!=(const Float16& other) const { return data != other.data; }
    bool operator<(const Float16& other) const { return static_cast<float>(*this) < static_cast<float>(other); }
    bool operator>(const Float16& other) const { return static_cast<float>(*this) > static_cast<float>(other); }
    bool operator<=(const Float16& other) const { return static_cast<float>(*this) <= static_cast<float>(other); }
    bool operator>=(const Float16& other) const { return static_cast<float>(*this) >= static_cast<float>(other); }

    // Conversion implementation (IEEE 754)
    static uint16_t float_to_half(float x) {
        uint32_t f;
        std::memcpy(&f, &x, sizeof(float));
        
        uint32_t sign = (f >> 31) & 0x1;
        uint32_t exp = (f >> 23) & 0xFF;
        uint32_t mant = f & 0x7FFFFF;

        uint16_t h_sign = (uint16_t)(sign << 15);
        uint16_t h_exp, h_mant;

        if (exp == 0) {
            // Zero or Subnormal
            h_exp = 0;
            h_mant = 0; // Flush subnormals to zero for speed
        } else if (exp == 0xFF) {
            // Inf or NaN
            h_exp = 0x1F;
            h_mant = (mant != 0) ? 0x200 : 0;
        } else {
            // Normalized
            int new_exp = (int)exp - 127 + 15;
            if (new_exp <= 0) {
                // Underflow to zero
                h_exp = 0;
                h_mant = 0;
            } else if (new_exp >= 0x1F) {
                // Overflow to Inf
                h_exp = 0x1F;
                h_mant = 0;
            } else {
                h_exp = (uint16_t)new_exp;
                h_mant = (uint16_t)(mant >> 13);
            }
        }
        
        return h_sign | (h_exp << 10) | h_mant;
    }

    static float half_to_float(uint16_t h) {
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;

        uint32_t f_sign = sign << 31;
        uint32_t f_exp, f_mant;

        if (exp == 0) {
            if (mant == 0) {
                // Zero
                f_exp = 0;
                f_mant = 0;
            } else {
                // Subnormal - flush to zero or handle?
                // Let's just return 0 for simplicity/speed in this context
                f_exp = 0;
                f_mant = 0;
            }
        } else if (exp == 0x1F) {
            // Inf or NaN
            f_exp = 0xFF;
            f_mant = (mant != 0) ? 0x400000 : 0;
        } else {
            // Normalized
            f_exp = exp + 127 - 15;
            f_mant = mant << 13;
        }

        uint32_t f = f_sign | (f_exp << 23) | f_mant;
        float val;
        std::memcpy(&val, &f, sizeof(float));
        return val;
    }
};

}
