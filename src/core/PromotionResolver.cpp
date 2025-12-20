#include "pycauset/core/PromotionResolver.hpp"

#include <algorithm>
#include <stdexcept>

namespace pycauset::promotion {

namespace {
thread_local PrecisionMode g_precision_mode = PrecisionMode::Lowest;

inline bool is_real_float(DataType dt) {
    return dt == DataType::FLOAT16 || dt == DataType::FLOAT32 || dt == DataType::FLOAT64;
}

inline bool is_complex_float(DataType dt) {
    return dt == DataType::COMPLEX_FLOAT16 || dt == DataType::COMPLEX_FLOAT32 || dt == DataType::COMPLEX_FLOAT64;
}

inline bool is_float_like(DataType dt) {
    return is_real_float(dt) || is_complex_float(dt);
}

inline bool is_signed_int(DataType dt) {
    return dt == DataType::INT8 || dt == DataType::INT16 || dt == DataType::INT32 || dt == DataType::INT64;
}

inline bool is_unsigned_int(DataType dt) {
    return dt == DataType::UINT8 || dt == DataType::UINT16 || dt == DataType::UINT32 || dt == DataType::UINT64;
}

inline bool is_int_like(DataType dt) {
    return is_signed_int(dt) || is_unsigned_int(dt);
}

inline bool is_bit(DataType dt) {
    return dt == DataType::BIT;
}

inline uint8_t int_rank_bits(DataType dt) {
    switch (dt) {
        case DataType::INT8:
        case DataType::UINT8:
            return 8;
        case DataType::INT16:
        case DataType::UINT16:
            return 16;
        case DataType::INT32:
        case DataType::UINT32:
            return 32;
        case DataType::INT64:
        case DataType::UINT64:
            return 64;
        default:
            return 0;
    }
}

inline DataType signed_from_bits(uint8_t bits) {
    switch (bits) {
        case 8:
            return DataType::INT8;
        case 16:
            return DataType::INT16;
        case 32:
            return DataType::INT32;
        case 64:
            return DataType::INT64;
        default:
            throw std::invalid_argument("resolve_promotion: invalid signed int width");
    }
}

inline DataType unsigned_from_bits(uint8_t bits) {
    switch (bits) {
        case 8:
            return DataType::UINT8;
        case 16:
            return DataType::UINT16;
        case 32:
            return DataType::UINT32;
        case 64:
            return DataType::UINT64;
        default:
            throw std::invalid_argument("resolve_promotion: invalid unsigned int width");
    }
}

inline uint8_t round_up_int_bits(uint8_t bits) {
    if (bits <= 8) return 8;
    if (bits <= 16) return 16;
    if (bits <= 32) return 32;
    if (bits <= 64) return 64;
    return 0;
}

inline uint8_t float_rank(DataType dt) {
    switch (dt) {
        case DataType::FLOAT16:
        case DataType::COMPLEX_FLOAT16:
            return 16;
        case DataType::FLOAT32:
        case DataType::COMPLEX_FLOAT32:
            return 32;
        case DataType::FLOAT64:
        case DataType::COMPLEX_FLOAT64:
            return 64;
        default:
            return 0;
    }
}

inline DataType complex_from_rank(uint8_t r) {
    switch (r) {
        case 16:
            return DataType::COMPLEX_FLOAT16;
        case 32:
            return DataType::COMPLEX_FLOAT32;
        case 64:
            return DataType::COMPLEX_FLOAT64;
        default:
            throw std::invalid_argument("resolve_promotion: invalid float rank for complex");
    }
}

inline DataType smallest_float(DataType a, DataType b) {
    // Caller guarantees at least one is float.
    if (a == DataType::FLOAT16 || b == DataType::FLOAT16) return DataType::FLOAT16;
    if (a == DataType::FLOAT32 || b == DataType::FLOAT32) return DataType::FLOAT32;
    return DataType::FLOAT64;
}

inline DataType largest_float(DataType a, DataType b) {
    // Caller guarantees both are float.
    if (a == DataType::FLOAT64 || b == DataType::FLOAT64) return DataType::FLOAT64;
    if (a == DataType::FLOAT32 || b == DataType::FLOAT32) return DataType::FLOAT32;
    return DataType::FLOAT16;
}

inline uint8_t choose_rank(uint8_t ra, uint8_t rb, PrecisionMode mode) {
    if (mode == PrecisionMode::Highest) {
        return static_cast<uint8_t>(std::max(ra, rb));
    }
    return static_cast<uint8_t>(std::min(ra, rb));
}
}

PrecisionMode get_precision_mode() {
    return g_precision_mode;
}

void set_precision_mode(PrecisionMode mode) {
    g_precision_mode = mode;
}

Decision resolve(BinaryOp op, DataType a, DataType b) {
    if (a == DataType::UNKNOWN || b == DataType::UNKNOWN) {
        throw std::invalid_argument("resolve_promotion: UNKNOWN dtype");
    }

    const PrecisionMode mode = get_precision_mode();

    // Complex dominates real float/int/bit for supported matrix-matrix ops.
    // Precision rule: underpromote to the smallest participating float precision.
    if (is_complex_float(a) || is_complex_float(b)) {
        switch (op) {
            case BinaryOp::Add:
            case BinaryOp::Subtract:
            case BinaryOp::ElementwiseMultiply:
            case BinaryOp::Divide:
            case BinaryOp::Matmul: {
                const uint8_t ra = float_rank(a);
                const uint8_t rb = float_rank(b);
                uint8_t r = 0;
                if (ra && rb) {
                    r = choose_rank(ra, rb, mode);
                } else {
                    r = static_cast<uint8_t>(std::max(ra, rb));
                }
                if (r == 0) {
                    // One side is complex, so this should never happen.
                    r = 64;
                }
                return Decision{complex_from_rank(r), false, DataType::UNKNOWN};
            }
            case BinaryOp::MatrixVectorMultiply:
            case BinaryOp::VectorMatrixMultiply:
            case BinaryOp::OuterProduct: {
                const uint8_t ra = float_rank(a);
                const uint8_t rb = float_rank(b);
                uint8_t r = 0;
                if (ra && rb) {
                    r = choose_rank(ra, rb, mode);
                } else {
                    r = static_cast<uint8_t>(std::max(ra, rb));
                }
                if (r == 0) {
                    r = 64;
                }
                return Decision{complex_from_rank(r), false, DataType::UNKNOWN};
            }
        }
    }

    // Fundamental kind rule: float dominates.
    if (is_real_float(a) || is_real_float(b)) {
        Decision d;
        if (is_real_float(a) && is_real_float(b)) {
            d.result_dtype = (mode == PrecisionMode::Highest) ? largest_float(a, b) : smallest_float(a, b);

            // Underpromotion is only meaningful for mixed-float ops.
            if (a != b) {
                const DataType max_dt = largest_float(a, b);
                d.float_underpromotion = (d.result_dtype != max_dt);
                if (d.float_underpromotion) {
                    d.chosen_float_dtype = d.result_dtype;
                }
            }
            return d;
        }

        // Only one side is a real float; float kind dominates but there is no rank choice.
        d.result_dtype = is_real_float(a) ? a : b;
        return d;
    }

    // Division on non-float numeric types is not generally closed in the integer ring.
    // NumPy-style behavior: promote to float64 when neither operand is float/complex.
    if (op == BinaryOp::Divide) {
        return Decision{(mode == PrecisionMode::Highest) ? DataType::FLOAT64 : DataType::FLOAT32, false, DataType::UNKNOWN};
    }

    // Fundamental kind rule: integer dominates bit for numeric ops.
    if (is_int_like(a) || is_int_like(b)) {
        // NOTE: bit×bit reduction/count ops are handled in the bit×bit section below
        // and intentionally produce int32 for practical count semantics.
        if (is_bit(a)) {
            return Decision{b, false, DataType::UNKNOWN};
        }
        if (is_bit(b)) {
            return Decision{a, false, DataType::UNKNOWN};
        }

        const bool a_signed = is_signed_int(a);
        const bool b_signed = is_signed_int(b);
        const uint8_t a_bits = int_rank_bits(a);
        const uint8_t b_bits = int_rank_bits(b);

        if (a_bits == 0 || b_bits == 0) {
            throw std::invalid_argument("resolve_promotion: invalid integer dtype");
        }

        // Same signedness: widen within that family.
        if (a_signed == b_signed) {
            const uint8_t bits = static_cast<uint8_t>(std::max(a_bits, b_bits));
            return Decision{a_signed ? signed_from_bits(bits) : unsigned_from_bits(bits), false, DataType::UNKNOWN};
        }

        // Mixed signed/unsigned: choose a signed type that can represent:
        // - all values of the unsigned operand (needs signed_bits - 1 >= unsigned_bits)
        // - and all values of the signed operand.
        const uint8_t signed_bits = a_signed ? a_bits : b_bits;
        const uint8_t unsigned_bits = a_signed ? b_bits : a_bits;
        const uint8_t required_signed_bits = static_cast<uint8_t>(std::max<uint8_t>(signed_bits, unsigned_bits + 1));

        const uint8_t rounded_signed_bits = round_up_int_bits(required_signed_bits);
        if (rounded_signed_bits == 0) {
            // There is no integer dtype that can represent both int64 and uint64 ranges.
            throw std::invalid_argument("resolve_promotion: unsupported signed/unsigned mix (no safe integer supertype)");
        }

        return Decision{signed_from_bits(rounded_signed_bits), false, DataType::UNKNOWN};
    }

    // Both are bit.
    if (is_bit(a) && is_bit(b)) {
        switch (op) {
            case BinaryOp::Add:
            case BinaryOp::Subtract:
            case BinaryOp::Matmul:
            case BinaryOp::MatrixVectorMultiply:
            case BinaryOp::VectorMatrixMultiply:
                return Decision{DataType::INT32, false, DataType::UNKNOWN};
            case BinaryOp::OuterProduct:
                // Outer product is elementwise products (no sum reduction), so BIT is sufficient.
                return Decision{DataType::BIT, false, DataType::UNKNOWN};
            case BinaryOp::ElementwiseMultiply:
                return Decision{DataType::BIT, false, DataType::UNKNOWN};
            case BinaryOp::Divide:
                // Handled above (policy chooses float32 or float64 when neither operand is float/complex).
                return Decision{DataType::FLOAT64, false, DataType::UNKNOWN};
        }
    }

    throw std::invalid_argument("resolve_promotion: unsupported dtype combination");
}

} // namespace pycauset::promotion
