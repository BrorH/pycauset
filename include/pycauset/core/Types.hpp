#pragma once

#include <cstdint>

namespace pycauset {

// How a matrix or vector stores its elements. Independent of the element type (DataType):
// a causal matrix is TRIANGULAR with DataType::BIT.
enum class Structure : uint32_t {
    UNKNOWN = 0,
    DENSE = 1,
    TRIANGULAR = 2,      // packed upper triangle, with or without the diagonal (has_diagonal); transposed views read as lower
    DIAGONAL = 3,
    IDENTITY = 4,
    SYMMETRIC = 5,       // packed upper triangle, A == A^T
    ANTISYMMETRIC = 6,   // packed strict upper triangle, A == -A^T
    VECTOR = 7,
    UNIT_VECTOR = 8,
    HERMITIAN = 9,       // packed upper triangle, A == A^H (complex dtypes)
    SKEW_HERMITIAN = 10  // packed upper triangle, A == -A^H (complex dtypes)
};

enum class DataType : uint32_t {
    UNKNOWN = 0,
    BIT = 1,
    INT32 = 2,
    FLOAT64 = 3,
    FLOAT16 = 4,
    FLOAT32 = 5,
    INT16 = 6,
    COMPLEX_FLOAT16 = 7,
    COMPLEX_FLOAT32 = 8,
    COMPLEX_FLOAT64 = 9,
    // (Files store dtype names, not these numbers.)
    INT8 = 10,
    INT64 = 11,
    UINT8 = 12,
    UINT16 = 13,
    UINT32 = 14,
    UINT64 = 15
};

// Kind predicates: the one definition (review 3.3 C-12; six files had their own copies).
constexpr bool is_complex_dtype(DataType d) {
    return d == DataType::COMPLEX_FLOAT16 || d == DataType::COMPLEX_FLOAT32 || d == DataType::COMPLEX_FLOAT64;
}
constexpr bool is_real_float_dtype(DataType d) {
    return d == DataType::FLOAT16 || d == DataType::FLOAT32 || d == DataType::FLOAT64;
}
constexpr bool is_signed_int_dtype(DataType d) {
    return d == DataType::INT8 || d == DataType::INT16 || d == DataType::INT32 || d == DataType::INT64;
}
constexpr bool is_unsigned_int_dtype(DataType d) {
    return d == DataType::UINT8 || d == DataType::UINT16 || d == DataType::UINT32 || d == DataType::UINT64;
}
constexpr bool is_int_dtype(DataType d) { return is_signed_int_dtype(d) || is_unsigned_int_dtype(d); }  // not bool
constexpr bool is_integer_or_bit(DataType d) { return d == DataType::BIT || is_int_dtype(d); }

}
