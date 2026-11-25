#pragma once

#include <cstdint>

namespace pycauset {

// How a matrix or vector stores its elements. Independent of the element type (DataType):
// a causal matrix is TRIANGULAR with DataType::BIT.
enum class Structure : uint32_t {
    UNKNOWN = 0,
    DENSE = 1,
    TRIANGULAR = 2,      // strictly upper triangular storage (transposed views read as lower)
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
    // Phase 2: additional integer widths + unsigned
    // NOTE: numeric values are part of on-disk metadata; do not renumber existing entries.
    INT8 = 10,
    INT64 = 11,
    UINT8 = 12,
    UINT16 = 13,
    UINT32 = 14,
    UINT64 = 15
};

}
