#pragma once

#include <cstdint>

namespace pycauset {

enum class MatrixType : uint32_t {
    UNKNOWN = 0,
    CAUSAL = 1,
    INTEGER = 2,
    TRIANGULAR_FLOAT = 3,
    DENSE_FLOAT = 4,
    IDENTITY = 5,
    VECTOR = 6,
    DIAGONAL = 7,
    UNIT_VECTOR = 8,
    SYMMETRIC = 9,
    ANTISYMMETRIC = 10
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
