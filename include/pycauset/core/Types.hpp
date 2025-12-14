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
    COMPLEX_FLOAT64 = 4,
    FLOAT32 = 5
};

}
