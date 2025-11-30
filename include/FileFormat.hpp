#pragma once
#include <cstdint>

namespace pycauset {

enum class MatrixType : uint32_t {
    UNKNOWN = 0,
    CAUSAL = 1,
    INTEGER = 2,
    TRIANGULAR_FLOAT = 3,
    DENSE_FLOAT = 4
};

enum class DataType : uint32_t {
    UNKNOWN = 0,
    BIT = 1,
    INT32 = 2,
    FLOAT64 = 3
};

struct FileHeader {
    char magic[8];          // "PYCAUSET"
    uint32_t version;       // 2
    MatrixType matrix_type;
    DataType data_type;
    uint64_t rows;
    uint64_t cols;
    uint64_t seed;          // 0 if not applicable
    double scalar;          // Scaling factor (default 1.0)
    uint8_t is_temporary;   // 1 if temporary, 0 if permanent
    uint8_t reserved[4039]; // Padding to 4096 bytes
};

static_assert(sizeof(FileHeader) == 4096, "FileHeader must be 4096 bytes");

}
