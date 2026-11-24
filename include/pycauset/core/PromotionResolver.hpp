#pragma once

#include "pycauset/core/Types.hpp"

#include <cstdint>

namespace pycauset::promotion {

enum class BinaryOp : uint8_t {
    Add,
    Subtract,
    ElementwiseMultiply,
    Divide,
    Matmul,
    MatrixVectorMultiply,
    VectorMatrixMultiply,
    OuterProduct,
};

struct Decision {
    DataType result_dtype{DataType::UNKNOWN};

    // True when a mixed-float operation chooses the smaller float dtype.
    bool float_underpromotion{false};

    // If float_underpromotion is true, these describe the chosen target dtype.
    DataType chosen_float_dtype{DataType::UNKNOWN};
};

Decision resolve(BinaryOp op, DataType a, DataType b);

} // namespace pycauset::promotion
