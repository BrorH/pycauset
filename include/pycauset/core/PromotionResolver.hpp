#pragma once

#include "pycauset/core/Types.hpp"

#include <cstdint>

namespace pycauset::promotion {

enum class PrecisionMode : uint8_t {
    Lowest,
    Highest,
};

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

PrecisionMode get_precision_mode();
void set_precision_mode(PrecisionMode mode);

class ScopedPrecisionMode {
public:
    explicit ScopedPrecisionMode(PrecisionMode mode) : previous_(get_precision_mode()) {
        set_precision_mode(mode);
    }

    ScopedPrecisionMode(const ScopedPrecisionMode&) = delete;
    ScopedPrecisionMode& operator=(const ScopedPrecisionMode&) = delete;

    ~ScopedPrecisionMode() {
        set_precision_mode(previous_);
    }

private:
    PrecisionMode previous_;
};

Decision resolve(BinaryOp op, DataType a, DataType b);

} // namespace pycauset::promotion
