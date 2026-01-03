/**
 * @file ScalarExpression.hpp
 * @brief Leaf node representing a scalar value in an expression tree.
 *
 * This expression represents a constant value (e.g., 5.0 in "A + 5.0").
 * It effectively broadcasts the scalar to the dimensions of the other operand
 * during binary operations.
 */

#pragma once

#include "MatrixExpression.hpp"

namespace pycauset {

class ScalarExpression : public MatrixExpression<ScalarExpression> {
public:
    explicit ScalarExpression(double value) : value_(value) {}

    double get_element(uint64_t /*i*/, uint64_t /*j*/) const {
        return value_;
    }

    bool aliases(const MatrixBase* /*target*/) const {
        return false;
    }

    void touch_operands() const {
        // No memory to touch for a scalar
    }

    // Scalars don't have inherent dimensions, but in a binary op
    // they adopt the dimensions of the other operand.
    // If asked directly, they are effectively 1x1 or infinite.
    // For safety, we return 0 here and let the BinaryExpression handle sizing logic.
    uint64_t rows() const { return 0; }
    uint64_t cols() const { return 0; }

    DataType get_dtype() const { return DataType::FLOAT64; }
    MatrixType get_matrix_type() const { return MatrixType::SYMMETRIC; }

private:
    double value_;
};

} // namespace pycauset
