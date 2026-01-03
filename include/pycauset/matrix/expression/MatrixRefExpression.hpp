/**
 * @file MatrixRefExpression.hpp
 * @brief Leaf node representing a reference to an existing MatrixBase.
 *
 * This wraps a const reference to a MatrixBase. It is the bridge between
 * the runtime-polymorphic MatrixBase world and the compile-time template world.
 */

#pragma once

#include "MatrixExpression.hpp"
#include "pycauset/matrix/MatrixBase.hpp"
#include <iostream>

namespace pycauset {

class MatrixRefExpression : public MatrixExpression<MatrixRefExpression> {
public:
    explicit MatrixRefExpression(const MatrixBase& matrix) : matrix_(matrix) {}

    double get_element(uint64_t i, uint64_t j) const {
        return matrix_.get_element_as_double(i, j);
    }

    bool aliases(const MatrixBase* target) const {
        return &matrix_ == target;
    }

    void touch_operands() const {
        // Update LRU status in MemoryGovernor.
        // const_cast is safe here because touch() only updates metadata (LRU position),
        // not the matrix data itself.
        const_cast<MatrixBase&>(matrix_).touch();
    }

    uint64_t rows() const { return matrix_.rows(); }
    uint64_t cols() const { return matrix_.cols(); }

    DataType get_dtype() const { return matrix_.get_data_type(); }
    MatrixType get_matrix_type() const { return matrix_.get_matrix_type(); }

private:
    const MatrixBase& matrix_;
};

} // namespace pycauset
