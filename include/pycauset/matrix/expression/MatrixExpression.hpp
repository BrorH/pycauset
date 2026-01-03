/**
 * @file MatrixExpression.hpp
 * @brief Base interface for the Lazy Evaluation Expression Template engine.
 *
 * This file defines the MatrixExpression CRTP base class, which serves as the
 * common ancestor for all lazy nodes (BinaryExpression, UnaryExpression, etc.).
 * It provides the contract for element access, aliasing detection, and batch evaluation.
 *
 * The Expression Template system allows fusing operations like D = A + B + C into
 * a single pass without allocating temporary memory for (A + B).
 *
 * Dependencies:
 * - pycauset/matrix/MatrixBase.hpp (forward declaration)
 *
 * Thread Safety:
 * - Expressions are generally immutable and thread-safe for reading.
 * - Evaluation (writing to a destination) must be synchronized if the destination is shared.
 */

#pragma once

#include <cstdint>
#include <type_traits>
#include "pycauset/core/Types.hpp"

namespace pycauset {

class MatrixBase;

/**
 * @brief Curiously Recurring Template Pattern (CRTP) base for all matrix expressions.
 * 
 * @tparam Derived The concrete expression type (e.g., BinaryExpression<...>)
 */
template <typename Derived>
class MatrixExpression {
public:
    /**
     * @brief Get a single element at (i, j).
     * This is the core method used by the evaluation loop.
     */
    double get_element(uint64_t i, uint64_t j) const {
        return static_cast<const Derived*>(this)->get_element(i, j);
    }

    /**
     * @brief Check if this expression refers to the given matrix memory.
     * Used to detect aliasing (e.g., A = A + B) and trigger temporary buffering if needed.
     */
    bool aliases(const MatrixBase* target) const {
        return static_cast<const Derived*>(this)->aliases(target);
    }

    /**
     * @brief Notify the memory governor that operands are being used.
     * This prevents operands from being evicted during evaluation.
     */
    void touch_operands() const {
        static_cast<const Derived*>(this)->touch_operands();
    }

    /**
     * @brief Get the number of rows in the result.
     */
    uint64_t rows() const {
        return static_cast<const Derived*>(this)->rows();
    }

    /**
     * @brief Get the number of columns in the result.
     */
    uint64_t cols() const {
        return static_cast<const Derived*>(this)->cols();
    }

    /**
     * @brief Get the data type of the result.
     */
    DataType get_dtype() const {
        return static_cast<const Derived*>(this)->get_dtype();
    }

    /**
     * @brief Get the structural matrix type of the result.
     */
    MatrixType get_matrix_type() const {
        return static_cast<const Derived*>(this)->get_matrix_type();
    }

    /**
     * @brief Fill a buffer with a block of elements.
     * Uses C++20 requires clause to dispatch to optimized Derived::eval_block if available,
     * otherwise falls back to scalar get_element loop.
     */
    void fill_buffer(double* buffer, uint64_t start_row, uint64_t start_col, 
                    uint64_t num_rows, uint64_t num_cols, uint64_t stride) const {
        if constexpr (requires { static_cast<const Derived*>(this)->eval_block(buffer, start_row, start_col, num_rows, num_cols, stride); }) {
             static_cast<const Derived*>(this)->eval_block(buffer, start_row, start_col, num_rows, num_cols, stride);
        } else {
            for (uint64_t i = 0; i < num_rows; ++i) {
                for (uint64_t j = 0; j < num_cols; ++j) {
                    buffer[i * stride + j] = get_element(start_row + i, start_col + j);
                }
            }
        }
    }

    /**
     * @brief Cast to the derived type (helper).
     */
    const Derived& derived() const {
        return *static_cast<const Derived*>(this);
    }
};

} // namespace pycauset
