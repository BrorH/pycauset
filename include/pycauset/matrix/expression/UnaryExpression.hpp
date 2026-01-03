/**
 * @file UnaryExpression.hpp
 * @brief Expression node modifying a sub-expression with a unary operator.
 *
 * Represents operations like -A, sin(A), etc.
 */

#pragma once

#include "MatrixExpression.hpp"

namespace pycauset {

template <typename E, typename Op>
class UnaryExpression : public MatrixExpression<UnaryExpression<E, Op>> {
public:
    explicit UnaryExpression(const E& expr) : expr_(expr) {}

    double get_element(uint64_t i, uint64_t j) const {
        return Op::apply(expr_.get_element(i, j));
    }

    bool aliases(const MatrixBase* target) const {
        return expr_.aliases(target);
    }

    void touch_operands() const {
        expr_.touch_operands();
    }

    uint64_t rows() const { return expr_.rows(); }
    uint64_t cols() const { return expr_.cols(); }

    DataType get_dtype() const {
        return expr_.get_dtype();
    }

    MatrixType get_matrix_type() const {
        MatrixType t = expr_.get_matrix_type();
        
        // Symmetric is always preserved for elementwise unary ops
        if (t == MatrixType::SYMMETRIC) return MatrixType::SYMMETRIC;
        
        // Identity becomes Diagonal (values change) if zero-preserving, else Dense
        if (t == MatrixType::IDENTITY) {
             if constexpr (requires { Op::maps_zero_to_zero; }) {
                 if (Op::maps_zero_to_zero) return MatrixType::DIAGONAL;
             }
             return MatrixType::DENSE_FLOAT;
        }

        // Diagonal/Triangular preserved only if f(0)=0
        if constexpr (requires { Op::maps_zero_to_zero; }) {
            if (Op::maps_zero_to_zero) {
                return t;
            }
        }
        
        return MatrixType::DENSE_FLOAT;
    }

private:
    E expr_;
};

// Operator helper
template <typename E, typename Op>
UnaryExpression<E, Op> make_unary(const MatrixExpression<E>& expr) {
    return UnaryExpression<E, Op>(expr.derived());
}

} // namespace pycauset
