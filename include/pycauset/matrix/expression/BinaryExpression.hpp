/**
 * @file BinaryExpression.hpp
 * @brief Expression node combining two sub-expressions with a binary operator.
 *
 * Represents operations like (A + B), (A * 5), etc.
 * Performs lazy evaluation by calling the Op functor on the elements of lhs and rhs.
 */

#pragma once

#include "MatrixExpression.hpp"
#include "pycauset/core/PromotionResolver.hpp"
#include "pycauset/core/StructureResolver.hpp"
#include <algorithm>
#include <stdexcept>

namespace pycauset {

template <typename L, typename R, typename Op>
class BinaryExpression : public MatrixExpression<BinaryExpression<L, R, Op>> {
public:
    BinaryExpression(const L& lhs, const R& rhs) : lhs_(lhs), rhs_(rhs) {
        // Basic dimension check (if both are non-scalar)
        uint64_t l_rows = lhs.rows();
        uint64_t l_cols = lhs.cols();
        uint64_t r_rows = rhs.rows();
        uint64_t r_cols = rhs.cols();

        if (l_rows > 0 && r_rows > 0 && (l_rows != r_rows || l_cols != r_cols)) {
            throw std::runtime_error("Dimension mismatch in binary expression");
        }
    }

    double get_element(uint64_t i, uint64_t j) const {
        return Op::apply(lhs_.get_element(i, j), rhs_.get_element(i, j));
    }

    bool aliases(const MatrixBase* target) const {
        return lhs_.aliases(target) || rhs_.aliases(target);
    }

    void touch_operands() const {
        lhs_.touch_operands();
        rhs_.touch_operands();
    }

    uint64_t rows() const {
        return std::max(lhs_.rows(), rhs_.rows());
    }

    uint64_t cols() const {
        return std::max(lhs_.cols(), rhs_.cols());
    }

    DataType get_dtype() const {
        if constexpr (requires { Op::binary_op; }) {
             return pycauset::promotion::resolve(Op::binary_op, lhs_.get_dtype(), rhs_.get_dtype()).result_dtype;
        } else {
             return DataType::FLOAT64;
        }
    }

    MatrixType get_matrix_type() const {
        if constexpr (requires { Op::binary_op; }) {
             return pycauset::structure::resolve_structure(Op::binary_op, lhs_.get_matrix_type(), rhs_.get_matrix_type());
        } else {
             return MatrixType::DENSE_FLOAT;
        }
    }

private:
    L lhs_;
    R rhs_;
};

// Operator overloads helper
template <typename L, typename R, typename Op>
BinaryExpression<L, R, Op> make_binary(const MatrixExpression<L>& lhs, const MatrixExpression<R>& rhs) {
    return BinaryExpression<L, R, Op>(lhs.derived(), rhs.derived());
}

} // namespace pycauset
