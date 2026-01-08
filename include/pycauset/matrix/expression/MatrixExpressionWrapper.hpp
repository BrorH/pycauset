/**
 * @file MatrixExpressionWrapper.hpp
 * @brief Type-erased wrapper for MatrixExpressions.
 *
 * Allows passing arbitrary template expressions to non-template code (like Python bindings).
 */

#pragma once

#include <cstdint>
#include <memory>
#include "MatrixExpression.hpp"

#include <iostream>
#include "pycauset/matrix/MatrixBase.hpp"

namespace pycauset {

class MatrixExpressionWrapper {
public:
    virtual ~MatrixExpressionWrapper() = default;

    virtual double get_element(uint64_t i, uint64_t j) const = 0;
    virtual uint64_t rows() const = 0;
    virtual uint64_t cols() const = 0;
    virtual DataType get_dtype() const = 0;
    virtual MatrixType get_matrix_type() const = 0;
    virtual void eval_into(MatrixBase& target) const = 0;
    virtual bool aliases(const MatrixBase* target) const = 0;
    virtual void touch_operands() const = 0;
    
    // We might need to know the "source" matrices for aliasing checks,
    // but type erasure makes that hard.
    // For now, we assume the wrapper is used for immediate evaluation or simple chaining.
};

template <typename E>
class MatrixExpressionHolder : public MatrixExpressionWrapper {
public:
    explicit MatrixExpressionHolder(E expr) : expr_(std::move(expr)) {}

    double get_element(uint64_t i, uint64_t j) const override {
        return expr_.get_element(i, j);
    }

    uint64_t rows() const override { return expr_.rows(); }
    uint64_t cols() const override { return expr_.cols(); }
    DataType get_dtype() const override { return expr_.get_dtype(); }
    MatrixType get_matrix_type() const override { return expr_.get_matrix_type(); }
    void eval_into(MatrixBase& target) const override {
        target = expr_;
    }
    bool aliases(const MatrixBase* target) const override {
        return expr_.aliases(target);
    }
    void touch_operands() const override {
        expr_.touch_operands();
    }

private:
    E expr_;
};

// Adapter to use a MatrixExpressionWrapper as a MatrixExpression
class MatrixWrapperExpression : public MatrixExpression<MatrixWrapperExpression> {
public:
    explicit MatrixWrapperExpression(std::shared_ptr<MatrixExpressionWrapper> wrapper) 
        : wrapper_(std::move(wrapper)) {}

    double get_element(uint64_t i, uint64_t j) const {
        return wrapper_->get_element(i, j);
    }

    bool aliases(const MatrixBase* target) const {
        return wrapper_->aliases(target);
    }

    void touch_operands() const {
        wrapper_->touch_operands();
    }


    uint64_t rows() const { return wrapper_->rows(); }
    uint64_t cols() const { return wrapper_->cols(); }
    DataType get_dtype() const { return wrapper_->get_dtype(); }
    MatrixType get_matrix_type() const { return wrapper_->get_matrix_type(); }

private:
    std::shared_ptr<MatrixExpressionWrapper> wrapper_;
};

// Helper to wrap an expression
template <typename E>
std::unique_ptr<MatrixExpressionWrapper> wrap_expression(E expr) {
    return std::make_unique<MatrixExpressionHolder<E>>(std::move(expr));
}

} // namespace pycauset
