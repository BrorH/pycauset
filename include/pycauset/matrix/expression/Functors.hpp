/**
 * @file Functors.hpp
 * @brief Operation functors for the Expression Template engine.
 *
 * Defines lightweight structs with static `apply` methods for arithmetic
 * and mathematical operations. These are used as template arguments for
 * BinaryExpression and UnaryExpression.
 */

#pragma once

#include <cmath>
#include "pycauset/core/PromotionResolver.hpp"

namespace pycauset {
namespace ops {

// --- Binary Operations ---

struct Add {
    static double apply(double lhs, double rhs) { return lhs + rhs; }
    static constexpr auto binary_op = pycauset::promotion::BinaryOp::Add;
};

struct Sub {
    static double apply(double lhs, double rhs) { return lhs - rhs; }
    static constexpr auto binary_op = pycauset::promotion::BinaryOp::Subtract;
};

struct Mul {
    static double apply(double lhs, double rhs) { return lhs * rhs; }
    static constexpr auto binary_op = pycauset::promotion::BinaryOp::ElementwiseMultiply;
};

struct Div {
    static double apply(double lhs, double rhs) { return lhs / rhs; }
    static constexpr auto binary_op = pycauset::promotion::BinaryOp::Divide;
};

// --- Unary Operations ---

struct Neg {
    static double apply(double val) { return -val; }
    static constexpr bool maps_zero_to_zero = true;
};

struct Sin {
    static double apply(double val) { return std::sin(val); }
    static constexpr bool maps_zero_to_zero = true;
};

struct Cos {
    static double apply(double val) { return std::cos(val); }
    static constexpr bool maps_zero_to_zero = false;
};

struct Exp {
    static double apply(double val) { return std::exp(val); }
    static constexpr bool maps_zero_to_zero = false;
};

struct Log {
    static double apply(double val) { return std::log(val); }
};

struct Sqrt {
    static double apply(double val) { return std::sqrt(val); }
};

struct Abs {
    static double apply(double val) { return std::abs(val); }
};

} // namespace ops
} // namespace pycauset
