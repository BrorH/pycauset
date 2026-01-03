/**
 * @file MatrixOps.hpp
 * @brief Global operator overloads for Lazy Evaluation.
 *
 * Defines standard C++ operators (+, -, *, /) and math functions (sin, exp)
 * for MatrixBase objects. These return lightweight Expression objects
 * instead of computing the result immediately.
 */

#pragma once

#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/matrix/expression/MatrixRefExpression.hpp"
#include "pycauset/matrix/expression/ScalarExpression.hpp"
#include "pycauset/matrix/expression/BinaryExpression.hpp"
#include "pycauset/matrix/expression/UnaryExpression.hpp"
#include "pycauset/matrix/expression/Functors.hpp"

namespace pycauset {

// --- Binary Operators ---

// Matrix + Matrix
inline auto operator+(const MatrixBase& lhs, const MatrixBase& rhs) {
    return make_binary<MatrixRefExpression, MatrixRefExpression, ops::Add>(
        MatrixRefExpression(lhs), MatrixRefExpression(rhs));
}

// Matrix - Matrix
inline auto operator-(const MatrixBase& lhs, const MatrixBase& rhs) {
    return make_binary<MatrixRefExpression, MatrixRefExpression, ops::Sub>(
        MatrixRefExpression(lhs), MatrixRefExpression(rhs));
}

// Matrix * Scalar
inline auto operator*(const MatrixBase& lhs, double rhs) {
    return make_binary<MatrixRefExpression, ScalarExpression, ops::Mul>(
        MatrixRefExpression(lhs), ScalarExpression(rhs));
}

// Scalar * Matrix
inline auto operator*(double lhs, const MatrixBase& rhs) {
    return make_binary<ScalarExpression, MatrixRefExpression, ops::Mul>(
        ScalarExpression(lhs), MatrixRefExpression(rhs));
}

// Matrix / Scalar
inline auto operator/(const MatrixBase& lhs, double rhs) {
    return make_binary<MatrixRefExpression, ScalarExpression, ops::Div>(
        MatrixRefExpression(lhs), ScalarExpression(rhs));
}

// Matrix + Scalar
inline auto operator+(const MatrixBase& lhs, double rhs) {
    return make_binary<MatrixRefExpression, ScalarExpression, ops::Add>(
        MatrixRefExpression(lhs), ScalarExpression(rhs));
}

// Scalar + Matrix
inline auto operator+(double lhs, const MatrixBase& rhs) {
    return make_binary<ScalarExpression, MatrixRefExpression, ops::Add>(
        ScalarExpression(lhs), MatrixRefExpression(rhs));
}

// Matrix - Scalar
inline auto operator-(const MatrixBase& lhs, double rhs) {
    return make_binary<MatrixRefExpression, ScalarExpression, ops::Sub>(
        MatrixRefExpression(lhs), ScalarExpression(rhs));
}

// Scalar - Matrix
inline auto operator-(double lhs, const MatrixBase& rhs) {
    return make_binary<ScalarExpression, MatrixRefExpression, ops::Sub>(
        ScalarExpression(lhs), MatrixRefExpression(rhs));
}

// --- Opaque Operations (Eager Evaluation) ---

// Matrix * Matrix (MatMul)
// This cannot be fused into the expression template engine efficiently (O(N^3)).
// It triggers immediate evaluation via the AutoSolver.
std::unique_ptr<MatrixBase> operator*(const MatrixBase& lhs, const MatrixBase& rhs);

// --- Unary Operators ---

// -Matrix
inline auto operator-(const MatrixBase& matrix) {
    return make_unary<MatrixRefExpression, ops::Neg>(MatrixRefExpression(matrix));
}

// --- Math Functions ---

inline auto sin(const MatrixBase& matrix) {
    return make_unary<MatrixRefExpression, ops::Sin>(MatrixRefExpression(matrix));
}

inline auto cos(const MatrixBase& matrix) {
    return make_unary<MatrixRefExpression, ops::Cos>(MatrixRefExpression(matrix));
}

inline auto exp(const MatrixBase& matrix) {
    return make_unary<MatrixRefExpression, ops::Exp>(MatrixRefExpression(matrix));
}

// --- Expression + Expression Support ---
// To allow chaining like (A + B) + C, we need operators for MatrixExpression too.

template <typename L>
inline auto operator+(const MatrixExpression<L>& lhs, const MatrixBase& rhs) {
    return make_binary<L, MatrixRefExpression, ops::Add>(lhs.derived(), MatrixRefExpression(rhs));
}

template <typename R>
inline auto operator+(const MatrixBase& lhs, const MatrixExpression<R>& rhs) {
    return make_binary<MatrixRefExpression, R, ops::Add>(MatrixRefExpression(lhs), rhs.derived());
}

template <typename L, typename R>
inline auto operator+(const MatrixExpression<L>& lhs, const MatrixExpression<R>& rhs) {
    return make_binary<L, R, ops::Add>(lhs.derived(), rhs.derived());
}

template <typename L>
inline auto operator*(const MatrixExpression<L>& lhs, double rhs) {
    return make_binary<L, ScalarExpression, ops::Mul>(lhs.derived(), ScalarExpression(rhs));
}

template <typename R>
inline auto operator*(double lhs, const MatrixExpression<R>& rhs) {
    return make_binary<ScalarExpression, R, ops::Mul>(ScalarExpression(lhs), rhs.derived());
}

template <typename L>
inline auto operator+(const MatrixExpression<L>& lhs, double rhs) {
    return make_binary<L, ScalarExpression, ops::Add>(lhs.derived(), ScalarExpression(rhs));
}

template <typename R>
inline auto operator+(double lhs, const MatrixExpression<R>& rhs) {
    return make_binary<ScalarExpression, R, ops::Add>(ScalarExpression(lhs), rhs.derived());
}

template <typename L>
inline auto operator-(const MatrixExpression<L>& lhs, double rhs) {
    return make_binary<L, ScalarExpression, ops::Sub>(lhs.derived(), ScalarExpression(rhs));
}

template <typename R>
inline auto operator-(double lhs, const MatrixExpression<R>& rhs) {
    return make_binary<ScalarExpression, R, ops::Sub>(ScalarExpression(lhs), rhs.derived());
}

// --- Subtraction Support ---

template <typename L>
inline auto operator-(const MatrixExpression<L>& lhs, const MatrixBase& rhs) {
    return make_binary<L, MatrixRefExpression, ops::Sub>(lhs.derived(), MatrixRefExpression(rhs));
}

template <typename R>
inline auto operator-(const MatrixBase& lhs, const MatrixExpression<R>& rhs) {
    return make_binary<MatrixRefExpression, R, ops::Sub>(MatrixRefExpression(lhs), rhs.derived());
}

template <typename L, typename R>
inline auto operator-(const MatrixExpression<L>& lhs, const MatrixExpression<R>& rhs) {
    return make_binary<L, R, ops::Sub>(lhs.derived(), rhs.derived());
}

// --- Unary Negation for Expressions ---
template <typename E>
inline auto operator-(const MatrixExpression<E>& expr) {
    return make_unary<E, ops::Neg>(expr.derived());
}

// --- In-Place Operators (Template Implementations) ---

template <typename E>
inline MatrixBase& MatrixBase::operator+=(const MatrixExpression<E>& expr) {
    return *this = *this + expr;
}

template <typename E>
inline MatrixBase& MatrixBase::operator-=(const MatrixExpression<E>& expr) {
    return *this = *this - expr;
}

} // namespace pycauset
