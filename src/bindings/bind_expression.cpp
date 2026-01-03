/**
 * @file bind_expression.cpp
 * @brief Python bindings for the Lazy Evaluation Expression system.
 *
 * Exposes the MatrixExpressionWrapper as "LazyMatrix" to Python.
 * This allows Python to inspect and manipulate lazy expressions before evaluation.
 */

#include "bindings_common.hpp"
#include "pycauset/matrix/expression/MatrixExpressionWrapper.hpp"
#include "pycauset/matrix/MatrixOps.hpp"
#include "pycauset/core/ObjectFactory.hpp"

void bind_expression_classes(py::module& m) {
    py::class_<pycauset::MatrixExpressionWrapper, std::shared_ptr<pycauset::MatrixExpressionWrapper>>(m, "LazyMatrix", py::dynamic_attr())
        .def("rows", &pycauset::MatrixExpressionWrapper::rows)
        .def("cols", &pycauset::MatrixExpressionWrapper::cols)
        .def("get", &pycauset::MatrixExpressionWrapper::get_element,
             "Get a single element (slow, for debugging only)",
             py::arg("i"), py::arg("j"))
        .def("__getitem__", [](const pycauset::MatrixExpressionWrapper& self, std::pair<uint64_t, uint64_t> idx) {
            return self.get_element(idx.first, idx.second);
        })
        .def("eval", [](const pycauset::MatrixExpressionWrapper& self) {
            auto result = pycauset::ObjectFactory::create_matrix(
                self.rows(), self.cols(), self.get_dtype(), self.get_matrix_type());
            self.eval_into(*result);
            return std::shared_ptr<pycauset::MatrixBase>(std::move(result));
        })
        .def_property_readonly("dtype", &pycauset::MatrixExpressionWrapper::get_dtype)
        .def_property_readonly("matrix_type", &pycauset::MatrixExpressionWrapper::get_matrix_type)
        .def("__repr__", [](const pycauset::MatrixExpressionWrapper& self) {
            return "<LazyMatrix rows=" + std::to_string(self.rows()) + 
                   " cols=" + std::to_string(self.cols()) + ">";
        })
        // Lazy + Matrix
        .def("__add__", [](std::shared_ptr<pycauset::MatrixExpressionWrapper> self, const pycauset::MatrixBase& other) {
            pycauset::MatrixWrapperExpression lhs(self);
            std::shared_ptr<pycauset::MatrixExpressionWrapper> result = pycauset::wrap_expression(lhs + other);
            return result;
        }, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        // Matrix + Lazy (handled by __radd__ on Lazy if Matrix doesn't support it, but MatrixBase bindings might need update too)
        .def("__radd__", [](std::shared_ptr<pycauset::MatrixExpressionWrapper> self, const pycauset::MatrixBase& other) {
            pycauset::MatrixWrapperExpression rhs(self);
            std::shared_ptr<pycauset::MatrixExpressionWrapper> result = pycauset::wrap_expression(other + rhs);
            return result;
        }, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        // Lazy + Lazy
        .def("__add__", [](std::shared_ptr<pycauset::MatrixExpressionWrapper> self, std::shared_ptr<pycauset::MatrixExpressionWrapper> other) {
            pycauset::MatrixWrapperExpression lhs(self);
            pycauset::MatrixWrapperExpression rhs(other);
            std::shared_ptr<pycauset::MatrixExpressionWrapper> result = pycauset::wrap_expression(lhs + rhs);
            return result;
        }, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        // Lazy + Scalar
        .def("__add__", [](std::shared_ptr<pycauset::MatrixExpressionWrapper> self, double other) {
            pycauset::MatrixWrapperExpression lhs(self);
            std::shared_ptr<pycauset::MatrixExpressionWrapper> result = pycauset::wrap_expression(lhs + other);
            return result;
        }, py::keep_alive<0, 1>())
        .def("__radd__", [](std::shared_ptr<pycauset::MatrixExpressionWrapper> self, double other) {
            pycauset::MatrixWrapperExpression rhs(self);
            std::shared_ptr<pycauset::MatrixExpressionWrapper> result = pycauset::wrap_expression(other + rhs);
            return result;
        }, py::keep_alive<0, 1>())
        // Lazy - Matrix
        .def("__sub__", [](std::shared_ptr<pycauset::MatrixExpressionWrapper> self, const pycauset::MatrixBase& other) {
            pycauset::MatrixWrapperExpression lhs(self);
            std::shared_ptr<pycauset::MatrixExpressionWrapper> result = pycauset::wrap_expression(lhs - other);
            return result;
        }, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("__rsub__", [](std::shared_ptr<pycauset::MatrixExpressionWrapper> self, const pycauset::MatrixBase& other) {
            pycauset::MatrixWrapperExpression rhs(self);
            std::shared_ptr<pycauset::MatrixExpressionWrapper> result = pycauset::wrap_expression(other - rhs);
            return result;
        }, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        // Lazy - Lazy
        .def("__sub__", [](std::shared_ptr<pycauset::MatrixExpressionWrapper> self, std::shared_ptr<pycauset::MatrixExpressionWrapper> other) {
            pycauset::MatrixWrapperExpression lhs(self);
            pycauset::MatrixWrapperExpression rhs(other);
            std::shared_ptr<pycauset::MatrixExpressionWrapper> result = pycauset::wrap_expression(lhs - rhs);
            return result;
        }, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        // Lazy - Scalar
        .def("__sub__", [](std::shared_ptr<pycauset::MatrixExpressionWrapper> self, double other) {
            pycauset::MatrixWrapperExpression lhs(self);
            std::shared_ptr<pycauset::MatrixExpressionWrapper> result = pycauset::wrap_expression(lhs - other);
            return result;
        }, py::keep_alive<0, 1>())
        .def("__rsub__", [](std::shared_ptr<pycauset::MatrixExpressionWrapper> self, double other) {
            pycauset::MatrixWrapperExpression rhs(self);
            std::shared_ptr<pycauset::MatrixExpressionWrapper> result = pycauset::wrap_expression(other - rhs);
            return result;
        }, py::keep_alive<0, 1>())
        // Lazy * Scalar
        .def("__mul__", [](std::shared_ptr<pycauset::MatrixExpressionWrapper> self, double other) {
            pycauset::MatrixWrapperExpression lhs(self);
            std::shared_ptr<pycauset::MatrixExpressionWrapper> result = pycauset::wrap_expression(lhs * other);
            return result;
        }, py::keep_alive<0, 1>())
        .def("__rmul__", [](std::shared_ptr<pycauset::MatrixExpressionWrapper> self, double other) {
            pycauset::MatrixWrapperExpression rhs(self);
            std::shared_ptr<pycauset::MatrixExpressionWrapper> result = pycauset::wrap_expression(other * rhs);
            return result;
        }, py::keep_alive<0, 1>());

    // Factory functions for lazy operations
    m.def("lazy_add", [](const pycauset::MatrixBase& a, const pycauset::MatrixBase& b) {
        std::shared_ptr<pycauset::MatrixExpressionWrapper> result = pycauset::wrap_expression(a + b);
        return result;
    }, py::keep_alive<0, 1>(), py::keep_alive<0, 2>());
    
    m.def("lazy_sub", [](const pycauset::MatrixBase& a, const pycauset::MatrixBase& b) {
        std::shared_ptr<pycauset::MatrixExpressionWrapper> result = pycauset::wrap_expression(a - b);
        return result;
    }, py::keep_alive<0, 1>(), py::keep_alive<0, 2>());

    m.def("lazy_mul_scalar", [](const pycauset::MatrixBase& a, double b) {
        std::shared_ptr<pycauset::MatrixExpressionWrapper> result = pycauset::wrap_expression(a * b);
        return result;
    }, py::keep_alive<0, 1>());

    m.def("lazy_sin", [](const pycauset::MatrixBase& a) {
        std::shared_ptr<pycauset::MatrixExpressionWrapper> result = pycauset::wrap_expression(pycauset::sin(a));
        return result;
    }, py::keep_alive<0, 1>());

    m.def("lazy_cos", [](const pycauset::MatrixBase& a) {
        std::shared_ptr<pycauset::MatrixExpressionWrapper> result = pycauset::wrap_expression(pycauset::cos(a));
        return result;
    }, py::keep_alive<0, 1>());
    
    // TODO: Add more ops (exp, log, etc.)
}
