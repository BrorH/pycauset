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
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/core/ObjectFactory.hpp"

namespace {
std::string dtype_to_string(pycauset::DataType dt) {
    switch (dt) {
        case pycauset::DataType::FLOAT16:
            return "float16";
        case pycauset::DataType::FLOAT32:
            return "float32";
        case pycauset::DataType::FLOAT64:
            return "float64";
        case pycauset::DataType::COMPLEX_FLOAT16:
            return "complex_float16";
        case pycauset::DataType::COMPLEX_FLOAT32:
            return "complex_float32";
        case pycauset::DataType::COMPLEX_FLOAT64:
            return "complex_float64";
        case pycauset::DataType::INT8:
            return "int8";
        case pycauset::DataType::INT16:
            return "int16";
        case pycauset::DataType::INT32:
            return "int32";
        case pycauset::DataType::INT64:
            return "int64";
        case pycauset::DataType::UINT8:
            return "uint8";
        case pycauset::DataType::UINT16:
            return "uint16";
        case pycauset::DataType::UINT32:
            return "uint32";
        case pycauset::DataType::UINT64:
            return "uint64";
        case pycauset::DataType::BIT:
            return "bit";
        default:
            return "unknown";
    }
}
} // namespace

void bind_expression_classes(py::module& m) {
    py::class_<pycauset::MatrixExpressionWrapper, std::shared_ptr<pycauset::MatrixExpressionWrapper>>(m, "LazyMatrix", py::dynamic_attr())
        .def("rows", &pycauset::MatrixExpressionWrapper::rows)
        .def("cols", &pycauset::MatrixExpressionWrapper::cols)
        .def_property_readonly("shape", [](const pycauset::MatrixExpressionWrapper& self) {
            return py::make_tuple(self.rows(), self.cols());
        })
        .def("size", [](const pycauset::MatrixExpressionWrapper& self) {
            return static_cast<uint64_t>(self.rows() * self.cols());
        })
        .def("close", [](const pycauset::MatrixExpressionWrapper&) { return; })
        .def("get", &pycauset::MatrixExpressionWrapper::get_element,
             "Get a single element (slow, for debugging only)",
             py::arg("i"), py::arg("j"))
        .def("get_element_as_double", [](const pycauset::MatrixExpressionWrapper& self, uint64_t i, uint64_t j) {
            return self.get_element(i, j);
        })
        .def("get_element_as_complex", [](const pycauset::MatrixExpressionWrapper& self, uint64_t i, uint64_t j) {
            return std::complex<double>(self.get_element(i, j), 0.0);
        })
        .def("__getitem__", [](const pycauset::MatrixExpressionWrapper& self, std::pair<uint64_t, uint64_t> idx) {
            return self.get_element(idx.first, idx.second);
        })
        .def("eval", [](const pycauset::MatrixExpressionWrapper& self) {
            auto result = pycauset::ObjectFactory::create_matrix(
                self.rows(), self.cols(), self.get_dtype(), self.get_matrix_type());
            self.eval_into(*result);
            return std::shared_ptr<pycauset::MatrixBase>(std::move(result));
        })
        // R1_NUMPY: Bind __array__ to trigger evaluation automatically (interop fix)
        .def("__array__", [](const pycauset::MatrixExpressionWrapper& self, py::args args, py::kwargs kwargs) {
             auto result_ptr = pycauset::ObjectFactory::create_matrix(
                self.rows(), self.cols(), self.get_dtype(), self.get_matrix_type());
             self.eval_into(*result_ptr);
             
             // Convert unique_ptr to shared_ptr to allow shared ownership with Python
             std::shared_ptr<pycauset::MatrixBase> shared_res = std::move(result_ptr);
             
             // The __array__ protocol requires returning a numpy.ndarray.
             py::object mat_obj = py::cast(shared_res);
             return py::array(mat_obj);
        })
        .def_property_readonly("dtype", [](const pycauset::MatrixExpressionWrapper& self) {
            return dtype_to_string(self.get_dtype());
        })
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
        }, py::keep_alive<0, 1>())
        // Lazy * Matrix (eager elementwise multiply)
        .def("__mul__", [](std::shared_ptr<pycauset::MatrixExpressionWrapper> self, const pycauset::MatrixBase& other) {
            auto lhs = pycauset::ObjectFactory::create_matrix(
                self->rows(), self->cols(), self->get_dtype(), self->get_matrix_type());
            self->eval_into(*lhs);
            auto out = pycauset::elementwise_multiply(*lhs, other, "");
            return std::shared_ptr<pycauset::MatrixBase>(out.release());
        }, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("__rmul__", [](std::shared_ptr<pycauset::MatrixExpressionWrapper> self, const pycauset::MatrixBase& other) {
            auto rhs = pycauset::ObjectFactory::create_matrix(
                self->rows(), self->cols(), self->get_dtype(), self->get_matrix_type());
            self->eval_into(*rhs);
            auto out = pycauset::elementwise_multiply(other, *rhs, "");
            return std::shared_ptr<pycauset::MatrixBase>(out.release());
        }, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        // Lazy / Matrix (eager elementwise divide)
        .def("__truediv__", [](std::shared_ptr<pycauset::MatrixExpressionWrapper> self, const pycauset::MatrixBase& other) {
            auto lhs = pycauset::ObjectFactory::create_matrix(
                self->rows(), self->cols(), self->get_dtype(), self->get_matrix_type());
            self->eval_into(*lhs);
            auto out = pycauset::elementwise_divide(*lhs, other, "");
            return std::shared_ptr<pycauset::MatrixBase>(out.release());
        }, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("__rtruediv__", [](std::shared_ptr<pycauset::MatrixExpressionWrapper> self, const pycauset::MatrixBase& other) {
            auto rhs = pycauset::ObjectFactory::create_matrix(
                self->rows(), self->cols(), self->get_dtype(), self->get_matrix_type());
            self->eval_into(*rhs);
            auto out = pycauset::elementwise_divide(other, *rhs, "");
            return std::shared_ptr<pycauset::MatrixBase>(out.release());
        }, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("__matmul__", [](std::shared_ptr<pycauset::MatrixExpressionWrapper> self, std::shared_ptr<pycauset::MatrixExpressionWrapper> other) {
            auto lhs = pycauset::ObjectFactory::create_matrix(
                self->rows(), self->cols(), self->get_dtype(), self->get_matrix_type());
            self->eval_into(*lhs);
            
            auto rhs = pycauset::ObjectFactory::create_matrix(
                other->rows(), other->cols(), other->get_dtype(), other->get_matrix_type());
            other->eval_into(*rhs);
            
            auto out = pycauset::dispatch_matmul(*lhs, *rhs, "");
            return std::shared_ptr<pycauset::MatrixBase>(out.release());
        })
        .def("__matmul__", [](std::shared_ptr<pycauset::MatrixExpressionWrapper> self, const pycauset::MatrixBase& other) {
            auto lhs = pycauset::ObjectFactory::create_matrix(
                self->rows(), self->cols(), self->get_dtype(), self->get_matrix_type());
            self->eval_into(*lhs);
            
            auto out = pycauset::dispatch_matmul(*lhs, other, "");
            return std::shared_ptr<pycauset::MatrixBase>(out.release());
        })
        .def("__rmatmul__", [](std::shared_ptr<pycauset::MatrixExpressionWrapper> self, const pycauset::MatrixBase& other) {
            auto rhs = pycauset::ObjectFactory::create_matrix(
                self->rows(), self->cols(), self->get_dtype(), self->get_matrix_type());
            self->eval_into(*rhs);
            
            auto out = pycauset::dispatch_matmul(other, *rhs, "");
            return std::shared_ptr<pycauset::MatrixBase>(out.release());
        });

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
