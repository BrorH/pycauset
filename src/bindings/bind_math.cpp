#include "bindings_common.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/vector/VectorBase.hpp"

namespace py = pybind11;

void bind_math_ops(py::module_& m) {
    // Bind generic math functions exposed as module-level functions
    // (e.g. _native.norm(x))

    m.def("norm", [](const pycauset::MatrixBase& m) {
        return pycauset::norm(m);
    }, py::arg("x"), "Compute the Frobenius norm of a matrix");

    m.def("norm", [](const pycauset::VectorBase& v) {
        return pycauset::norm(v);
    }, py::arg("x"), "Compute the L2 norm of a vector");

    m.def("cholesky", [](const pycauset::MatrixBase& a) {
        return pycauset::cholesky(a); 
    }, py::arg("a"), "Compute Cholesky decomposition");

    m.def("qr", [](const pycauset::MatrixBase& a) {
        return pycauset::qr(a);
    }, py::arg("a"), "Compute QR decomposition");

    m.def("lu", [](const pycauset::MatrixBase& a) {
        return pycauset::lu(a);
    }, py::arg("a"), "Compute LU decomposition");

    m.def("svd", [](const pycauset::MatrixBase& a) {
        return pycauset::svd(a); // returns tuple(U, S, VT)
    }, py::arg("a"), "Compute SVD decomposition");

    m.def("solve", [](const pycauset::MatrixBase& a, const pycauset::MatrixBase& b) {
        return pycauset::solve(a, b);
    }, py::arg("a"), py::arg("b"), "Solve linear system AX=B");
}
