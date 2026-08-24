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
        auto out = pycauset::cholesky(a);
        return std::shared_ptr<pycauset::MatrixBase>(out.release());
    }, py::arg("a"), "Compute Cholesky decomposition");

    m.def("qr", [](const pycauset::MatrixBase& a) {
        auto [q, r] = pycauset::qr(a);
        return py::make_tuple(
            std::shared_ptr<pycauset::MatrixBase>(q.release()),
            std::shared_ptr<pycauset::MatrixBase>(r.release()));
    }, py::arg("a"), "Compute QR decomposition");

    m.def("lu", [](const pycauset::MatrixBase& a) {
        auto [p, l, u] = pycauset::lu(a);
        return py::make_tuple(
            std::shared_ptr<pycauset::MatrixBase>(p.release()),
            std::shared_ptr<pycauset::MatrixBase>(l.release()),
            std::shared_ptr<pycauset::MatrixBase>(u.release()));
    }, py::arg("a"), "Compute LU decomposition");

    m.def("svd", [](const pycauset::MatrixBase& a) {
        auto [u, s, vt] = pycauset::svd(a);
        return py::make_tuple(
            std::shared_ptr<pycauset::MatrixBase>(u.release()),
            std::shared_ptr<pycauset::VectorBase>(s.release()),
            std::shared_ptr<pycauset::MatrixBase>(vt.release()));
    }, py::arg("a"), "Compute SVD decomposition");

    m.def("solve", [](const pycauset::MatrixBase& a, const pycauset::MatrixBase& b) {
        auto out = pycauset::solve(a, b);
        return std::shared_ptr<pycauset::MatrixBase>(out.release());
    }, py::arg("a"), py::arg("b"), "Solve linear system AX=B");
}
