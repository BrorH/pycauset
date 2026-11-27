#include "pycauset/math/VectorOps.hpp"
#include "bindings_common.hpp"
#include "pycauset/math/CausalMatrix.hpp"
#include "pycauset/math/LinearAlgebra.hpp"
#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/vector/VectorBase.hpp"

namespace py = pybind11;

void bind_math_ops(py::module_& m) {
    // Link matrix of a transitively closed causal matrix (bit kernel, math/CausalMatrix.cpp).
    m.def("_transitive_reduction", [](const std::shared_ptr<pycauset::MatrixBase>& c) {
        const auto* tri = dynamic_cast<const pycauset::TriangularMatrix<bool>*>(c.get());
        if (tri == nullptr || tri->has_diagonal() || c->is_transposed() || c->is_subview() ||
            c->get_scalar() != std::complex<double>(1.0, 0.0)) {
            throw py::type_error("_transitive_reduction expects a plain causal matrix (TriangularMatrix of dtype 'bool')");
        }
        return std::shared_ptr<pycauset::MatrixBase>(pycauset::transitive_reduction(*tri).release());
    }, py::arg("C"), "Internal: link matrix of a transitively closed causal matrix (bit kernel).");

    // Bind generic math functions exposed as module-level functions
    // (e.g. _native.norm(x))

    m.def("norm", [](const pycauset::MatrixBase& m) {
        return pycauset::norm(m);
    }, py::arg("x"), "Compute the Frobenius norm of a matrix");

    m.def("norm", [](const pycauset::VectorBase& v) {
        return pycauset::norm(v);
    }, py::arg("x"), "Compute the L2 norm of a vector");

    m.def("qr", [](const pycauset::MatrixBase& a, bool complete) {
        auto [q, r] = pycauset::qr(a, complete);
        return py::make_tuple(
            std::shared_ptr<pycauset::MatrixBase>(q.release()),
            std::shared_ptr<pycauset::MatrixBase>(r.release()));
    }, py::arg("a"), py::arg("complete") = false, "QR (NumPy reduced/complete shapes; any dtype, structure, view)");

    m.def("pinv", [](const pycauset::MatrixBase& a) {
        return std::shared_ptr<pycauset::MatrixBase>(pycauset::pinv(a).release());
    }, py::arg("a"), "Moore-Penrose pseudoinverse (NumPy's default cutoff)");

    m.def("_matrix_norm", &pycauset::matrix_norm, py::arg("a"), py::arg("ord"), "NumPy matrix norm of order ord");
    m.def("_vector_norm", &pycauset::vector_norm, py::arg("v"), py::arg("p"), "NumPy vector p-norm");
    m.def("matrix_rank", &pycauset::matrix_rank, py::arg("a"), py::arg("tol") = -1.0, "NumPy matrix_rank");

    m.def("outer", [](const pycauset::VectorBase& a, const pycauset::VectorBase& b) {
        return std::shared_ptr<pycauset::MatrixBase>(pycauset::outer_product(a, b).release());
    }, py::arg("a"), py::arg("b"), "Outer product a[i] * b[j] (NumPy's multiply dtype)");

    m.def("cross", [](const pycauset::VectorBase& a, const pycauset::VectorBase& b) {
        return std::shared_ptr<pycauset::VectorBase>(pycauset::cross_product(a, b).release());
    }, py::arg("a"), py::arg("b"), "Cross product of two 3-vectors (NumPy's multiply dtype)");

    m.def("lu", [](const pycauset::MatrixBase& a) {
        auto [p, l, u] = pycauset::lu(a);
        return py::make_tuple(
            std::shared_ptr<pycauset::MatrixBase>(p.release()),
            std::shared_ptr<pycauset::MatrixBase>(l.release()),
            std::shared_ptr<pycauset::MatrixBase>(u.release()));
    }, py::arg("a"), "Compute LU decomposition");

    m.def("svd", [](const pycauset::MatrixBase& a, bool full_matrices, bool compute_uv) {
        auto [u, s, vt] = pycauset::svd(a, full_matrices, compute_uv);
        return py::make_tuple(
            std::shared_ptr<pycauset::MatrixBase>(u.release()),
            std::shared_ptr<pycauset::VectorBase>(s.release()),
            std::shared_ptr<pycauset::MatrixBase>(vt.release()));
    }, py::arg("a"), py::arg("full_matrices") = false, py::arg("compute_uv") = true,
       "SVD (NumPy shapes; any dtype, structure, view); compute_uv=False: S only");

    m.def("solve", [](const pycauset::MatrixBase& a, const pycauset::MatrixBase& b) {
        auto out = pycauset::solve(a, b);
        return std::shared_ptr<pycauset::MatrixBase>(out.release());
    }, py::arg("a"), py::arg("b"), "Solve linear system AX=B");
    // b a vector: solved as an n x 1 column over the vector's storage, returned as a vector.
    m.def("solve", [](const pycauset::MatrixBase& a, const pycauset::VectorBase& b) {
        auto col = pycauset::as_row_matrix(b);
        col->set_transposed(true);
        return std::shared_ptr<pycauset::VectorBase>(
            pycauset::as_vector(pycauset::solve(a, *col), /*transposed=*/false).release());
    }, py::arg("a"), py::arg("b"), "Solve linear system Ax=b for a vector b");
    m.def("lstsq", [](const pycauset::MatrixBase& a, const pycauset::MatrixBase& b) {
        return std::shared_ptr<pycauset::MatrixBase>(pycauset::lstsq(a, b).release());
    }, py::arg("a"), py::arg("b"), "Least-squares solution of a x = b (NumPy's lstsq x, ?gelsd)");
    m.def("lstsq", [](const pycauset::MatrixBase& a, const pycauset::VectorBase& b) {
        auto col = pycauset::as_row_matrix(b);
        col->set_transposed(true);
        return std::shared_ptr<pycauset::VectorBase>(
            pycauset::as_vector(pycauset::lstsq(a, *col), /*transposed=*/false).release());
    }, py::arg("a"), py::arg("b"), "Least-squares solution of a x = b for a vector b");
}
