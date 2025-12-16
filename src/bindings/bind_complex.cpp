#include "bindings_common.hpp"

#include "pycauset/matrix/ComplexMatrix.hpp"
#include "pycauset/vector/ComplexVector.hpp"

#include <complex>
#include <memory>

void bind_complex_classes(py::module_& m) {
    py::class_<pycauset::ComplexVector, std::shared_ptr<pycauset::ComplexVector>>(m, "ComplexVector")
        .def(
            py::init([](uint64_t n) {
                return std::make_shared<pycauset::ComplexVector>(n, "", "");
            }),
            py::arg("n"))
        .def_static(
            "_from_storage",
            [](uint64_t n, const std::string& backing_file_real, const std::string& backing_file_imag) {
                return std::make_shared<pycauset::ComplexVector>(n, backing_file_real, backing_file_imag);
            },
            py::arg("n"),
            py::arg("backing_file_real"),
            py::arg("backing_file_imag"))
        .def("size", &pycauset::ComplexVector::size)
        .def("__len__", &pycauset::ComplexVector::size)
        .def("get", &pycauset::ComplexVector::get)
        .def("set", &pycauset::ComplexVector::set)
        .def("close", &pycauset::ComplexVector::close)
        .def("__getitem__", [](const pycauset::ComplexVector& v, uint64_t i) { return v.get(i); })
        .def("__setitem__", [](pycauset::ComplexVector& v, uint64_t i, std::complex<double> val) { v.set(i, val); })
        .def(
            "__array__",
            [](const pycauset::ComplexVector& v, py::object /*dtype*/, py::object /*copy*/) {
                uint64_t n_u = v.size();
                py::array_t<std::complex<double>> out(static_cast<py::ssize_t>(n_u));
                auto r = out.mutable_unchecked<1>();
                for (uint64_t i = 0; i < n_u; ++i) {
                    r(i) = v.get(i);
                }
                return out;
            },
            py::arg("dtype") = py::none(),
            py::arg("copy") = py::none())
        .def_property_readonly(
            "real",
            [](pycauset::ComplexVector& v) { return static_cast<pycauset::VectorBase*>(v.real()); },
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "imag",
            [](pycauset::ComplexVector& v) { return static_cast<pycauset::VectorBase*>(v.imag()); },
            py::return_value_policy::reference_internal);

    py::class_<pycauset::ComplexMatrix, std::shared_ptr<pycauset::ComplexMatrix>>(m, "ComplexMatrix")
        .def(
            py::init([](uint64_t n) {
                return std::make_shared<pycauset::ComplexMatrix>(n, "", "");
            }),
            py::arg("n"))
        .def_static(
            "_from_storage",
            [](uint64_t n,
               const std::string& backing_file_real,
               size_t offset_real,
               const std::string& backing_file_imag,
               size_t offset_imag) {
                return std::make_shared<pycauset::ComplexMatrix>(n, backing_file_real, offset_real, backing_file_imag, offset_imag);
            },
            py::arg("n"),
            py::arg("backing_file_real"),
            py::arg("offset_real"),
            py::arg("backing_file_imag"),
            py::arg("offset_imag"))
        .def("size", &pycauset::ComplexMatrix::size)
        .def("__len__", &pycauset::ComplexMatrix::size)
        .def("get", &pycauset::ComplexMatrix::get)
        .def("set", &pycauset::ComplexMatrix::set)
        .def("close", &pycauset::ComplexMatrix::close)
        .def_property_readonly(
            "real",
            [](pycauset::ComplexMatrix& mtx) { return static_cast<pycauset::MatrixBase*>(mtx.real()); },
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "imag",
            [](pycauset::ComplexMatrix& mtx) { return static_cast<pycauset::MatrixBase*>(mtx.imag()); },
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "T",
            [](const pycauset::ComplexMatrix& mtx) {
                auto out = mtx.transpose("", "");
                return std::shared_ptr<pycauset::ComplexMatrix>(out.release());
            })
        .def_property_readonly(
            "H",
            [](const pycauset::ComplexMatrix& mtx) {
                auto out = mtx.hermitian("", "");
                return std::shared_ptr<pycauset::ComplexMatrix>(out.release());
            })
        .def(
            "transpose",
            [](const pycauset::ComplexMatrix& mtx) {
                auto out = mtx.transpose("", "");
                return std::shared_ptr<pycauset::ComplexMatrix>(out.release());
            })
        .def(
            "hermitian",
            [](const pycauset::ComplexMatrix& mtx) {
                auto out = mtx.hermitian("", "");
                return std::shared_ptr<pycauset::ComplexMatrix>(out.release());
            })
        .def(
            "conjugate",
            [](const pycauset::ComplexMatrix& mtx) {
                auto out = mtx.conjugate("", "");
                return std::shared_ptr<pycauset::ComplexMatrix>(out.release());
            })
        .def(
            "__array__",
            [](const pycauset::ComplexMatrix& mtx, py::object /*dtype*/, py::object /*copy*/) {
                uint64_t n_u = mtx.size();
                py::ssize_t n = static_cast<py::ssize_t>(n_u);
                py::array_t<std::complex<double>> out({n, n});
                auto r = out.mutable_unchecked<2>();
                for (uint64_t i = 0; i < n_u; ++i) {
                    for (uint64_t j = 0; j < n_u; ++j) {
                        r(i, j) = mtx.get(i, j);
                    }
                }
                return out;
            },
            py::arg("dtype") = py::none(),
            py::arg("copy") = py::none())
        .def(
            "__add__",
            [](const pycauset::ComplexMatrix& a, const std::shared_ptr<pycauset::ComplexMatrix>& b) {
                auto out = pycauset::add(a, *b, "", "");
                return std::shared_ptr<pycauset::ComplexMatrix>(out.release());
            },
            py::is_operator())
        .def(
            "__add__",
            [](const pycauset::ComplexMatrix& a, std::complex<double> s) {
                auto out = pycauset::add_scalar(a, s, "", "");
                return std::shared_ptr<pycauset::ComplexMatrix>(out.release());
            },
            py::is_operator())
        .def(
            "__radd__",
            [](const pycauset::ComplexMatrix& a, std::complex<double> s) {
                auto out = pycauset::add_scalar(a, s, "", "");
                return std::shared_ptr<pycauset::ComplexMatrix>(out.release());
            },
            py::is_operator())
        .def(
            "__mul__",
            [](const pycauset::ComplexMatrix& a, const std::shared_ptr<pycauset::ComplexMatrix>& b) {
                auto out = pycauset::multiply(a, *b, "", "");
                return std::shared_ptr<pycauset::ComplexMatrix>(out.release());
            },
            py::is_operator())
        .def(
            "__mul__",
            [](const pycauset::ComplexMatrix& a, std::complex<double> s) {
                auto out = pycauset::multiply_scalar(a, s, "", "");
                return std::shared_ptr<pycauset::ComplexMatrix>(out.release());
            },
            py::is_operator())
        .def(
            "__rmul__",
            [](const pycauset::ComplexMatrix& a, std::complex<double> s) {
                auto out = pycauset::multiply_scalar(a, s, "", "");
                return std::shared_ptr<pycauset::ComplexMatrix>(out.release());
            },
            py::is_operator());
}
