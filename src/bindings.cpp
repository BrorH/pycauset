#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "CausalMatrix.hpp"
#include "IntegerMatrix.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pycauset, m) {
    m.doc() = "Causal++ Python Interface";

    py::class_<IntegerMatrix>(m, "IntegerMatrix")
        .def("get", &IntegerMatrix::get)
        .def("size", &IntegerMatrix::size)
        .def_property_readonly("shape", [](const IntegerMatrix& m) {
            return std::make_pair(m.size(), m.size());
        })
        .def("__getitem__", [](const IntegerMatrix& m, std::pair<uint64_t, uint64_t> idx) {
            return m.get(idx.first, idx.second);
        })
        .def("__repr__", [](const IntegerMatrix& m) {
            return "<IntegerMatrix shape=(" + std::to_string(m.size()) + ", " + std::to_string(m.size()) + ")>";
        });

    py::class_<CausalMatrix>(m, "CausalMatrix")
        .def(py::init<uint64_t, const std::string&>(), 
             py::arg("N"), py::arg("backing_file") = "")
        // Numpy Constructor
        .def(py::init([](py::array_t<bool> arr, std::string backing_file) {
            auto buf = arr.unchecked<2>();
            if (buf.ndim() != 2 || buf.shape(0) != buf.shape(1)) {
                throw std::invalid_argument("Array must be a square 2D matrix");
            }
            
            uint64_t N = buf.shape(0);
            auto mat = std::make_unique<CausalMatrix>(N, backing_file);
            
            for (py::ssize_t i = 0; i < N; ++i) {
                for (py::ssize_t j = i + 1; j < N; ++j) {
                    if (buf(i, j)) {
                        mat->set(i, j, true); 
                    }
                }
            }
            return mat;
        }), py::arg("array"), py::arg("backing_file") = "")
        
        // Random Factory
        .def_static("random", &CausalMatrix::random, 
            py::arg("N"), py::arg("density") = 0.5, py::arg("backing_file") = "")

        .def("set", &CausalMatrix::set)
        .def("get", &CausalMatrix::get)
        .def("size", &CausalMatrix::size)
        .def_property_readonly("shape", [](const CausalMatrix& m) {
            return std::make_pair(m.size(), m.size());
        })
        .def("__len__", &CausalMatrix::size)
        .def("__getitem__", [](const CausalMatrix& m, std::pair<uint64_t, uint64_t> idx) {
            return m.get(idx.first, idx.second);
        })
        // Guardrailed __setitem__
        .def("__setitem__", [](CausalMatrix& m, std::pair<uint64_t, uint64_t> idx, bool val) {
            if (idx.first >= idx.second) {
                if (PyErr_WarnEx(PyExc_UserWarning, "Ignoring diagonal or lower-triangular set operation. CausalMatrix is strictly upper triangular.", 1) == -1) {
                    throw py::error_already_set();
                }
                return;
            }
            m.set(idx.first, idx.second, val);
        })
        // Advanced Indexing (Batch Set)
        .def("__setitem__", [](CausalMatrix& m, std::pair<py::array_t<int64_t>, py::array_t<int64_t>> idx, py::array_t<bool> vals) {
            auto rows = idx.first.unchecked<1>();
            auto cols = idx.second.unchecked<1>();
            auto values = vals.unchecked<1>();
            
            if (rows.size() != cols.size() || rows.size() != values.size()) {
                throw std::invalid_argument("Index and value arrays must have the same size");
            }

            bool warned = false;
            for (py::ssize_t k = 0; k < rows.size(); ++k) {
                uint64_t r = rows(k);
                uint64_t c = cols(k);
                if (r >= c) {
                    if (!warned) {
                        if (PyErr_WarnEx(PyExc_UserWarning, "Ignoring diagonal or lower-triangular set operation in batch.", 1) == -1) {
                            throw py::error_already_set();
                        }
                        warned = true;
                    }
                    continue; 
                }
                m.set(r, c, values(k));
            }
        })
        .def("__matmul__", [](const CausalMatrix& self, const CausalMatrix& other) {
            // Default to a temp file if none provided (handled by C++ logic if empty string passed?)
            // Actually C++ multiply requires a filename. Let's generate a temp one or use empty string if supported.
            // The C++ implementation currently requires a filename.
            // Let's use a default name pattern or require user to manage it?
            // For operator overloading, we can't easily ask for a filename.
            // We'll use a temporary name.
            static int counter = 0;
            std::string tempName = "temp_matmul_" + std::to_string(counter++) + ".bin";
            return self.multiply(other, tempName);
        })
        .def("multiply", &CausalMatrix::multiply, 
             py::arg("other"), py::arg("result_file") = "",
             "Multiply this matrix by another CausalMatrix. Returns an IntegerMatrix.")
        .def("__repr__", [](const CausalMatrix& m) {
            return "<CausalMatrix shape=(" + std::to_string(m.size()) + ", " + std::to_string(m.size()) + ")>";
        });
}
