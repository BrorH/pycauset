#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <fstream>
#include <filesystem>
#include "CausalMatrix.hpp"
#include "IntegerMatrix.hpp"
#include "KMatrix.hpp"
#include "TriangularFloatMatrix.hpp"
#include "StoragePaths.hpp"
#include "FileFormat.hpp"

namespace py = pybind11;


namespace {
bool coerce_bool_like(const py::handle& value) {
    if (py::isinstance<py::bool_>(value)) {
        return py::cast<bool>(value);
    }
    if (py::isinstance<py::int_>(value)) {
        long long v = py::cast<long long>(value);
        if (v == 0 || v == 1) {
            return v == 1;
        }
        throw py::type_error("Integer assignments must be 0 or 1.");
    }
    if (py::isinstance<py::float_>(value)) {
        double v = py::cast<double>(value);
        if (v == 0.0 || v == 1.0) {
            return v == 1.0;
        }
        throw py::type_error("Float assignments must be 0.0 or 1.0.");
    }
    throw py::type_error("CausalMatrix entries accept bool, 0/1, or 0.0/1.0 values.");
}
} // namespace
PYBIND11_MODULE(pycauset, m) {
    m.doc() = "Causal++ Python Interface";


    auto matmul_wrapper = [](const CausalMatrix& self, const CausalMatrix& other, const std::string& saveas) {
        std::string target = saveas.empty() ? make_unique_storage_file("matmul") : saveas;
        return self.multiply(other, target);
    };
    auto elementwise_wrapper = [](const CausalMatrix& self, const CausalMatrix& other, const std::string& saveas) {
        std::string target = saveas.empty() ? make_unique_storage_file("elementwise") : saveas;
        return self.elementwise_multiply(other, target);
    };

    py::class_<TriangularFloatMatrix>(m, "TriangularFloatMatrix")
        .def(py::init<uint64_t, const std::string&>(), py::arg("n"), py::arg("backing_file") = "")
        .def("get", &TriangularFloatMatrix::get)
        .def("set", &TriangularFloatMatrix::set)
        .def("close", &TriangularFloatMatrix::close)
        .def("size", &TriangularFloatMatrix::size)
        .def_property_readonly("shape", [](const TriangularFloatMatrix& m) {
            return std::make_pair(m.size(), m.size());
        })
        .def("__getitem__", [](const TriangularFloatMatrix& m, std::pair<uint64_t, uint64_t> idx) {
            return m.get(idx.first, idx.second);
        })
        .def("__repr__", [](const TriangularFloatMatrix& m) {
            return "<TriangularFloatMatrix shape=(" + std::to_string(m.size()) + ", " + std::to_string(m.size()) + ")>";
        });

    py::class_<FloatMatrix>(m, "FloatMatrix")
        .def(py::init<uint64_t, const std::string&>(), py::arg("n"), py::arg("backing_file") = "")
        .def("get", &FloatMatrix::get)
        .def("set", &FloatMatrix::set)
        .def("close", &FloatMatrix::close)
        .def("size", &FloatMatrix::size)
        .def_property_readonly("shape", [](const FloatMatrix& m) {
            return std::make_pair(m.size(), m.size());
        })
        .def("__getitem__", [](const FloatMatrix& m, std::pair<uint64_t, uint64_t> idx) {
            return m.get(idx.first, idx.second);
        })
        .def("__repr__", [](const FloatMatrix& m) {
            return "<FloatMatrix shape=(" + std::to_string(m.size()) + ", " + std::to_string(m.size()) + ")>";
        });

    m.def("compute_k_matrix", &compute_k_matrix, 
          py::arg("C"), py::arg("a"), py::arg("output_path"), py::arg("num_threads") = 0,
          "Compute K = C(aI + C)^-1");

    py::class_<IntegerMatrix>(m, "IntegerMatrix")
        .def(py::init<uint64_t, const std::string&>(), py::arg("n"), py::arg("backing_file") = "")
        .def("get", &IntegerMatrix::get)
        .def("set", &IntegerMatrix::set)
        .def("close", &IntegerMatrix::close)
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
        .def(py::init<uint64_t, const std::string&, bool, std::optional<uint64_t>>(), 
             py::arg("n"), py::arg("saveas") = "", py::arg("populate") = true,
             py::arg("seed") = py::none())
        // Numpy Constructor
        .def(py::init([](py::array_t<bool> arr, std::string backing_file) {
            auto buf = arr.unchecked<2>();
            if (buf.ndim() != 2 || buf.shape(0) != buf.shape(1)) {
                throw std::invalid_argument("Array must be a square 2D matrix");
            }
            
            uint64_t n = buf.shape(0);
            auto mat = std::make_unique<CausalMatrix>(n, backing_file);
            
            for (py::ssize_t i = 0; i < n; ++i) {
                for (py::ssize_t j = i + 1; j < n; ++j) {
                    if (buf(i, j)) {
                        mat->set(i, j, true); 
                    }
                }
            }
            return mat;
        }), py::arg("array"), py::arg("backing_file") = "")
        
        // Random Factory
        .def_static("random", &CausalMatrix::random, 
            py::arg("n"), py::arg("density") = 0.5, py::arg("backing_file") = "",
            py::arg("seed") = py::none())

        .def("set", [](CausalMatrix& m, uint64_t i, uint64_t j, py::object value) {
            bool boolVal = coerce_bool_like(value);
            m.set(i, j, boolVal);
        })
        .def("get", &CausalMatrix::get)
        .def("size", &CausalMatrix::size)
        .def_property_readonly("shape", [](const CausalMatrix& m) {
            return std::make_pair(m.size(), m.size());
        })
        .def("close", &CausalMatrix::close,
             "Release the memory-mapped backing file. The matrix becomes unusable afterward.")
        .def("__len__", &CausalMatrix::size)
        .def("__getitem__", [](const CausalMatrix& m, std::pair<uint64_t, uint64_t> idx) {
            return m.get(idx.first, idx.second) ? 1 : 0;
        })
        // Guardrailed __setitem__
        .def("__setitem__", [](CausalMatrix& m, std::pair<uint64_t, uint64_t> idx, py::object value) {
            if (idx.first >= idx.second) {
                if (PyErr_WarnEx(PyExc_UserWarning, "Ignoring diagonal or lower-triangular set operation. CausalMatrix is strictly upper triangular.", 1) == -1) {
                    throw py::error_already_set();
                }
                return;
            }
            bool boolVal = coerce_bool_like(value);
            m.set(idx.first, idx.second, boolVal);
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
        .def("__mul__", [elementwise_wrapper](const CausalMatrix& self, const CausalMatrix& other) {
            return elementwise_wrapper(self, other, "");
        })
        .def("multiply", &CausalMatrix::multiply, 
             py::arg("other"), py::arg("result_file") = "",
             "Multiply this matrix by another CausalMatrix. Returns an IntegerMatrix.")
        .def("elementwise_multiply", [elementwise_wrapper](const CausalMatrix& self, const CausalMatrix& other, const std::string& saveas) {
            return elementwise_wrapper(self, other, saveas);
        }, py::arg("other"), py::arg("saveas") = "")
        .def("__repr__", [](const CausalMatrix& m) {
            return "<CausalMatrix shape=(" + std::to_string(m.size()) + ", " + std::to_string(m.size()) + ")>";
        });

    m.def("matmul", [matmul_wrapper](const CausalMatrix& a, const CausalMatrix& b, const std::string& saveas) {
        return matmul_wrapper(a, b, saveas);
    }, py::arg("a"), py::arg("b"), py::arg("saveas") = "",
    "Matrix multiplication producing an IntegerMatrix, mirroring numpy.matmul semantics for square matrices.");

    m.def("load", [](const std::string& path) -> py::object {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + path);
        }
        
        pycauset::FileHeader header;
        if (!file.read(reinterpret_cast<char*>(&header), sizeof(header))) {
             throw std::runtime_error("Failed to read file header: " + path);
        }
        file.close();

        if (std::strncmp(header.magic, "PYCAUSET", 8) != 0) {
             throw std::runtime_error("Invalid file format (Magic mismatch)");
        }
        
        uint64_t n = header.rows;
        uint64_t file_size = std::filesystem::file_size(path);
        if (file_size < sizeof(pycauset::FileHeader)) {
             throw std::runtime_error("File too small");
        }
        uint64_t data_size = file_size - sizeof(pycauset::FileHeader);
        
        auto mapper = std::make_unique<MemoryMapper>(path, data_size, false);
        
        switch (header.matrix_type) {
            case pycauset::MatrixType::CAUSAL:
                return py::cast(new CausalMatrix(n, std::move(mapper)));
            case pycauset::MatrixType::INTEGER:
                return py::cast(new IntegerMatrix(n, std::move(mapper)));
            case pycauset::MatrixType::TRIANGULAR_FLOAT:
                 return py::cast(new TriangularFloatMatrix(n, std::move(mapper)));
            case pycauset::MatrixType::DENSE_FLOAT:
                 return py::cast(new FloatMatrix(n, std::move(mapper)));
            default:
                throw std::runtime_error("Unsupported matrix type for loading");
        }
    }, "Load a matrix from a file");
}
