#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <fstream>
#include <filesystem>
#include "TriangularBitMatrix.hpp"
#include "DenseMatrix.hpp"
#include "TriangularMatrix.hpp"
#include "StoragePaths.hpp"
#include "FileFormat.hpp"
#include "MatrixOperations.hpp"

namespace py = pybind11;

using FloatMatrix = DenseMatrix<double>;
using IntegerMatrix = DenseMatrix<int32_t>;
using TriangularFloatMatrix = TriangularMatrix<double>;
using TriangularIntegerMatrix = TriangularMatrix<int32_t>;
using TriangularBitMatrix = TriangularMatrix<bool>;

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
    throw py::type_error("TriangularBitMatrix entries accept bool, 0/1, or 0.0/1.0 values.");
}

template <typename T, typename... Options>
void bind_arithmetic(py::class_<T, Options...>& cls) {
    cls.def("__add__", [](const T& self, const MatrixBase& other) {
        return pycauset::add(self, other, make_unique_storage_file("add"));
    });
    cls.def("__sub__", [](const T& self, const MatrixBase& other) {
        return pycauset::subtract(self, other, make_unique_storage_file("sub"));
    });
    cls.def("__mul__", [](const T& self, const MatrixBase& other) {
        return pycauset::elementwise_multiply(self, other, make_unique_storage_file("mul"));
    });
    cls.def("__mul__", [](const T& self, double scalar) {
        return self.multiply_scalar(scalar, make_unique_storage_file("scalar_mul"));
    });
    cls.def("__rmul__", [](const T& self, double scalar) {
        return self.multiply_scalar(scalar, make_unique_storage_file("scalar_mul"));
    });
}

// Dispatcher
std::unique_ptr<MatrixBase> dispatch_matmul(const MatrixBase& a, const MatrixBase& b, std::string saveas) {
    if (saveas.empty()) saveas = make_unique_storage_file("matmul");

    // Try to cast to known types
    auto* a_fm = dynamic_cast<const FloatMatrix*>(&a);
    auto* a_im = dynamic_cast<const IntegerMatrix*>(&a);
    auto* a_tfm = dynamic_cast<const TriangularFloatMatrix*>(&a);
    auto* a_tim = dynamic_cast<const TriangularIntegerMatrix*>(&a);
    auto* a_tbm = dynamic_cast<const TriangularBitMatrix*>(&a);

    auto* b_fm = dynamic_cast<const FloatMatrix*>(&b);
    auto* b_im = dynamic_cast<const IntegerMatrix*>(&b);
    auto* b_tfm = dynamic_cast<const TriangularFloatMatrix*>(&b);
    auto* b_tim = dynamic_cast<const TriangularIntegerMatrix*>(&b);
    auto* b_tbm = dynamic_cast<const TriangularBitMatrix*>(&b);

    // If either is FloatMatrix (Dense<double>), result is FloatMatrix
    if (a_fm || b_fm) {
        // We need to implement mixed multiplication for all combinations.
        // For now, we can rely on DenseMatrix<double>::multiply which might handle templates if we implemented it.
        // But DenseMatrix::multiply expects DenseMatrix<T>.
        // We haven't implemented generic mixed multiply in DenseMatrix.hpp yet.
        // We only implemented multiply(const DenseMatrix<T>&).
        
        // Fallback: Convert both to FloatMatrix and multiply?
        // Or implement specific dispatch here.
        // Given the complexity, let's throw for mixed types not explicitly supported, 
        // or implement a slow fallback using get_element_as_double.
        
        // Implementing slow fallback for now to ensure correctness over speed for mixed types.
        // Ideally we would have optimized kernels.
        
        // Actually, let's just support FloatMatrix x FloatMatrix for now, and maybe others if easy.
        if (a_fm && b_fm) return a_fm->multiply(*b_fm, saveas);
        
        // TODO: Implement mixed kernels or conversion.
        throw std::runtime_error("Mixed Dense/Triangular multiplication not yet optimized. Convert to Dense first.");
    }

    // IntegerMatrix x IntegerMatrix
    if (a_im && b_im) return a_im->multiply(*b_im, saveas);

    // TriangularBitMatrix x TriangularBitMatrix -> TriangularIntegerMatrix
    if (a_tbm && b_tbm) return a_tbm->multiply(*b_tbm, saveas);

    // TriangularFloatMatrix x TriangularFloatMatrix -> TriangularFloatMatrix
    // We need to implement multiply for TriangularMatrix<T>.
    // Currently only TriangularMatrix<bool> has multiply implemented in .cpp.
    // TriangularMatrix<T> in .hpp doesn't have multiply implemented?
    // Wait, I didn't check TriangularMatrix.hpp for multiply implementation.
    // I should check.
    
    throw std::runtime_error("Unsupported matrix multiplication types.");
}

} // namespace

PYBIND11_MODULE(pycauset, m) {
    m.doc() = "pycauset Python Interface";

    py::class_<MatrixBase>(m, "MatrixBase")
        .def_property("scalar", &MatrixBase::get_scalar, &MatrixBase::set_scalar)
        .def_property_readonly("seed", &MatrixBase::get_seed)
        .def_property("is_temporary", &MatrixBase::is_temporary, &MatrixBase::set_temporary);

    // TriangularFloatMatrix
    py::class_<TriangularFloatMatrix, MatrixBase> tfm(m, "TriangularFloatMatrix");
    tfm.def(py::init<uint64_t, const std::string&>(), 
            py::arg("n"), py::arg("backing_file") = "")
        .def("get", &TriangularFloatMatrix::get)
        .def("set", &TriangularFloatMatrix::set)
        .def("close", &TriangularFloatMatrix::close)
        .def("size", &TriangularFloatMatrix::size)
        .def("get_backing_file", &TriangularFloatMatrix::get_backing_file)
        .def_property_readonly("shape", [](const TriangularFloatMatrix& m) {
            return std::make_pair(m.size(), m.size());
        })
        .def("__getitem__", [](const TriangularFloatMatrix& m, std::pair<uint64_t, uint64_t> idx) {
            return m.get(idx.first, idx.second) * m.get_scalar();
        })
        .def("__setitem__", [](TriangularFloatMatrix& m, std::pair<uint64_t, uint64_t> idx, double value) {
            m.set(idx.first, idx.second, value);
        })
        .def("__repr__", [](const TriangularFloatMatrix& m) {
            return "<TriangularFloatMatrix shape=(" + std::to_string(m.size()) + ", " + std::to_string(m.size()) + ")>";
        })
        .def("invert", [](const TriangularFloatMatrix& m) {
            return m.inverse(make_unique_storage_file("inverse"));
        }, "Matrix Inversion (Linear Algebra)")
        .def("__invert__", [](const TriangularFloatMatrix& m) {
            return m.bitwise_not(make_unique_storage_file("bitwise_not"));
        });
    bind_arithmetic(tfm);

    // TriangularIntegerMatrix
    py::class_<TriangularIntegerMatrix, MatrixBase> tim(m, "TriangularIntegerMatrix");
    tim.def(py::init<uint64_t, const std::string&>(), 
            py::arg("n"), py::arg("backing_file") = "")
        .def("get", &TriangularIntegerMatrix::get)
        .def("set", &TriangularIntegerMatrix::set)
        .def("close", &TriangularIntegerMatrix::close)
        .def("size", &TriangularIntegerMatrix::size)
        .def("get_backing_file", &TriangularIntegerMatrix::get_backing_file)
        .def_property_readonly("shape", [](const TriangularIntegerMatrix& m) {
            return std::make_pair(m.size(), m.size());
        })
        .def("__getitem__", [](const TriangularIntegerMatrix& m, std::pair<uint64_t, uint64_t> idx) {
            return m.get(idx.first, idx.second); // Integers usually don't use scalar? But MatrixBase has it.
        })
        .def("__setitem__", [](TriangularIntegerMatrix& m, std::pair<uint64_t, uint64_t> idx, int32_t value) {
            m.set(idx.first, idx.second, value);
        })
        .def("__repr__", [](const TriangularIntegerMatrix& m) {
            return "<TriangularIntegerMatrix shape=(" + std::to_string(m.size()) + ", " + std::to_string(m.size()) + ")>";
        })
        .def("invert", [](const TriangularIntegerMatrix& m) {
            return m.inverse(make_unique_storage_file("inverse"));
        }, "Matrix Inversion (Linear Algebra)")
        .def("__invert__", [](const TriangularIntegerMatrix& m) {
            return m.bitwise_not(make_unique_storage_file("bitwise_not"));
        });
    bind_arithmetic(tim);

    // FloatMatrix (Dense)
    py::class_<FloatMatrix, MatrixBase> fm(m, "FloatMatrix");
    fm.def(py::init<uint64_t, const std::string&>(), 
           py::arg("n"), py::arg("backing_file") = "")
        .def("get", &FloatMatrix::get)
        .def("set", &FloatMatrix::set)
        .def("close", &FloatMatrix::close)
        .def("size", &FloatMatrix::size)
        .def("get_backing_file", &FloatMatrix::get_backing_file)
        .def_property_readonly("shape", [](const FloatMatrix& m) {
            return std::make_pair(m.size(), m.size());
        })
        .def("__getitem__", [](const FloatMatrix& m, std::pair<uint64_t, uint64_t> idx) {
            return m.get(idx.first, idx.second) * m.get_scalar();
        })
        .def("__setitem__", [](FloatMatrix& m, std::pair<uint64_t, uint64_t> idx, double value) {
            m.set(idx.first, idx.second, value);
        })
        .def("__repr__", [](const FloatMatrix& m) {
            return "<FloatMatrix shape=(" + std::to_string(m.size()) + ", " + std::to_string(m.size()) + ")>";
        })
        .def("invert", [](const FloatMatrix& m) {
            return m.inverse(make_unique_storage_file("inverse"));
        }, "Matrix Inversion (Linear Algebra)")
        .def("__invert__", [](const FloatMatrix& m) {
            return m.bitwise_not(make_unique_storage_file("bitwise_not"));
        })
        .def("multiply", [](const FloatMatrix& self, const FloatMatrix& other, const std::string& saveas) {
            std::string target = saveas.empty() ? make_unique_storage_file("matmul_fm") : saveas;
            return self.multiply(other, target);
        }, py::arg("other"), py::arg("saveas") = "");
    bind_arithmetic(fm);

    m.def("compute_k_matrix", &compute_k_matrix, 
          py::arg("C"), py::arg("a"), py::arg("output_path"), py::arg("num_threads") = 0,
          "Compute K = C(aI + C)^-1");

    // IntegerMatrix (Dense)
    py::class_<IntegerMatrix, MatrixBase> im(m, "IntegerMatrix");
    im.def(py::init<uint64_t, const std::string&>(), 
           py::arg("n"), py::arg("backing_file") = "")
        .def("get", &IntegerMatrix::get)
        .def("set", &IntegerMatrix::set)
        .def("close", &IntegerMatrix::close)
        .def("size", &IntegerMatrix::size)
        .def("get_backing_file", &IntegerMatrix::get_backing_file)
        .def_property_readonly("shape", [](const IntegerMatrix& m) {
            return std::make_pair(m.size(), m.size());
        })
        .def("__getitem__", [](const IntegerMatrix& m, std::pair<uint64_t, uint64_t> idx) {
            return m.get(idx.first, idx.second);
        })
        .def("__setitem__", [](IntegerMatrix& m, std::pair<uint64_t, uint64_t> idx, int32_t value) {
            m.set(idx.first, idx.second, value);
        })
        .def("__repr__", [](const IntegerMatrix& m) {
            return "<IntegerMatrix shape=(" + std::to_string(m.size()) + ", " + std::to_string(m.size()) + ")>";
        })
        .def("invert", [](const IntegerMatrix& m) {
            return m.inverse(make_unique_storage_file("inverse"));
        }, "Matrix Inversion (Linear Algebra)")
        .def("__invert__", [](const IntegerMatrix& m) {
            return m.bitwise_not(make_unique_storage_file("bitwise_not"));
        })
        .def("multiply", [](const IntegerMatrix& self, const IntegerMatrix& other, const std::string& saveas) {
            std::string target = saveas.empty() ? make_unique_storage_file("matmul_im") : saveas;
            return self.multiply(other, target);
        }, py::arg("other"), py::arg("saveas") = "");
    bind_arithmetic(im);

    // TriangularBitMatrix
    py::class_<TriangularBitMatrix, MatrixBase> tbm(m, "TriangularBitMatrix");
    tbm.def(py::init<uint64_t, const std::string&>(), 
             py::arg("n"), py::arg("saveas") = "")
        // Numpy Constructor
        .def(py::init([](py::array_t<bool> arr, std::string backing_file) {
            auto buf = arr.unchecked<2>();
            if (buf.ndim() != 2 || buf.shape(0) != buf.shape(1)) {
                throw std::invalid_argument("Array must be a square 2D matrix");
            }
            
            uint64_t n = buf.shape(0);
            auto mat = std::make_unique<TriangularBitMatrix>(n, backing_file);
            
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
        .def_static("random", &TriangularBitMatrix::random, 
            py::arg("n"), py::arg("density") = 0.5, py::arg("backing_file") = "",
            py::arg("seed") = py::none())

        .def("set", [](TriangularBitMatrix& m, uint64_t i, uint64_t j, py::object value) {
            bool boolVal = coerce_bool_like(value);
            m.set(i, j, boolVal);
        })
        .def("get", &TriangularBitMatrix::get)
        .def("size", &TriangularBitMatrix::size)
        .def_property_readonly("shape", [](const TriangularBitMatrix& m) {
            return std::make_pair(m.size(), m.size());
        })
        .def("close", &TriangularBitMatrix::close,
             "Release the memory-mapped backing file. The matrix becomes unusable afterward.")
        .def("__len__", &TriangularBitMatrix::size)
        .def("__getitem__", [](const TriangularBitMatrix& m, std::pair<uint64_t, uint64_t> idx) {
            return m.get_element_as_double(idx.first, idx.second);
        })
        // Guardrailed __setitem__
        .def("__setitem__", [](TriangularBitMatrix& m, std::pair<uint64_t, uint64_t> idx, py::object value) {
            if (idx.first >= idx.second) {
                if (PyErr_WarnEx(PyExc_UserWarning, "Ignoring diagonal or lower-triangular set operation. TriangularBitMatrix is strictly upper triangular.", 1) == -1) {
                    throw py::error_already_set();
                }
                return;
            }
            bool boolVal = coerce_bool_like(value);
            m.set(idx.first, idx.second, boolVal);
        })
        // Advanced Indexing (Batch Set)
        .def("__setitem__", [](TriangularBitMatrix& m, std::pair<py::array_t<int64_t>, py::array_t<int64_t>> idx, py::array_t<bool> vals) {
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
        .def("multiply", &TriangularBitMatrix::multiply, 
             py::arg("other"), py::arg("result_file") = "",
             "Multiply this matrix by another TriangularBitMatrix. Returns a TriangularIntegerMatrix.")
        .def("elementwise_multiply", [](const TriangularBitMatrix& self, const TriangularBitMatrix& other, const std::string& saveas) {
            return self.elementwise_multiply(other, saveas);
        }, py::arg("other"), py::arg("saveas") = "")
        .def("invert", [](const TriangularBitMatrix& m) {
            return m.inverse(make_unique_storage_file("inverse"));
        }, "Matrix Inversion (Linear Algebra)")
        .def("__invert__", [](const TriangularBitMatrix& m) {
            return m.bitwise_not(make_unique_storage_file("bitwise_not"));
        })
        .def("get_backing_file", &TriangularBitMatrix::get_backing_file)
        .def("__repr__", [](const TriangularBitMatrix& m) {
            return "<TriangularBitMatrix shape=(" + std::to_string(m.size()) + ", " + std::to_string(m.size()) + ")>";
        });
    bind_arithmetic(tbm);

    m.def("matmul", &dispatch_matmul, py::arg("a"), py::arg("b"), py::arg("saveas") = "",
          "Generic matrix multiplication. Supports mixed types (Dense/Triangular).");

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
                return py::cast(new TriangularBitMatrix(n, std::move(mapper)));
            case pycauset::MatrixType::INTEGER:
                // Assuming INTEGER is now Dense Integer
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
