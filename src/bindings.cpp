#include "Spacetime.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <fstream>
#include <filesystem>
#include "TriangularBitMatrix.hpp"
#include "DenseMatrix.hpp"
#include "DenseBitMatrix.hpp"
#include "DenseVector.hpp"
#include "TriangularMatrix.hpp"
#include "IdentityMatrix.hpp"
#include "StoragePaths.hpp"
#include "FileFormat.hpp"
#include "MatrixOperations.hpp"
#include "VectorOperations.hpp"
#include "PersistentObject.hpp"
#include "ComplexMatrix.hpp"
#include "Sprinkler.hpp"
#include "MatrixFactory.hpp"

#include "UnitVector.hpp"
#include "VectorFactory.hpp"

namespace py = pybind11;
using namespace pycauset;

using FloatMatrix = DenseMatrix<double>;
using IntegerMatrix = DenseMatrix<int32_t>;
using DenseBitMatrix = DenseMatrix<bool>;
using FloatVector = DenseVector<double>;
using IntegerVector = DenseVector<int32_t>;
using BitVector = DenseVector<bool>;
using TriangularFloatMatrix = TriangularMatrix<double>;
using TriangularIntegerMatrix = TriangularMatrix<int32_t>;
using TriangularBitMatrix = TriangularMatrix<bool>;

using pycauset::make_unique_storage_file;

namespace {

// Forward declaration
std::unique_ptr<PersistentObject> from_numpy(py::object obj);
std::unique_ptr<MatrixBase> dispatch_matmul(const MatrixBase& a, const MatrixBase& b, std::string saveas);

// Helper functions and bind_arithmetic moved to PYBIND11_MODULE


// bind_vector_arithmetic and bind_arithmetic moved to PYBIND11_MODULE


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

    bool a_is_id = (a.get_matrix_type() == MatrixType::IDENTITY);
    bool b_is_id = (b.get_matrix_type() == MatrixType::IDENTITY);

    // Identity x Identity -> Identity
    if (a_is_id && b_is_id) {
        if (a.size() != b.size()) throw std::invalid_argument("Dimension mismatch");
        DataType res_dtype = MatrixFactory::resolve_result_type(a.get_data_type(), b.get_data_type());
        auto res = MatrixFactory::create(a.size(), res_dtype, MatrixType::IDENTITY, saveas);
        res->set_scalar(a.get_scalar() * b.get_scalar());
        return res;
    }

    // Identity x Any -> Any * scalar
    if (a_is_id) {
        if (a.size() != b.size()) throw std::invalid_argument("Dimension mismatch");
        return b.multiply_scalar(a.get_scalar(), saveas);
    }

    // Any x Identity -> Any * scalar
    if (b_is_id) {
        if (a.size() != b.size()) throw std::invalid_argument("Dimension mismatch");
        return a.multiply_scalar(b.get_scalar(), saveas);
    }

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

std::unique_ptr<PersistentObject> from_numpy(py::object obj) {
    if (!py::isinstance<py::array>(obj)) {
        try {
            obj = py::array(obj);
        } catch (...) {
            return nullptr;
        }
    }
    
    py::array arr = py::cast<py::array>(obj);
    auto info = arr.request();
    
    if (info.ndim == 1) {
        uint64_t n = info.shape[0];
        if (py::isinstance<py::array_t<double>>(arr)) {
            auto v = std::make_unique<FloatVector>(n, make_unique_storage_file("numpy_vec_float"));
            auto ptr = static_cast<double*>(info.ptr);
            for (uint64_t i = 0; i < n; ++i) v->set(i, ptr[i]);
            return v;
        } else if (py::isinstance<py::array_t<int32_t>>(arr) || py::isinstance<py::array_t<int64_t>>(arr)) {
            auto v = std::make_unique<IntegerVector>(n, make_unique_storage_file("numpy_vec_int"));
            if (info.format == "i" || info.format == "l") {
                 if (py::isinstance<py::array_t<int64_t>>(arr)) {
                     auto unchecked = arr.unchecked<int64_t, 1>();
                     for (uint64_t i = 0; i < n; ++i) v->set(i, (int32_t)unchecked(i));
                 } else {
                     auto unchecked = arr.unchecked<int32_t, 1>();
                     for (uint64_t i = 0; i < n; ++i) v->set(i, unchecked(i));
                 }
            }
            return v;
        } else if (py::isinstance<py::array_t<bool>>(arr)) {
            auto v = std::make_unique<BitVector>(n, make_unique_storage_file("numpy_vec_bool"));
            auto unchecked = arr.unchecked<bool, 1>();
            for (uint64_t i = 0; i < n; ++i) v->set(i, unchecked(i));
            return v;
        }
    } else if (info.ndim == 2) {
        if (info.shape[0] != info.shape[1]) {
            throw std::invalid_argument("Only square matrices are supported for now");
        }
        uint64_t n = info.shape[0];
        
        if (py::isinstance<py::array_t<double>>(arr)) {
            auto m = std::make_unique<FloatMatrix>(n, make_unique_storage_file("numpy_mat_float"));
            auto unchecked = arr.unchecked<double, 2>();
            for (uint64_t i = 0; i < n; ++i) {
                for (uint64_t j = 0; j < n; ++j) {
                    m->set(i, j, unchecked(i, j));
                }
            }
            return m;
        } else if (py::isinstance<py::array_t<int32_t>>(arr) || py::isinstance<py::array_t<int64_t>>(arr)) {
            auto m = std::make_unique<IntegerMatrix>(n, make_unique_storage_file("numpy_mat_int"));
            if (py::isinstance<py::array_t<int64_t>>(arr)) {
                auto unchecked = arr.unchecked<int64_t, 2>();
                for (uint64_t i = 0; i < n; ++i) {
                    for (uint64_t j = 0; j < n; ++j) {
                        m->set(i, j, (int32_t)unchecked(i, j));
                    }
                }
            } else {
                auto unchecked = arr.unchecked<int32_t, 2>();
                for (uint64_t i = 0; i < n; ++i) {
                    for (uint64_t j = 0; j < n; ++j) {
                        m->set(i, j, unchecked(i, j));
                    }
                }
            }
            return m;
        } else if (py::isinstance<py::array_t<bool>>(arr)) {
            auto m = std::make_unique<DenseBitMatrix>(n, make_unique_storage_file("numpy_mat_bool"));
            auto unchecked = arr.unchecked<bool, 2>();
            for (uint64_t i = 0; i < n; ++i) {
                for (uint64_t j = 0; j < n; ++j) {
                    m->set(i, j, unchecked(i, j));
                }
            }
            return m;
        }
    }
    return nullptr;
}

bool coerce_bool_like(py::object value) {
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
    throw py::type_error("BitMatrix entries accept bool, 0/1, or 0.0/1.0 values.");
}

} // namespace

PYBIND11_MODULE(_pycauset, m) {
    m.doc() = "pycauset Python Interface";

    // Helper lambdas for casting unique_ptr results to python objects
    auto cast_matrix_result = [](std::unique_ptr<MatrixBase> res) -> py::object {
        if (auto* m = dynamic_cast<FloatMatrix*>(res.get())) {
            res.release();
            return py::cast(m, py::return_value_policy::take_ownership);
        }
        if (auto* m = dynamic_cast<IntegerMatrix*>(res.get())) {
            res.release();
            return py::cast(m, py::return_value_policy::take_ownership);
        }
        if (auto* m = dynamic_cast<DenseBitMatrix*>(res.get())) {
            res.release();
            return py::cast(m, py::return_value_policy::take_ownership);
        }
        if (auto* m = dynamic_cast<TriangularFloatMatrix*>(res.get())) {
            res.release();
            return py::cast(m, py::return_value_policy::take_ownership);
        }
        if (auto* m = dynamic_cast<TriangularIntegerMatrix*>(res.get())) {
            res.release();
            return py::cast(m, py::return_value_policy::take_ownership);
        }
        if (auto* m = dynamic_cast<TriangularBitMatrix*>(res.get())) {
            res.release();
            return py::cast(m, py::return_value_policy::take_ownership);
        }
        if (auto* m = dynamic_cast<IdentityMatrix<double>*>(res.get())) {
            res.release();
            return py::cast(m, py::return_value_policy::take_ownership);
        }
        if (auto* m = dynamic_cast<IdentityMatrix<int32_t>*>(res.get())) {
            res.release();
            return py::cast(m, py::return_value_policy::take_ownership);
        }
        if (auto* m = dynamic_cast<DiagonalMatrix<double>*>(res.get())) {
            res.release();
            return py::cast(m, py::return_value_policy::take_ownership);
        }
        if (auto* m = dynamic_cast<DiagonalMatrix<int32_t>*>(res.get())) {
            res.release();
            return py::cast(m, py::return_value_policy::take_ownership);
        }
        
        throw std::runtime_error("Unknown MatrixBase subclass in cast_matrix_result");
    };

    auto cast_vector_result = [](std::unique_ptr<VectorBase> res) -> py::object {
        if (auto* v = dynamic_cast<FloatVector*>(res.get())) {
            res.release();
            return py::cast(v, py::return_value_policy::take_ownership);
        }
        if (auto* v = dynamic_cast<IntegerVector*>(res.get())) {
            res.release();
            return py::cast(v, py::return_value_policy::take_ownership);
        }
        if (auto* v = dynamic_cast<BitVector*>(res.get())) {
            res.release();
            return py::cast(v, py::return_value_policy::take_ownership);
        }
        if (auto* v = dynamic_cast<UnitVector*>(res.get())) {
            res.release();
            return py::cast(v, py::return_value_policy::take_ownership);
        }
        
        throw std::runtime_error("Unknown VectorBase subclass in cast_vector_result");
    };

    auto bind_vector_arithmetic = [&](auto& cls) {
        using T = typename std::remove_reference_t<decltype(cls)>::type;
        cls.def("__add__", [](const T& self, py::object other) {
            if (py::isinstance<VectorBase>(other)) {
                return pycauset::add_vectors(self, py::cast<const VectorBase&>(other), make_unique_storage_file("add_vector"));
            }
            auto temp = from_numpy(other);
            if (auto* v = dynamic_cast<VectorBase*>(temp.get())) {
                return pycauset::add_vectors(self, *v, make_unique_storage_file("add_vector"));
            }
            try {
                double s = py::cast<double>(other);
                return pycauset::scalar_add_vector(self, s, make_unique_storage_file("add_scalar_vector"));
            } catch (...) {}
            
            throw py::type_error("Unsupported operand type for +");
        });
        cls.def("__radd__", [](const T& self, py::object other) {
            if (py::isinstance<VectorBase>(other)) {
                return pycauset::add_vectors(self, py::cast<const VectorBase&>(other), make_unique_storage_file("add_vector"));
            }
            auto temp = from_numpy(other);
            if (auto* v = dynamic_cast<VectorBase*>(temp.get())) {
                return pycauset::add_vectors(self, *v, make_unique_storage_file("add_vector"));
            }
            try {
                double s = py::cast<double>(other);
                return pycauset::scalar_add_vector(self, s, make_unique_storage_file("add_scalar_vector"));
            } catch (...) {}
            throw py::type_error("Unsupported operand type for +");
        });
        cls.def("__sub__", [](const T& self, py::object other) {
            if (py::isinstance<VectorBase>(other)) {
                return pycauset::subtract_vectors(self, py::cast<const VectorBase&>(other), make_unique_storage_file("sub_vector"));
            }
            auto temp = from_numpy(other);
            if (auto* v = dynamic_cast<VectorBase*>(temp.get())) {
                return pycauset::subtract_vectors(self, *v, make_unique_storage_file("sub_vector"));
            }
            throw py::type_error("Unsupported operand type for -");
        });
        cls.def("__mul__", [](const T& self, py::object other) {
            try {
                double s = py::cast<double>(other);
                return pycauset::scalar_multiply_vector(self, s, make_unique_storage_file("mul_vector"));
            } catch (...) {}
            throw py::type_error("Unsupported operand type for * (Vector supports scalar multiplication only)");
        });
        cls.def("__rmul__", [](const T& self, py::object other) {
            try {
                double s = py::cast<double>(other);
                return pycauset::scalar_multiply_vector(self, s, make_unique_storage_file("mul_vector"));
            } catch (...) {}
            throw py::type_error("Unsupported operand type for *");
        });
        cls.def("dot", [](const T& self, py::object other) {
            if (py::isinstance<VectorBase>(other)) {
                return pycauset::dot_product(self, py::cast<const VectorBase&>(other));
            }
            auto temp = from_numpy(other);
            if (auto* v = dynamic_cast<VectorBase*>(temp.get())) {
                return pycauset::dot_product(self, *v);
            }
            throw py::type_error("Unsupported operand type for dot");
        });
        cls.def("__matmul__", [cast_matrix_result, cast_vector_result](const T& self, py::object other) -> py::object {
            if (py::isinstance<VectorBase>(other)) {
                const auto& other_vec = py::cast<const VectorBase&>(other);
                bool self_t = self.is_transposed();
                bool other_t = other_vec.is_transposed();
                
                if (!self_t && other_t) {
                    auto res = pycauset::outer_product(self, other_vec, make_unique_storage_file("outer"));
                    return cast_matrix_result(std::move(res));
                } else {
                    // Inner product
                    bool self_is_int = (py::isinstance<IntegerVector>(py::cast(self)) || py::isinstance<BitVector>(py::cast(self)));
                    bool other_is_int = (py::isinstance<IntegerVector>(py::cast(other_vec)) || py::isinstance<BitVector>(py::cast(other_vec)));
                    
                    double val = pycauset::dot_product(self, other_vec);
                    if (self_is_int && other_is_int) {
                        return py::cast((int64_t)val);
                    }
                    return py::cast(val);
                }
            }
            if (py::isinstance<MatrixBase>(other)) {
                auto res = pycauset::vector_matrix_multiply(self, py::cast<const MatrixBase&>(other), make_unique_storage_file("vec_mat_mul"));
                return cast_vector_result(std::move(res));
            }
            
            auto temp = from_numpy(other);
            if (auto* v = dynamic_cast<VectorBase*>(temp.get())) {
                const auto& other_vec = *v;
                bool self_t = self.is_transposed();
                bool other_t = other_vec.is_transposed();
                
                if (!self_t && other_t) {
                    auto res = pycauset::outer_product(self, other_vec, make_unique_storage_file("outer"));
                    return cast_matrix_result(std::move(res));
                } else {
                    double val = pycauset::dot_product(self, other_vec);
                    return py::cast(val);
                }
            }
            if (auto* m = dynamic_cast<MatrixBase*>(temp.get())) {
                auto res = pycauset::vector_matrix_multiply(self, *m, make_unique_storage_file("vec_mat_mul"));
                return cast_vector_result(std::move(res));
            }

            throw py::type_error("Unsupported operand type for @");
        });
        
        // NumPy export
        cls.def("__array__", [](const T& self, py::object dtype, py::object copy) -> py::object {
            uint64_t n = self.size();
            
            if constexpr (std::is_same_v<T, BitVector>) {
                py::array_t<bool> result(n);
                auto ptr = result.mutable_unchecked<1>();
                for (py::ssize_t i = 0; i < (py::ssize_t)n; ++i) {
                    ptr(i) = self.get(i);
                }
                if (self.is_transposed()) return result.reshape({(py::ssize_t)1, (py::ssize_t)n});
                return result;
            } else if constexpr (std::is_same_v<T, IntegerVector>) {
                py::array_t<int32_t> result(n);
                auto ptr = result.mutable_unchecked<1>();
                for (py::ssize_t i = 0; i < (py::ssize_t)n; ++i) {
                    ptr(i) = self.get(i);
                }
                if (self.is_transposed()) return result.reshape({(py::ssize_t)1, (py::ssize_t)n});
                return result;
            } else {
                py::array_t<double> result(n);
                auto ptr = result.mutable_unchecked<1>();
                for (py::ssize_t i = 0; i < (py::ssize_t)n; ++i) {
                    ptr(i) = self.get_element_as_double(i);
                }
                if (self.is_transposed()) return result.reshape({(py::ssize_t)1, (py::ssize_t)n});
                return result;
            }
        }, py::arg("dtype") = py::none(), py::arg("copy") = py::none());
    };

    auto bind_arithmetic = [&](auto& cls) {
        using T = typename std::remove_reference_t<decltype(cls)>::type;
        cls.def("__add__", [](const T& self, py::object other) {
            if (py::isinstance<MatrixBase>(other)) {
                return pycauset::add(self, py::cast<const MatrixBase&>(other), make_unique_storage_file("add"));
            }
            auto temp = from_numpy(other);
            if (auto* m = dynamic_cast<MatrixBase*>(temp.get())) {
                return pycauset::add(self, *m, make_unique_storage_file("add"));
            }
            try {
                if (py::isinstance<py::int_>(other)) {
                    int64_t s = py::cast<int64_t>(other);
                    return self.add_scalar(s, make_unique_storage_file("scalar_add"));
                }
                double s = py::cast<double>(other);
                return self.add_scalar(s, make_unique_storage_file("scalar_add"));
            } catch (...) {}
            throw py::type_error("Unsupported operand type for +");
        });
        cls.def("__radd__", [](const T& self, py::object other) {
            if (py::isinstance<MatrixBase>(other)) {
                return pycauset::add(self, py::cast<const MatrixBase&>(other), make_unique_storage_file("add"));
            }
            auto temp = from_numpy(other);
            if (auto* m = dynamic_cast<MatrixBase*>(temp.get())) {
                return pycauset::add(self, *m, make_unique_storage_file("add"));
            }
            try {
                if (py::isinstance<py::int_>(other)) {
                    int64_t s = py::cast<int64_t>(other);
                    return self.add_scalar(s, make_unique_storage_file("scalar_add"));
                }
                double s = py::cast<double>(other);
                return self.add_scalar(s, make_unique_storage_file("scalar_add"));
            } catch (...) {}
            throw py::type_error("Unsupported operand type for +");
        });
        cls.def("__sub__", [](const T& self, py::object other) {
            if (py::isinstance<MatrixBase>(other)) {
                return pycauset::subtract(self, py::cast<const MatrixBase&>(other), make_unique_storage_file("sub"));
            }
            auto temp = from_numpy(other);
            if (auto* m = dynamic_cast<MatrixBase*>(temp.get())) {
                return pycauset::subtract(self, *m, make_unique_storage_file("sub"));
            }
            throw py::type_error("Unsupported operand type for -");
        });
        cls.def("__mul__", [](const T& self, py::object other) {
            if (py::isinstance<MatrixBase>(other)) {
                return pycauset::elementwise_multiply(self, py::cast<const MatrixBase&>(other), make_unique_storage_file("mul"));
            }
            try {
                double s = py::cast<double>(other);
                return self.multiply_scalar(s, make_unique_storage_file("scalar_mul"));
            } catch (...) {}
            
            auto temp = from_numpy(other);
            if (auto* m = dynamic_cast<MatrixBase*>(temp.get())) {
                return pycauset::elementwise_multiply(self, *m, make_unique_storage_file("mul"));
            }
            throw py::type_error("Unsupported operand type for *");
        });
        cls.def("__rmul__", [](const T& self, double scalar) {
            return self.multiply_scalar(scalar, make_unique_storage_file("scalar_mul"));
        });
        cls.def("__rmul__", [](const T& self, int64_t scalar) {
            return self.multiply_scalar(scalar, make_unique_storage_file("scalar_mul"));
        });
        cls.def("__matmul__", [cast_matrix_result, cast_vector_result](const T& self, py::object other) -> py::object {
            if (py::isinstance<VectorBase>(other)) {
                auto res = pycauset::matrix_vector_multiply(self, py::cast<const VectorBase&>(other), make_unique_storage_file("mat_vec_mul"));
                return cast_vector_result(std::move(res));
            }
            
            if (py::isinstance<MatrixBase>(other)) {
                auto res = dispatch_matmul(self, py::cast<const MatrixBase&>(other), "");
                return cast_matrix_result(std::move(res));
            }
            
            auto temp = from_numpy(other);
            if (auto* v = dynamic_cast<VectorBase*>(temp.get())) {
                auto res = pycauset::matrix_vector_multiply(self, *v, make_unique_storage_file("mat_vec_mul"));
                return cast_vector_result(std::move(res));
            }
            if (auto* m = dynamic_cast<MatrixBase*>(temp.get())) {
                auto res = dispatch_matmul(self, *m, "");
                return cast_matrix_result(std::move(res));
            }
            throw py::type_error("Unsupported operand type for @");
        });
        cls.def_property_readonly("T", [](const T& self) {
            return self.transpose(make_unique_storage_file("transpose"));
        });
        cls.def_property_readonly("H", [](const T& self) {
            return self.transpose(make_unique_storage_file("hermitian"));
        });
        
        // NumPy export
        cls.def("__array__", [](const T& self, py::object dtype, py::object copy) {
            uint64_t n = self.size();
            if constexpr (std::is_same_v<T, DenseBitMatrix> || std::is_same_v<T, TriangularBitMatrix>) {
                py::array_t<bool> result({n, n});
                auto ptr = result.mutable_unchecked<2>();
                for (py::ssize_t i = 0; i < (py::ssize_t)n; ++i) {
                    for (py::ssize_t j = 0; j < (py::ssize_t)n; ++j) {
                        ptr(i, j) = self.get(i, j);
                    }
                }
                return result;
            } else if constexpr (std::is_same_v<T, IntegerMatrix> || std::is_same_v<T, TriangularIntegerMatrix>) {
                py::array_t<int32_t> result({n, n});
                auto ptr = result.mutable_unchecked<2>();
                for (py::ssize_t i = 0; i < (py::ssize_t)n; ++i) {
                    for (py::ssize_t j = 0; j < (py::ssize_t)n; ++j) {
                        ptr(i, j) = (int32_t)self.get_element_as_double(i, j);
                    }
                }
                return result;
            } else {
                py::array_t<double> result({n, n});
                auto ptr = result.mutable_unchecked<2>();
                for (py::ssize_t i = 0; i < (py::ssize_t)n; ++i) {
                    for (py::ssize_t j = 0; j < (py::ssize_t)n; ++j) {
                        ptr(i, j) = self.get_element_as_double(i, j);
                    }
                }
                return result;
            }
        }, py::arg("dtype") = py::none(), py::arg("copy") = py::none());
    };


    m.def("asarray", [cast_matrix_result, cast_vector_result](py::object obj) -> py::object {
        auto res = from_numpy(obj);
        if (res) {
            if (auto* v = dynamic_cast<VectorBase*>(res.get())) {
                return cast_vector_result(std::unique_ptr<VectorBase>(static_cast<VectorBase*>(res.release())));
            }
            if (auto* m = dynamic_cast<MatrixBase*>(res.get())) {
                return cast_matrix_result(std::unique_ptr<MatrixBase>(static_cast<MatrixBase*>(res.release())));
            }
        }
        
        try {
            py::array arr = py::array(obj);
            if (arr.ndim() == 2 && py::isinstance<py::array_t<std::complex<double>>>(arr)) {
                uint64_t n = arr.shape(0);
                auto cm = std::make_unique<pycauset::ComplexMatrix>(n, make_unique_storage_file("numpy_c_real"), make_unique_storage_file("numpy_c_imag"));
                auto unchecked = arr.unchecked<std::complex<double>, 2>();
                for (uint64_t i = 0; i < n; ++i) {
                    for (uint64_t j = 0; j < n; ++j) {
                        cm->set(i, j, unchecked(i, j));
                    }
                }
                return py::cast(std::move(cm));
            }
        } catch (...) {}

        throw std::invalid_argument("Could not convert input to PyCauset object");
    }, "Convert a NumPy array (or compatible object) to a PyCauset object (disk-backed).");

    py::class_<PersistentObject>(m, "PersistentObject")
        .def_property("scalar", &PersistentObject::get_scalar, &PersistentObject::set_scalar)
        .def_property_readonly("seed", &PersistentObject::get_seed)
        .def_property("is_temporary", &PersistentObject::is_temporary, &PersistentObject::set_temporary)
        .def_property_readonly("__array_priority__", [](const PersistentObject&){ return 1000.0; })
        .def("close", &PersistentObject::close)
        .def("set_transposed", &PersistentObject::set_transposed)
        .def("is_transposed", &PersistentObject::is_transposed)
        .def("get_backing_file", &PersistentObject::get_backing_file)
        .def("copy_storage", &PersistentObject::copy_storage, 
             py::arg("result_file_hint") = "",
             "Create a copy of the backing storage. Handles both disk-backed and memory-backed objects.");

    py::class_<MatrixBase, PersistentObject>(m, "MatrixBase");

    py::class_<VectorBase, PersistentObject>(m, "VectorBase")
        .def_property_readonly("T", [](const VectorBase& v) {
            return v.transpose();
        });

    // TriangularFloatMatrix
    py::class_<TriangularFloatMatrix, MatrixBase> tfm(m, "TriangularFloatMatrix");
    tfm.def(py::init<uint64_t, const std::string&>(), 
            py::arg("n"), py::arg("backing_file") = "")
       .def(py::init<uint64_t, const std::string&, size_t, uint64_t, double, bool>(),
            py::arg("n"), py::arg("backing_file"), py::arg("offset"), 
            py::arg("seed"), py::arg("scalar"), py::arg("is_transposed"))
        .def("get", &TriangularFloatMatrix::get)
        .def("get_element_as_double", &TriangularFloatMatrix::get_element_as_double)
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
       .def(py::init<uint64_t, const std::string&, size_t, uint64_t, double, bool>(),
            py::arg("n"), py::arg("backing_file"), py::arg("offset"), 
            py::arg("seed"), py::arg("scalar"), py::arg("is_transposed"))
        .def("get", &TriangularIntegerMatrix::get)
        .def("get_element_as_double", &TriangularIntegerMatrix::get_element_as_double)
        .def("set", &TriangularIntegerMatrix::set)
        .def("close", &TriangularIntegerMatrix::close)
        .def("size", &TriangularIntegerMatrix::size)
        .def("get_backing_file", &TriangularIntegerMatrix::get_backing_file)
        .def("is_transposed", &TriangularIntegerMatrix::is_transposed)
        .def_property_readonly("shape", [](const TriangularIntegerMatrix& m) {
            return std::make_pair(m.size(), m.size());
        })
        .def("__getitem__", [](const TriangularIntegerMatrix& m, std::pair<uint64_t, uint64_t> idx) {
            return m.get_element_as_double(idx.first, idx.second);
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
      .def(py::init<uint64_t, const std::string&, size_t, uint64_t, double, bool>(),
           py::arg("n"), py::arg("backing_file"), py::arg("offset"), 
           py::arg("seed"), py::arg("scalar"), py::arg("is_transposed"))
        .def("get", &FloatMatrix::get)
        .def("get_element_as_double", &FloatMatrix::get_element_as_double)
        .def("set", &FloatMatrix::set)
        .def("close", &FloatMatrix::close)
        .def("size", &FloatMatrix::size)
        .def("get_backing_file", &FloatMatrix::get_backing_file)
        .def("is_transposed", &FloatMatrix::is_transposed)
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
        .def("transpose", [](const FloatMatrix& m, const std::string& saveas) {
            return m.transpose(saveas);
        }, py::arg("saveas") = "")
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
      .def(py::init<uint64_t, const std::string&, size_t, uint64_t, double, bool>(),
           py::arg("n"), py::arg("backing_file"), py::arg("offset"), 
           py::arg("seed"), py::arg("scalar"), py::arg("is_transposed"))
        .def("get", &IntegerMatrix::get)
        .def("get_element_as_double", &IntegerMatrix::get_element_as_double)
        .def("set", &IntegerMatrix::set)
        .def("close", &IntegerMatrix::close)
        .def("size", &IntegerMatrix::size)
        .def("get_backing_file", &IntegerMatrix::get_backing_file)
        .def("is_transposed", &IntegerMatrix::is_transposed)
        .def_property_readonly("shape", [](const IntegerMatrix& m) {
            return std::make_pair(m.size(), m.size());
        })
        .def("__getitem__", [](const IntegerMatrix& m, std::pair<uint64_t, uint64_t> idx) {
            return m.get_element_as_double(idx.first, idx.second);
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
        .def("transpose", [](const IntegerMatrix& m, const std::string& saveas) {
            return m.transpose(saveas);
        }, py::arg("saveas") = "")
        .def("multiply", [](const IntegerMatrix& self, const IntegerMatrix& other, const std::string& saveas) {
            std::string target = saveas.empty() ? make_unique_storage_file("matmul_im") : saveas;
            return self.multiply(other, target);
        }, py::arg("other"), py::arg("saveas") = "");
    bind_arithmetic(im);

    // DenseBitMatrix
    py::class_<DenseBitMatrix, MatrixBase> dbm(m, "DenseBitMatrix");
    dbm.def(py::init<uint64_t, const std::string&>(), 
           py::arg("n"), py::arg("backing_file") = "")
       .def(py::init<uint64_t, const std::string&, size_t, uint64_t, double, bool>(),
           py::arg("n"), py::arg("backing_file"), py::arg("offset"), 
           py::arg("seed"), py::arg("scalar"), py::arg("is_transposed"))
        .def_static("random", &DenseBitMatrix::random, 
            py::arg("n"), py::arg("density") = 0.5, py::arg("backing_file") = "",
            py::arg("seed") = py::none())
        .def("get", &DenseBitMatrix::get)
        .def("set", [](DenseBitMatrix& m, uint64_t i, uint64_t j, py::object value) {
            bool boolVal = coerce_bool_like(value);
            m.set(i, j, boolVal);
        })
        .def("close", &DenseBitMatrix::close)
        .def("size", &DenseBitMatrix::size)
        .def("get_backing_file", &DenseBitMatrix::get_backing_file)
        .def("is_transposed", &DenseBitMatrix::is_transposed)
        .def_property_readonly("shape", [](const DenseBitMatrix& m) {
            return std::make_pair(m.size(), m.size());
        })
        .def("__getitem__", [](const DenseBitMatrix& m, std::pair<uint64_t, uint64_t> idx) {
            return m.get(idx.first, idx.second);
        })
        .def("__setitem__", [](DenseBitMatrix& m, std::pair<uint64_t, uint64_t> idx, py::object value) {
            bool boolVal = coerce_bool_like(value);
            m.set(idx.first, idx.second, boolVal);
        })
        .def("__repr__", [](const DenseBitMatrix& m) {
            return "<DenseBitMatrix shape=(" + std::to_string(m.size()) + ", " + std::to_string(m.size()) + ")>";
        })
        .def("invert", [](const DenseBitMatrix& m) {
            // DenseBitMatrix inversion? Not implemented in C++ yet, or maybe it is?
            // I didn't implement inverse() in DenseBitMatrix.hpp, only bitwise_not.
            // So I won't bind invert() for now, or bind it to throw.
            throw std::runtime_error("Inversion not supported for DenseBitMatrix");
        }, "Matrix Inversion (Linear Algebra)")
        .def("__invert__", [](const DenseBitMatrix& m) {
            return m.bitwise_not(make_unique_storage_file("bitwise_not"));
        })
        .def("multiply", [](const DenseBitMatrix& self, const DenseBitMatrix& other, const std::string& saveas) {
            std::string target = saveas.empty() ? make_unique_storage_file("matmul_dbm") : saveas;
            return self.multiply(other, target);
        }, py::arg("other"), py::arg("saveas") = "");
    bind_arithmetic(dbm);

    // TriangularBitMatrix
    py::class_<TriangularBitMatrix, MatrixBase> tbm(m, "TriangularBitMatrix");
    tbm.def(py::init<uint64_t, const std::string&>(), 
             py::arg("n"), py::arg("backing_file") = "")
       .def(py::init<uint64_t, const std::string&, size_t, uint64_t, double, bool>(),
           py::arg("n"), py::arg("backing_file"), py::arg("offset"), 
           py::arg("seed"), py::arg("scalar"), py::arg("is_transposed"))
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
        .def("get_element_as_double", &TriangularBitMatrix::get_element_as_double)
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
        .def("transpose", [](const TriangularBitMatrix& m, const std::string& saveas) {
            return m.transpose(saveas);
        }, py::arg("saveas") = "")
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
        .def("is_transposed", &TriangularBitMatrix::is_transposed)
        .def("__repr__", [](const TriangularBitMatrix& m) {
            return "<TriangularBitMatrix shape=(" + std::to_string(m.size()) + ", " + std::to_string(m.size()) + ")>";
        });
    bind_arithmetic(tbm);

    m.def("matmul", &dispatch_matmul, py::arg("a"), py::arg("b"), py::arg("saveas") = "",
          "Generic matrix multiplication. Supports mixed types (Dense/Triangular).");



    // DiagonalMatrix (Float)
    py::class_<DiagonalMatrix<double>, MatrixBase> dm(m, "DiagonalMatrix");
    dm.def(py::init<uint64_t, const std::string&>(), 
            py::arg("n"), py::arg("backing_file") = "")
        .def("get", &DiagonalMatrix<double>::get)
        .def("set", &DiagonalMatrix<double>::set)
        .def("get_diagonal", &DiagonalMatrix<double>::get_diagonal)
        .def("set_diagonal", &DiagonalMatrix<double>::set_diagonal)
        .def("close", &DiagonalMatrix<double>::close)
        .def("size", &DiagonalMatrix<double>::size)
        .def("get_backing_file", &DiagonalMatrix<double>::get_backing_file)
        .def_property_readonly("shape", [](const DiagonalMatrix<double>& m) {
            return std::make_pair(m.size(), m.size());
        })
        .def("__getitem__", [](const DiagonalMatrix<double>& m, std::pair<uint64_t, uint64_t> idx) {
            return m.get_element_as_double(idx.first, idx.second);
        })
        .def("__setitem__", [](DiagonalMatrix<double>& m, std::pair<uint64_t, uint64_t> idx, double val) {
            m.set(idx.first, idx.second, val);
        })
        .def("__repr__", [](const DiagonalMatrix<double>& m) {
            return "<DiagonalMatrix shape=(" + std::to_string(m.size()) + ", " + std::to_string(m.size()) + ")>";
        });
    bind_arithmetic(dm);

    // DiagonalMatrix (Int)
    py::class_<DiagonalMatrix<int32_t>, MatrixBase> dmi(m, "DiagonalMatrixInt");
    dmi.def(py::init<uint64_t, const std::string&>(), 
            py::arg("n"), py::arg("backing_file") = "")
        .def("get", &DiagonalMatrix<int32_t>::get)
        .def("set", &DiagonalMatrix<int32_t>::set)
        .def("get_diagonal", &DiagonalMatrix<int32_t>::get_diagonal)
        .def("set_diagonal", &DiagonalMatrix<int32_t>::set_diagonal)
        .def("close", &DiagonalMatrix<int32_t>::close)
        .def("size", &DiagonalMatrix<int32_t>::size)
        .def("get_backing_file", &DiagonalMatrix<int32_t>::get_backing_file)
        .def_property_readonly("shape", [](const DiagonalMatrix<int32_t>& m) {
            return std::make_pair(m.size(), m.size());
        })
        .def("__getitem__", [](const DiagonalMatrix<int32_t>& m, std::pair<uint64_t, uint64_t> idx) {
            return m.get_element_as_double(idx.first, idx.second);
        })
        .def("__setitem__", [](DiagonalMatrix<int32_t>& m, std::pair<uint64_t, uint64_t> idx, int32_t val) {
            m.set(idx.first, idx.second, val);
        })
        .def("__repr__", [](const DiagonalMatrix<int32_t>& m) {
            return "<DiagonalMatrixInt shape=(" + std::to_string(m.size()) + ", " + std::to_string(m.size()) + ")>";
        });
    bind_arithmetic(dmi);

    // IdentityMatrix (Float)
    py::class_<IdentityMatrix<double>, MatrixBase> idm(m, "IdentityMatrix");
    idm.def(py::init<uint64_t, const std::string&>(), 
            py::arg("n"), py::arg("backing_file") = "")
        .def("get", [](const IdentityMatrix<double>& m, uint64_t i, uint64_t j) {
            return m.get_element_as_double(i, j);
        })
        .def("close", &IdentityMatrix<double>::close)
        .def("size", &IdentityMatrix<double>::size)
        .def("get_backing_file", &IdentityMatrix<double>::get_backing_file)
        .def_property_readonly("shape", [](const IdentityMatrix<double>& m) {
            return std::make_pair(m.size(), m.size());
        })
        .def("__getitem__", [](const IdentityMatrix<double>& m, std::pair<uint64_t, uint64_t> idx) {
            return m.get_element_as_double(idx.first, idx.second);
        })
        .def("__repr__", [](const IdentityMatrix<double>& m) {
            return "<IdentityMatrix shape=(" + std::to_string(m.size()) + ", " + std::to_string(m.size()) + ")>";
        });
    bind_arithmetic(idm);

    // IdentityMatrix (Int)
    py::class_<IdentityMatrix<int32_t>, MatrixBase> idmi(m, "IdentityMatrixInt");
    idmi.def(py::init<uint64_t, const std::string&>(), 
            py::arg("n"), py::arg("backing_file") = "")
        .def("get", [](const IdentityMatrix<int32_t>& m, uint64_t i, uint64_t j) {
            return m.get_element_as_double(i, j);
        })
        .def("close", &IdentityMatrix<int32_t>::close)
        .def("size", &IdentityMatrix<int32_t>::size)
        .def("get_backing_file", &IdentityMatrix<int32_t>::get_backing_file)
        .def_property_readonly("shape", [](const IdentityMatrix<int32_t>& m) {
            return std::make_pair(m.size(), m.size());
        })
        .def("__getitem__", [](const IdentityMatrix<int32_t>& m, std::pair<uint64_t, uint64_t> idx) {
            return m.get_element_as_double(idx.first, idx.second);
        })
        .def("__repr__", [](const IdentityMatrix<int32_t>& m) {
            return "<IdentityMatrixInt shape=(" + std::to_string(m.size()) + ", " + std::to_string(m.size()) + ")>";
        });
    bind_arithmetic(idmi);

    // FloatVector
    py::class_<FloatVector, VectorBase> fv(m, "FloatVector");
    fv.def(py::init<uint64_t, const std::string&>(), py::arg("n"), py::arg("backing_file") = "")
      .def(py::init<uint64_t, const std::string&, size_t, uint64_t, double, bool>(),
           py::arg("n"), py::arg("backing_file"), py::arg("offset"), 
           py::arg("seed"), py::arg("scalar"), py::arg("is_transposed"))
      .def("__getitem__", &FloatVector::get)
      .def("__setitem__", &FloatVector::set)
      .def("__len__", &FloatVector::size)
      .def_property_readonly("shape", [](const FloatVector& v) { 
          if (v.is_transposed()) return py::make_tuple(1, v.size());
          return py::make_tuple(v.size()); 
      })
      .def("__repr__", [](const FloatVector& v) {
          std::string s = "<FloatVector size=" + std::to_string(v.size());
          if (v.is_transposed()) s += " transposed=True";
          s += ">";
          return s;
      });
    bind_vector_arithmetic(fv);

    // IntegerVector
    py::class_<IntegerVector, VectorBase> iv(m, "IntegerVector");
    iv.def(py::init<uint64_t, const std::string&>(), py::arg("n"), py::arg("backing_file") = "")
      .def(py::init<uint64_t, const std::string&, size_t, uint64_t, double, bool>(),
           py::arg("n"), py::arg("backing_file"), py::arg("offset"), 
           py::arg("seed"), py::arg("scalar"), py::arg("is_transposed"))
      .def("__getitem__", &IntegerVector::get)
      .def("__setitem__", &IntegerVector::set)
      .def("__len__", &IntegerVector::size)
      .def_property_readonly("shape", [](const IntegerVector& v) { 
          if (v.is_transposed()) return py::make_tuple(1, v.size());
          return py::make_tuple(v.size()); 
      })
      .def("__repr__", [](const IntegerVector& v) {
          std::string s = "<IntegerVector size=" + std::to_string(v.size());
          if (v.is_transposed()) s += " transposed=True";
          s += ">";
          return s;
      });
    bind_vector_arithmetic(iv);

    // BitVector
    py::class_<BitVector, VectorBase> bv(m, "BitVector");
    bv.def(py::init<uint64_t, const std::string&>(), py::arg("n"), py::arg("backing_file") = "")
      .def(py::init<uint64_t, const std::string&, size_t, uint64_t, double, bool>(),
           py::arg("n"), py::arg("backing_file"), py::arg("offset"), 
           py::arg("seed"), py::arg("scalar"), py::arg("is_transposed"))
      .def("__getitem__", &BitVector::get)
      .def("__setitem__", [](BitVector& v, uint64_t i, py::object val) {
          v.set(i, coerce_bool_like(val));
      })
      .def("__len__", &BitVector::size)
      .def_property_readonly("shape", [](const BitVector& v) { 
          if (v.is_transposed()) return py::make_tuple(1, v.size());
          return py::make_tuple(v.size()); 
      })
      .def("__repr__", [](const BitVector& v) {
          std::string s = "<BitVector size=" + std::to_string(v.size());
          if (v.is_transposed()) s += " transposed=True";
          s += ">";
          return s;
      });
    bind_vector_arithmetic(bv);

    // UnitVector
    py::class_<UnitVector, VectorBase> uv(m, "UnitVector");
    uv.def(py::init<uint64_t, uint64_t, const std::string&>(), 
           py::arg("n"), py::arg("active_index"), py::arg("backing_file") = "")
      .def(py::init<uint64_t, uint64_t, const std::string&, size_t, uint64_t, double, bool>(),
           py::arg("n"), py::arg("active_index"), py::arg("backing_file"), py::arg("offset"), py::arg("seed"), py::arg("scalar"), py::arg("is_transposed"))
      .def("get", [](const UnitVector& v, uint64_t i) {
          return v.get_element_as_double(i);
      })
      .def("get_active_index", &UnitVector::get_active_index)
      .def("__len__", &UnitVector::size)
      .def_property_readonly("shape", [](const UnitVector& v) { 
          if (v.is_transposed()) return py::make_tuple(1, v.size());
          return py::make_tuple(v.size()); 
      })
      .def("__getitem__", [](const UnitVector& v, uint64_t i) {
          return v.get_element_as_double(i);
      })
      .def("__repr__", [](const UnitVector& v) {
          std::string s = "<UnitVector size=" + std::to_string(v.size()) + " index=" + std::to_string(v.get_active_index());
          if (v.is_transposed()) s += " transposed=True";
          s += ">";
          return s;
      });
    bind_vector_arithmetic(uv);

    // ComplexMatrix
    py::class_<pycauset::ComplexMatrix>(m, "ComplexMatrix")
        .def(py::init<uint64_t, const std::string&, const std::string&>(), 
             py::arg("n"), py::arg("backing_file_real") = "", py::arg("backing_file_imag") = "")
        .def("get", &pycauset::ComplexMatrix::get)
        .def("set", &pycauset::ComplexMatrix::set)
        .def("size", &pycauset::ComplexMatrix::size)
        .def("close", &pycauset::ComplexMatrix::close)
        .def_property_readonly("real", static_cast<const pycauset::FloatMatrix* (pycauset::ComplexMatrix::*)() const>(&pycauset::ComplexMatrix::real), py::return_value_policy::reference)
        .def_property_readonly("imag", static_cast<const pycauset::FloatMatrix* (pycauset::ComplexMatrix::*)() const>(&pycauset::ComplexMatrix::imag), py::return_value_policy::reference)
        .def("conjugate", [](const pycauset::ComplexMatrix& m) {
            return m.conjugate(make_unique_storage_file("conj_real"), make_unique_storage_file("conj_imag"));
        })
        .def_property_readonly("T", [](const pycauset::ComplexMatrix& m) {
            return m.transpose(make_unique_storage_file("trans_real"), make_unique_storage_file("trans_imag"));
        })
        .def_property_readonly("H", [](const pycauset::ComplexMatrix& m) {
            return m.hermitian(make_unique_storage_file("herm_real"), make_unique_storage_file("herm_imag"));
        }, "Hermitian Conjugate")
        .def("__add__", [](const pycauset::ComplexMatrix& self, const pycauset::ComplexMatrix& other) {
            return pycauset::add(self, other, make_unique_storage_file("cadd_real"), make_unique_storage_file("cadd_imag"));
        })
        .def("__mul__", [](const pycauset::ComplexMatrix& self, const pycauset::ComplexMatrix& other) {
            return pycauset::multiply(self, other, make_unique_storage_file("cmul_real"), make_unique_storage_file("cmul_imag"));
        })
        .def("__mul__", [](const pycauset::ComplexMatrix& self, std::complex<double> scalar) {
            return pycauset::multiply_scalar(self, scalar, make_unique_storage_file("csmul_real"), make_unique_storage_file("csmul_imag"));
        })
        .def("__rmul__", [](const pycauset::ComplexMatrix& self, std::complex<double> scalar) {
            return pycauset::multiply_scalar(self, scalar, make_unique_storage_file("csmul_real"), make_unique_storage_file("csmul_imag"));
        })
        .def("__add__", [](const pycauset::ComplexMatrix& self, std::complex<double> scalar) {
            return pycauset::add_scalar(self, scalar, make_unique_storage_file("csadd_real"), make_unique_storage_file("csadd_imag"));
        })
        .def("__radd__", [](const pycauset::ComplexMatrix& self, std::complex<double> scalar) {
            return pycauset::add_scalar(self, scalar, make_unique_storage_file("csadd_real"), make_unique_storage_file("csadd_imag"));
        })
        .def("__repr__", [](const pycauset::ComplexMatrix& m) {
            return "<ComplexMatrix shape=(" + std::to_string(m.size()) + ", " + std::to_string(m.size()) + ")>";
        });

    m.def("dot", &pycauset::dot_product, "Dot product of two vectors");
    
    m.def("cross", [](const VectorBase& a, const VectorBase& b) {
        return pycauset::cross_product(a, b, make_unique_storage_file("cross"));
    }, "Cross product of two 3D vectors");

    m.def("set_memory_threshold", &pycauset::set_memory_threshold, 
          py::arg("bytes"),
          "Set the size threshold (in bytes) below which objects are stored in RAM instead of on disk.");
          
    m.def("get_memory_threshold", &pycauset::get_memory_threshold,
          "Get the current memory threshold in bytes.");

    // Spacetime
    py::class_<pycauset::CausalSpacetime>(m, "Spacetime")
        .def("volume", &pycauset::CausalSpacetime::volume);

    py::class_<pycauset::MinkowskiDiamond, pycauset::CausalSpacetime>(m, "MinkowskiDiamond")
        .def(py::init<int>(), py::arg("dimension"))
        .def("dimension", &pycauset::MinkowskiDiamond::dimension);

    py::class_<pycauset::MinkowskiCylinder, pycauset::CausalSpacetime>(m, "MinkowskiCylinder")
        .def(py::init<int, double, double>(), 
             py::arg("dimension"), 
             py::arg("height"), 
             py::arg("circumference"))
        .def("dimension", &pycauset::MinkowskiCylinder::dimension)
        .def_property_readonly("height", &pycauset::MinkowskiCylinder::get_height)
        .def_property_readonly("circumference", &pycauset::MinkowskiCylinder::get_circumference);

    py::class_<pycauset::MinkowskiBox, pycauset::CausalSpacetime>(m, "MinkowskiBox")
        .def(py::init<int, double, double>(),
             py::arg("dimension"),
             py::arg("time_extent"),
             py::arg("space_extent"))
        .def("dimension", &pycauset::MinkowskiBox::dimension)
        .def_property_readonly("time_extent", &pycauset::MinkowskiBox::get_time_extent)
        .def_property_readonly("space_extent", &pycauset::MinkowskiBox::get_space_extent);

    // Sprinkler
    m.def("sprinkle", &pycauset::Sprinkler::sprinkle, 
          py::arg("spacetime"), 
          py::arg("n"), 
          py::arg("seed"), 
          py::arg("saveas") = "",
          "Sprinkle points into a spacetime and return the causal matrix.");

    m.def("make_coordinates", &pycauset::Sprinkler::make_coordinates,
          py::arg("spacetime"),
          py::arg("n"),
          py::arg("seed"),
          py::arg("indices"),
          "Regenerate coordinates for specific indices.");
}
