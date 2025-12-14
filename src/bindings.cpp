#include "bindings/bindings_common.hpp"
// Force rebuild 3
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/math/Eigen.hpp"
#include "pycauset/vector/ComplexVector.hpp"
#include "pycauset/core/MemoryGovernor.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

using namespace pycauset;

void bind_core_classes(py::module_& m) {
    // Minimal implementation
}

void bind_matrix_classes(py::module_& m) {
    py::class_<MatrixBase, std::shared_ptr<MatrixBase>>(m, "MatrixBase");

    py::class_<DenseMatrix<double>, std::shared_ptr<DenseMatrix<double>>>(m, "DenseMatrixFloat64", py::buffer_protocol())
        .def(py::init([](int n) {
            return std::make_shared<DenseMatrix<double>>(n);
        }), py::arg("n"))
        .def(py::init([](int n, std::string backing_file) {
            return std::make_shared<DenseMatrix<double>>(n, backing_file);
        }), py::arg("n"), py::arg("backing_file"))
        .def_buffer([](DenseMatrix<double> &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                               /* Pointer to buffer */
                sizeof(double),                         /* Size of one scalar */
                py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
                2,                                      /* Number of dimensions */
                { m.rows(), m.cols() },                 /* Buffer dimensions */
                { sizeof(double) * m.cols(),             /* Strides (in bytes) for each index */
                  sizeof(double) }
            );
        })
        .def("inverse", [](const DenseMatrix<double>& m, const std::string& result_file) {
            return m.inverse(result_file);
        }, py::arg("result_file") = "")
        .def("inverse_to", &DenseMatrix<double>::inverse_to)
        .def("set_identity", &DenseMatrix<double>::set_identity)
        .def("fill", &DenseMatrix<double>::fill)
        .def("matmul", [](const DenseMatrix<double>& m, const DenseMatrix<double>& other, const std::string& result_file) {
            return std::shared_ptr<DenseMatrix<double>>(m.multiply(other, result_file));
        }, py::arg("other"), py::arg("result_file") = "")
        .def("__matmul__", [](const DenseMatrix<double>& m, const DenseMatrix<double>& other) {
            return std::shared_ptr<DenseMatrix<double>>(m.multiply(other, ""));
        });

    // --- Float32 Support ---
    py::class_<DenseMatrix<float>, std::shared_ptr<DenseMatrix<float>>>(m, "DenseMatrixFloat32", py::buffer_protocol())
        .def(py::init([](int n) {
            return std::make_shared<DenseMatrix<float>>(n);
        }), py::arg("n"))
        .def(py::init([](int n, std::string backing_file) {
            return std::make_shared<DenseMatrix<float>>(n, backing_file);
        }), py::arg("n"), py::arg("backing_file"))
        .def_buffer([](DenseMatrix<float> &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                               /* Pointer to buffer */
                sizeof(float),                          /* Size of one scalar */
                py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
                2,                                      /* Number of dimensions */
                { m.rows(), m.cols() },                 /* Buffer dimensions */
                { sizeof(float) * m.cols(),             /* Strides (in bytes) for each index */
                  sizeof(float) }
            );
        })
        .def("inverse", [](const DenseMatrix<float>& m, const std::string& result_file) {
            return m.inverse(result_file);
        }, py::arg("result_file") = "")
        .def("inverse_to", &DenseMatrix<float>::inverse_to)
        .def("set_identity", &DenseMatrix<float>::set_identity)
        .def("fill", &DenseMatrix<float>::fill)
        .def("matmul", [](const DenseMatrix<float>& m, const DenseMatrix<float>& other, const std::string& result_file) {
            return std::shared_ptr<DenseMatrix<float>>(m.multiply(other, result_file));
        }, py::arg("other"), py::arg("result_file") = "")
        .def("__matmul__", [](const DenseMatrix<float>& m, const DenseMatrix<float>& other) {
            return std::shared_ptr<DenseMatrix<float>>(m.multiply(other, ""));
        });

    m.def("asarray", [](py::array_t<double> array) -> std::shared_ptr<DenseMatrix<double>> {
        auto buf = array.request();
        if (buf.ndim != 2) throw std::runtime_error("Number of dimensions must be 2");
        if (buf.shape[0] != buf.shape[1]) throw std::runtime_error("Matrix must be square");
        
        uint64_t n = buf.shape[0];
        auto result = std::make_shared<DenseMatrix<double>>(n);
        
        // Optimized Copy
        double* src_ptr = static_cast<double*>(buf.ptr);
        double* dst_ptr = result->data();
        
        // Check for contiguous C-style array
        if (buf.strides[1] == sizeof(double) && buf.strides[0] == n * sizeof(double)) {
            std::memcpy(dst_ptr, src_ptr, n * n * sizeof(double));
        } else {
            // Fallback for non-contiguous arrays
            for (uint64_t i = 0; i < n; ++i) {
                for (uint64_t j = 0; j < n; ++j) {
                    // Use raw pointer access, avoid set() overhead
                    dst_ptr[i * n + j] = src_ptr[i * buf.strides[0] / sizeof(double) + j * buf.strides[1] / sizeof(double)];
                }
            }
        }
        return result;
    });

    m.def("asarray", [](py::array_t<float> array) -> std::shared_ptr<DenseMatrix<float>> {
        auto buf = array.request();
        if (buf.ndim != 2) throw std::runtime_error("Number of dimensions must be 2");
        if (buf.shape[0] != buf.shape[1]) throw std::runtime_error("Matrix must be square");
        
        uint64_t n = buf.shape[0];
        auto result = std::make_shared<DenseMatrix<float>>(n);
        
        // Optimized Copy
        float* src_ptr = static_cast<float*>(buf.ptr);
        float* dst_ptr = result->data();
        
        // Check for contiguous C-style array
        if (buf.strides[1] == sizeof(float) && buf.strides[0] == n * sizeof(float)) {
            std::memcpy(dst_ptr, src_ptr, n * n * sizeof(float));
        } else {
            // Fallback for non-contiguous arrays
            for (uint64_t i = 0; i < n; ++i) {
                for (uint64_t j = 0; j < n; ++j) {
                    // Use raw pointer access, avoid set() overhead
                    dst_ptr[i * n + j] = src_ptr[i * buf.strides[0] / sizeof(float) + j * buf.strides[1] / sizeof(float)];
                }
            }
        }
        return result;
    });
    
    m.def("eigvals", [](std::shared_ptr<MatrixBase> matrix) {
        auto result = pycauset::eigvals(*matrix);
        uint64_t n = result->size();
        py::array_t<std::complex<double>> arr(n);
        auto ptr = static_cast<std::complex<double>*>(arr.request().ptr);
        for (uint64_t i = 0; i < n; ++i) {
            ptr[i] = result->get(i);
        }
        return arr;
    });
}

void bind_vector_classes(py::module_& m) {}
void bind_complex_classes(py::module_& m) {}
void bind_causet_classes(py::module_& m) {}

PYBIND11_MODULE(_pycauset, m) {
    m.doc() = "PyCauset: High-performance Causal Set Theory library";

    bind_core_classes(m);
    bind_matrix_classes(m);
    bind_vector_classes(m);
    bind_complex_classes(m);
    bind_causet_classes(m);

    // --- Memory Governor ---
    py::class_<pycauset::core::MemoryGovernor>(m, "MemoryGovernor")
        .def_static("instance", &pycauset::core::MemoryGovernor::instance, py::return_value_policy::reference)
        .def("get_total_system_ram", &pycauset::core::MemoryGovernor::get_total_system_ram)
        .def("get_available_system_ram", &pycauset::core::MemoryGovernor::get_available_system_ram)
        .def("get_safety_margin", &pycauset::core::MemoryGovernor::get_safety_margin)
        .def("set_safety_margin", &pycauset::core::MemoryGovernor::set_safety_margin)
        .def("can_fit_in_ram", &pycauset::core::MemoryGovernor::can_fit_in_ram)
        .def("should_use_direct_path", &pycauset::core::MemoryGovernor::should_use_direct_path);
}
