#include "bindings/bindings_common.hpp"

#include "pycauset/core/Errors.hpp"
#include "pycauset/core/MemoryGovernor.hpp"

PYBIND11_MODULE(_pycauset, m) {
    m.doc() = "PyCauset: High-performance Causal Set Theory library";

    // pycauset::dtype_error is a TypeError, pycauset::linalg_error NumPy's LinAlgError
    // (core/Errors.hpp).
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const pycauset::dtype_error& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
        } catch (const pycauset::linalg_error& e) {
            // NumPy's own class, so `except np.linalg.LinAlgError` works as with NumPy.
            py::object cls = py::module_::import("numpy.linalg").attr("LinAlgError");
            PyErr_SetString(cls.ptr(), e.what());
        }
    });

    pycauset::init_thread_budget();  // the OpenBLAS default (core/ParallelUtils.hpp)

    bind_core_classes(m);
    bind_matrix_classes(m);
    bind_vector_classes(m);
    bind_math_ops(m);
    bind_causet_classes(m);

    // --- Memory Governor ---
    py::class_<pycauset::core::MemoryGovernor>(m, "MemoryGovernor")
        .def_static("instance", &pycauset::core::MemoryGovernor::instance, py::return_value_policy::reference)
        .def("get_total_system_ram", &pycauset::core::MemoryGovernor::get_total_system_ram)
        .def("get_available_system_ram", &pycauset::core::MemoryGovernor::get_available_system_ram)
        .def("get_safety_margin", &pycauset::core::MemoryGovernor::get_safety_margin)
        .def("set_safety_margin", &pycauset::core::MemoryGovernor::set_safety_margin)
        .def("can_fit_in_ram", &pycauset::core::MemoryGovernor::can_fit_in_ram);
}
