#include "bindings/bindings_common.hpp"

#include "pycauset/core/Errors.hpp"
#include "pycauset/core/MemoryGovernor.hpp"
#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/vector/VectorBase.hpp"

PYBIND11_MODULE(_pycauset, m) {
    m.doc() = "PyCauset: High-performance Causal Set Theory library";

    // pycauset::dtype_error is a TypeError, pycauset::linalg_error NumPy's LinAlgError
    // (core/Errors.hpp).
    // Local to _pycauset: a global translator applied to every pybind11 module of this ABI
    // (review 3.7 C-16).
    py::register_local_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const pycauset::interrupted&) {
            // PyErr_CheckSignals raised the KeyboardInterrupt (or another signal handler's
            // exception) already: it goes on as it is.
            if (!PyErr_Occurred()) PyErr_SetNone(PyExc_KeyboardInterrupt);
        } catch (const pycauset::dtype_error& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
        } catch (const py::cast_error& e) {
            // A value of the wrong type reached a pybind11 cast: TypeError, as NumPy raises
            // (it surfaced as RuntimeError "Unable to cast", review 3.7 C-6).
            PyErr_SetString(PyExc_TypeError, e.what());
        } catch (const pycauset::linalg_error& e) {
            // NumPy's own class, so `except np.linalg.LinAlgError` works as with NumPy.
            py::object cls = py::module_::import("numpy.linalg").attr("LinAlgError");
            PyErr_SetString(cls.ptr(), e.what());
        }
    });

    pycauset::init_thread_budget();  // the OpenBLAS default (core/ParallelUtils.hpp)

    // The base class shells first, so every signature names them: functions bound before a class
    // showed its C++ name (`-> pycauset::VectorBase`, review 3.7 C-14). The bind_* files add
    // the methods to these.
    py::class_<pycauset::MatrixBase, std::shared_ptr<pycauset::MatrixBase>>(m, "MatrixBase");
    py::class_<pycauset::VectorBase, std::shared_ptr<pycauset::VectorBase>>(m, "VectorBase");

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
        .def("can_fit_in_ram", &pycauset::core::MemoryGovernor::can_fit_in_ram, py::arg("bytes"));
}
