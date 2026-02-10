#include "bindings/bindings_common.hpp"

#include "pycauset/core/MemoryGovernor.hpp"

PYBIND11_MODULE(_pycauset, m) {
    m.doc() = "PyCauset: High-performance Causal Set Theory library";

    bind_core_classes(m);
    bind_op_registry(m);
    bind_matrix_classes(m);
    bind_expression_classes(m);
    bind_vector_classes(m);
    bind_causet_classes(m);
    bind_math_ops(m);

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
