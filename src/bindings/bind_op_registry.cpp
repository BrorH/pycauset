#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pycauset/core/OpRegistry.hpp"

namespace py = pybind11;

void bind_op_registry(py::module& m) {
    using namespace pycauset;

    py::class_<OpContract>(m, "OpContract")
        .def(py::init<>())
        .def_readwrite("name", &OpContract::name)
        .def_readwrite("supports_streaming", &OpContract::supports_streaming)
        .def_readwrite("supports_block_matrix", &OpContract::supports_block_matrix)
        .def_readwrite("requires_square", &OpContract::requires_square)
        .def("__repr__", [](const OpContract& c) {
            return "<OpContract name='" + c.name + "'>";
        });

    py::class_<OpRegistry>(m, "OpRegistry")
        .def_static("instance", &OpRegistry::instance, py::return_value_policy::reference)
        .def("get_contract", &OpRegistry::get_contract, py::return_value_policy::reference_internal);
}
