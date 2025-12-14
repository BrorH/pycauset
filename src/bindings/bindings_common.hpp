#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

namespace py = pybind11;

void bind_core_classes(py::module_& m);
void bind_matrix_classes(py::module_& m);
void bind_vector_classes(py::module_& m);
void bind_complex_classes(py::module_& m);
void bind_causet_classes(py::module_& m);
