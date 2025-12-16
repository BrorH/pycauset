#include "bindings_common.hpp"

#include "pycauset/causet/Spacetime.hpp"
#include "pycauset/causet/Sprinkler.hpp"
#include "pycauset/matrix/TriangularBitMatrix.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace pycauset;

static std::shared_ptr<TriangularBitMatrix> sprinkle_to_tbm(
    const CausalSpacetime& spacetime,
    uint64_t n,
    uint64_t seed
) {
    auto base = Sprinkler::sprinkle(spacetime, n, seed, "");
    auto* tbm = dynamic_cast<TriangularBitMatrix*>(base.get());
    if (tbm == nullptr) {
        throw std::runtime_error("Sprinkler::sprinkle did not return a TriangularBitMatrix");
    }

    // Transfer ownership into shared_ptr<TriangularBitMatrix>
    std::unique_ptr<TriangularBitMatrix> typed(static_cast<TriangularBitMatrix*>(base.release()));
    return std::shared_ptr<TriangularBitMatrix>(typed.release());
}

void bind_causet_classes(py::module_& m) {
    py::class_<CausalSpacetime, std::shared_ptr<CausalSpacetime>>(m, "CausalSpacetime");

    py::class_<MinkowskiDiamond, CausalSpacetime, std::shared_ptr<MinkowskiDiamond>>(
        m, "MinkowskiDiamond")
        .def(py::init<int>(), py::arg("dimension"))
        .def("dimension", &MinkowskiDiamond::dimension)
        .def("volume", &MinkowskiDiamond::volume);

    py::class_<MinkowskiCylinder, CausalSpacetime, std::shared_ptr<MinkowskiCylinder>>(
        m, "MinkowskiCylinder")
        .def(py::init<int, double, double>(), py::arg("dimension"), py::arg("height"), py::arg("circumference"))
        .def("dimension", &MinkowskiCylinder::dimension)
        .def("volume", &MinkowskiCylinder::volume)
        .def_property_readonly("height", &MinkowskiCylinder::get_height)
        .def_property_readonly("circumference", &MinkowskiCylinder::get_circumference);

    py::class_<MinkowskiBox, CausalSpacetime, std::shared_ptr<MinkowskiBox>>(
        m, "MinkowskiBox")
        .def(
            py::init<int, double, double>(),
            py::arg("dimension"),
            py::arg("time_extent"),
            py::arg("space_extent"))
        .def("dimension", &MinkowskiBox::dimension)
        .def("volume", &MinkowskiBox::volume)
        .def_property_readonly("time_extent", &MinkowskiBox::get_time_extent)
        .def_property_readonly("space_extent", &MinkowskiBox::get_space_extent);

    m.def(
        "sprinkle",
        [](const std::shared_ptr<CausalSpacetime>& spacetime, uint64_t n, uint64_t seed) {
            if (!spacetime) {
                throw std::invalid_argument("spacetime must not be None");
            }
            return sprinkle_to_tbm(*spacetime, n, seed);
        },
        py::arg("spacetime"),
        py::arg("n"),
        py::arg("seed"));

    m.def(
        "make_coordinates",
        [](const std::shared_ptr<CausalSpacetime>& spacetime,
           uint64_t n,
           uint64_t seed,
           const std::vector<uint64_t>& indices) {
            if (!spacetime) {
                throw std::invalid_argument("spacetime must not be None");
            }
            return Sprinkler::make_coordinates(*spacetime, n, seed, indices);
        },
        py::arg("spacetime"),
        py::arg("n"),
        py::arg("seed"),
        py::arg("indices"));
}
