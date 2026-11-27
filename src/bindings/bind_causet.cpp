#include "bindings_common.hpp"

#include "pycauset/causet/Spacetime.hpp"
#include "pycauset/causet/Sprinkler.hpp"
#include "pycauset/matrix/TriangularBitMatrix.hpp"
#include "pycauset/core/ParallelUtils.hpp"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace pycauset;

// Poll Python's pending-signal flag (raises KeyboardInterrupt when SIGINT was
// received). Passed into the sprinkle loop so Ctrl+C reliably halts it.
static bool python_interrupt_pending() {
    return PyErr_CheckSignals() != 0;
}

static std::shared_ptr<TriangularBitMatrix> sprinkle_to_tbm(
    const CausalSpacetime& spacetime,
    uint64_t n,
    uint64_t seed
) {
    std::unique_ptr<MatrixBase> base;
    try {
        base = Sprinkler::sprinkle(spacetime, n, seed, "", &python_interrupt_pending);
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()) == "pycauset: interrupted") {
            // KeyboardInterrupt is already pending; propagate it.
            throw pybind11::error_already_set();
        }
        throw;
    }

    auto* tbm = dynamic_cast<TriangularBitMatrix*>(base.get());
    if (tbm == nullptr) {
        throw std::runtime_error("Sprinkler::sprinkle did not return a causal (bit triangular) matrix");
    }

    // Transfer ownership into shared_ptr<TriangularBitMatrix>
    std::unique_ptr<TriangularBitMatrix> typed(static_cast<TriangularBitMatrix*>(base.release()));
    return std::shared_ptr<TriangularBitMatrix>(typed.release());
}


// Causal-set sprinkling (R1: the Minkowski diamond in any dimension). The rest of the
// spacetime library is parked in experimental/physics.
void bind_causet_classes(py::module_& m) {
    py::class_<CausalSpacetime, std::shared_ptr<CausalSpacetime>>(m, "_CausalSpacetime");

    py::class_<MinkowskiDiamond, CausalSpacetime, std::shared_ptr<MinkowskiDiamond>>(m, "_MinkowskiDiamond")
        .def(py::init<int>(), py::arg("dimension"))
        .def("dimension", &MinkowskiDiamond::dimension)
        .def("volume", &MinkowskiDiamond::volume)
        .def("causality", &MinkowskiDiamond::causality, py::arg("u"), py::arg("v"),
             "True when u is in the causal past of v.");

    m.def(
        "_sprinkle",
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
        "_make_coordinates",
        [](const std::shared_ptr<CausalSpacetime>& spacetime,
           uint64_t n,
           uint64_t seed,
           const std::vector<uint64_t>& indices) {
            if (!spacetime) {
                throw std::invalid_argument("spacetime must not be None");
            }
            try {
                return Sprinkler::make_coordinates(*spacetime, n, seed, indices, &python_interrupt_pending);
            } catch (const std::runtime_error& e) {
                if (std::string(e.what()) == "pycauset: interrupted") {
                    throw pybind11::error_already_set();
                }
                throw;
            }
        },
        py::arg("spacetime"),
        py::arg("n"),
        py::arg("seed"),
        py::arg("indices"));
}
