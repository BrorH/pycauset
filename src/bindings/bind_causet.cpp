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

namespace {
// Link matrix (transitive reduction) of a TRANSITIVELY CLOSED causal matrix C, on the bits
// (F-053; LAZY.md L7.4). Row i: walk future(i) in increasing order; j is a link iff it is not
// yet covered by the futures of the links found so far (transitivity makes that equivalent to
// "no k with i < k < j, C_ik and C_kj"). Cost per row ~ (#links) * n/64 word ORs instead of
// an n^3 product; no n^2 integer counts. Rows start on 64-bit boundaries, so rows are
// independent and processed in parallel.
std::shared_ptr<pycauset::MatrixBase> transitive_reduction(const pycauset::TriangularMatrix<bool>& c) {
    const uint64_t n = c.rows();
    auto out = std::make_shared<pycauset::TriangularMatrix<bool>>(n, "");
    if (n < 2) return out;
    const auto* base = static_cast<const char*>(c.shared_mapper()->get_data());
    auto* obase = static_cast<char*>(out->shared_mapper()->get_data());
    const uint64_t full_words = (n + 63) / 64 + 1;
    pycauset::ParallelFor(0, n - 1, [&](size_t i) {
        const auto* row = reinterpret_cast<const uint64_t*>(base + c.get_row_offset(i));
        auto* orow = reinterpret_cast<uint64_t*>(obase + out->get_row_offset(i));
        const uint64_t nbits = n - 1 - i;
        std::vector<uint64_t> covered(full_words, 0);  // absolute column bitset
        for (uint64_t q = 0; q * 64 < nbits; ++q) {
            uint64_t bits = row[q];
            const uint64_t rem = nbits - q * 64;
            if (rem < 64) bits &= (uint64_t{1} << rem) - 1;
            while (bits) {
                const uint64_t t = static_cast<uint64_t>(__builtin_ctzll(bits));
                bits &= bits - 1;
                const uint64_t j = i + 1 + q * 64 + t;
                if ((covered[j >> 6] >> (j & 63)) & 1ULL) continue;
                orow[q] |= uint64_t{1} << t;  // (i, j) is a link
                // covered |= future(j): row j bit b is column j + 1 + b.
                const uint64_t jb = n - 1 - j;
                if (jb == 0) continue;
                const auto* rj = reinterpret_cast<const uint64_t*>(base + c.get_row_offset(j));
                const uint64_t s = j + 1, sw = s >> 6, sh = s & 63;
                for (uint64_t p = 0; p * 64 < jb; ++p) {
                    uint64_t w = rj[p];
                    const uint64_t r2 = jb - p * 64;
                    if (r2 < 64) w &= (uint64_t{1} << r2) - 1;
                    covered[sw + p] |= w << sh;
                    if (sh != 0) covered[sw + p + 1] |= w >> (64 - sh);
                }
            }
        }
    });
    return out;
}
}  // namespace

void bind_causet_classes(py::module_& m) {
    m.def("_transitive_reduction", [](const std::shared_ptr<pycauset::MatrixBase>& c) {
        const auto* tri = dynamic_cast<const pycauset::TriangularMatrix<bool>*>(c.get());
        if (tri == nullptr || c->is_transposed() || c->get_scalar() != std::complex<double>(1.0, 0.0)) {
            throw py::type_error("_transitive_reduction expects a causal matrix (TriangularMatrix of dtype 'bool')");
        }
        return transitive_reduction(*tri);
    }, py::arg("C"), "Internal: link matrix of a transitively closed causal matrix (bit kernel).");
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
