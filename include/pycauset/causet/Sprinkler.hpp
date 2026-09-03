#pragma once
#ifndef PYCAUSET_SPRINKLER_HPP
#define PYCAUSET_SPRINKLER_HPP

#include "pycauset/causet/Spacetime.hpp"
#include "pycauset/matrix/MatrixBase.hpp"
#include <functional>
#include <memory>
#include <string>

namespace pycauset {

class Sprinkler {
public:
    // Optional callback polled periodically inside the long O(n^2) sprinkle
    // loop. It must be cheap and is called with no lock held. Returning `true`
    // aborts the sprinkle by throwing `std::runtime_error` (the Python binding
    // translates this into `KeyboardInterrupt` so Ctrl+C actually halts).
    using AbortCheck = std::function<bool()>;

    static std::unique_ptr<MatrixBase> sprinkle(
        const CausalSpacetime& spacetime,
        uint64_t n,
        uint64_t seed,
        const std::string& saveas = "",
        const AbortCheck& should_abort = {}
    );

    static std::vector<std::vector<double>> make_coordinates(
        const CausalSpacetime& spacetime,
        uint64_t n,
        uint64_t seed,
        std::vector<uint64_t> indices,
        const AbortCheck& should_abort = {}
    );
};


}

#endif // PYCAUSET_SPRINKLER_HPP
