#pragma once
#ifndef PYCAUSET_SPRINKLER_HPP
#define PYCAUSET_SPRINKLER_HPP

#include "Spacetime.hpp"
#include "MatrixBase.hpp"
#include <memory>
#include <string>

namespace pycauset {

class Sprinkler {
public:
    static std::unique_ptr<MatrixBase> sprinkle(
        const CausalSpacetime& spacetime, 
        uint64_t n, 
        uint64_t seed,
        const std::string& saveas = ""
    );
};

}

#endif // PYCAUSET_SPRINKLER_HPP
