#pragma once

#include <stdexcept>

namespace pycauset {

// An operand of a dtype the operation does not take (a float operand of `&`, a complex matrix
// for a real-only solver): TypeError in Python, as NumPy raises. The translator is registered
// once in src/bindings.cpp. Not an std::invalid_argument, so the ValueError translators of
// the bindings do not catch it.
struct dtype_error : std::logic_error {
    using std::logic_error::logic_error;
};

// A matrix the factorisation cannot take (singular, not positive definite): NumPy's
// numpy.linalg.LinAlgError (a ValueError) in Python, translated in src/bindings.cpp.
struct linalg_error : std::domain_error {
    using std::domain_error::domain_error;
};

}  // namespace pycauset
