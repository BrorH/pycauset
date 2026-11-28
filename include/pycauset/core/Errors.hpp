#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

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

// LAPACK's ?gebal / ?lascl reject NaN through xerbla, which prints to stderr and returns an
// "illegal argument" (review 3.4 C-9): the kernels that call them check first and raise with
// NumPy's message.
template <typename T>
void require_finite(const T* p, std::size_t n, const char* numpy_message) {
    for (std::size_t i = 0; i < n; ++i) {
        bool ok;
        if constexpr (std::is_arithmetic_v<T>) ok = std::isfinite(p[i]);
        else ok = std::isfinite(p[i].real()) && std::isfinite(p[i].imag());
        if (!ok) throw linalg_error(numpy_message);
    }
}

}  // namespace pycauset
