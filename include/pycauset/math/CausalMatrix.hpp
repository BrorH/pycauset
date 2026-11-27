// Causal-matrix kernels on the packed bits (a causal matrix is TriangularMatrix<bool>,
// strictly upper triangular).
#pragma once

#include <memory>

#include "pycauset/matrix/TriangularBitMatrix.hpp"

namespace pycauset {

// Link matrix (transitive reduction) of a transitively closed causal matrix: (i, j) is a link
// iff C_ij and no k with C_ik and C_kj. Bit kernel, ~(#links) * n/64 word ORs per row.
std::unique_ptr<MatrixBase> transitive_reduction(const TriangularMatrix<bool>& c);

}  // namespace pycauset
