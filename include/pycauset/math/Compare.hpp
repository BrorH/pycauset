#pragma once

#include <memory>

#include "pycauset/matrix/MatrixBase.hpp"

namespace pycauset {

enum class CompareOp { Eq, Ne, Lt, Le, Gt, Ge };

// Element-wise a == b, a != b, a < b, a <= b, a > b, a >= b for any dtypes, structures and views,
// with NumPy's broadcasting (equal axes or 1): a dense bool matrix. Values compare as NumPy does:
// integers exactly (signed with unsigned too), an integer with a float in float64, complex
// lexicographically (NumPy 2), NaN unequal to everything (`!=` true). Comparisons were missing:
// `M == M2` was Python's identity test (review F-128, 3.6 C-7).
std::unique_ptr<MatrixBase> compare(const MatrixBase& a, const MatrixBase& b, CompareOp op);

}  // namespace pycauset
