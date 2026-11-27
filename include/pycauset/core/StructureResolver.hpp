#pragma once

#include "pycauset/core/Types.hpp"
#include "pycauset/core/PromotionResolver.hpp"

namespace pycauset {
class MatrixBase;
}

namespace pycauset::structure {

// Result structure for an element-wise binary op on concrete operands (F-039). Unlike
// the type-only overload below, it knows orientation (a transposed triangle is a LOWER
// triangle), strictness (diagonal included or not) and which structured classes exist
// for `result_dtype` (F-032), and falls back to DENSE_FLOAT whenever it can't prove the
// structure holds. Used by the eager add / subtract / elementwise_multiply paths.
struct ResultStructure {
    Structure structure = Structure::DENSE;
    bool has_diagonal = false;  // TRIANGULAR: the result stores its diagonal
};
ResultStructure resolve_result_structure(promotion::BinaryOp op, const MatrixBase& a, const MatrixBase& b,
                                          DataType result_dtype);

} // namespace pycauset::structure
