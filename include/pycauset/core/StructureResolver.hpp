#pragma once

#include "pycauset/core/Types.hpp"
#include "pycauset/core/PromotionResolver.hpp"

namespace pycauset {
class MatrixBase;
}

namespace pycauset::structure {

// Result structure for an element-wise binary op on concrete operands (F-039): it knows
// orientation (a transposed triangle is a LOWER triangle), strictness (diagonal included or
// not) and which structured classes exist for `result_dtype` (F-032), and gives DENSE
// whenever it cannot prove the structure holds (unequal shapes included: a 1 x 1 operand
// broadcasts). Used by the add / subtract / elementwise_multiply entry functions.
struct ResultStructure {
    Structure structure = Structure::DENSE;
    bool has_diagonal = false;  // TRIANGULAR: the result stores its diagonal
};
ResultStructure resolve_result_structure(promotion::BinaryOp op, const MatrixBase& a, const MatrixBase& b,
                                          DataType result_dtype);

} // namespace pycauset::structure
