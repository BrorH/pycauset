#pragma once

#include <memory>
#include <string>
#include "MatrixBase.hpp"
#include "FileFormat.hpp"

namespace pycauset {

class MatrixFactory {
public:
    // Creates a new matrix with the specified dimensions and types
    static std::unique_ptr<MatrixBase> create(
        uint64_t n, 
        DataType dtype, 
        MatrixType mtype, 
        const std::string& backing_file = ""
    );

    // Resolves the result data type for binary operations
    // e.g., INT32 + INT32 -> INT32, INT32 + FLOAT64 -> FLOAT64
    static DataType resolve_result_type(DataType a, DataType b);

    // Resolves the result matrix type (structure)
    // e.g., TRIANGULAR + TRIANGULAR -> TRIANGULAR, DENSE + TRIANGULAR -> DENSE
    static MatrixType resolve_result_matrix_type(MatrixType a, MatrixType b);
};

}
