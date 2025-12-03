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

    // Loads an existing matrix with explicit metadata
    static std::unique_ptr<MatrixBase> load(
        const std::string& backing_file,
        size_t offset,
        uint64_t rows,
        uint64_t cols,
        DataType dtype,
        MatrixType mtype,
        uint64_t seed = 0,
        double scalar = 1.0,
        bool is_transposed = false
    );

    // Resolves the result data type for binary operations
    // e.g., INT32 + INT32 -> INT32, INT32 + FLOAT64 -> FLOAT64
    static DataType resolve_result_type(DataType a, DataType b);

    // Resolves the result matrix type (structure)
    // e.g., TRIANGULAR + TRIANGULAR -> TRIANGULAR, DENSE + TRIANGULAR -> DENSE
    static MatrixType resolve_result_matrix_type(MatrixType a, MatrixType b);
};

}
