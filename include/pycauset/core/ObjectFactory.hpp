#pragma once

#include <memory>
#include <string>
#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/vector/VectorBase.hpp"
#include "pycauset/core/StorageUtils.hpp"

namespace pycauset {

class ObjectFactory {
public:
    // --- Matrix Creation ---
    
    // Creates a new matrix with the specified dimensions and types
    static std::unique_ptr<MatrixBase> create_matrix(
        uint64_t n, 
        DataType dtype, 
        MatrixType mtype, 
        const std::string& backing_file = ""
    );

    // Loads an existing matrix with explicit metadata
    static std::unique_ptr<MatrixBase> load_matrix(
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

    // --- Vector Creation ---

    static std::unique_ptr<VectorBase> create_vector(
        uint64_t n, 
        DataType dtype, 
        MatrixType mtype, 
        const std::string& backing_file = ""
    );

    static std::unique_ptr<VectorBase> load_vector(
        const std::string& backing_file,
        size_t offset,
        uint64_t rows, // Vectors are rows x 1 (or 1 x cols if transposed)
        uint64_t cols,
        DataType dtype,
        MatrixType mtype,
        uint64_t seed,
        double scalar,
        bool is_transposed
    );

    // --- Type Resolution ---

    // Resolves the result data type for binary operations
    // e.g., INT32 + INT32 -> INT32, INT32 + FLOAT64 -> FLOAT64
    static DataType resolve_result_type(DataType a, DataType b);

    // Resolves the result matrix type (structure)
    // e.g., TRIANGULAR + TRIANGULAR -> TRIANGULAR, DENSE + TRIANGULAR -> DENSE
    static MatrixType resolve_result_matrix_type(MatrixType a, MatrixType b);
};

}
