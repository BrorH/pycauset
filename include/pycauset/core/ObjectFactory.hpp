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
    
    // Creates a new matrix with the specified dimensions and types (square)
    static std::unique_ptr<MatrixBase> create_matrix(
        uint64_t n, 
        DataType dtype, 
        Structure mtype, 
        const std::string& backing_file = ""
    );

    // Creates a new matrix with the specified dimensions and types (rectangular)
    static std::unique_ptr<MatrixBase> create_matrix(
        uint64_t rows,
        uint64_t cols,
        DataType dtype,
        Structure mtype,
        const std::string& backing_file = ""
    );

    // Clones a matrix (Lazy Copy / CoW) sharing the same storage
    static std::unique_ptr<MatrixBase> clone_matrix(
        std::shared_ptr<MemoryMapper> mapper,
        uint64_t rows,
        uint64_t cols,
        DataType dtype,
        Structure mtype,
        uint64_t seed = 0,
        std::complex<double> scalar = 1.0,
        bool is_transposed = false
    );

    // --- Vector Creation ---

    static std::unique_ptr<VectorBase> create_vector(
        uint64_t n, 
        DataType dtype, 
        Structure mtype, 
        const std::string& backing_file = ""
    );

    static std::unique_ptr<VectorBase> clone_vector(
        std::shared_ptr<MemoryMapper> mapper,
        uint64_t rows,
        uint64_t cols,
        DataType dtype,
        Structure mtype,
        uint64_t seed = 0,
        std::complex<double> scalar = 1.0,
        bool is_transposed = false
    );
};

}
