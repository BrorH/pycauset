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
        std::complex<double> scalar = 1.0,
        bool is_transposed = false
    );

    // Clones a matrix (Lazy Copy / CoW) sharing the same storage
    static std::unique_ptr<MatrixBase> clone_matrix(
        std::shared_ptr<MemoryMapper> mapper,
        uint64_t rows,
        uint64_t cols,
        DataType dtype,
        MatrixType mtype,
        uint64_t seed = 0,
        std::complex<double> scalar = 1.0,
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
        std::complex<double> scalar,
        bool is_transposed
    );

    static std::unique_ptr<VectorBase> clone_vector(
        std::shared_ptr<MemoryMapper> mapper,
        uint64_t rows,
        uint64_t cols,
        DataType dtype,
        MatrixType mtype,
        uint64_t seed = 0,
        std::complex<double> scalar = 1.0,
        bool is_transposed = false
    );
};

}
