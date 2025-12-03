#pragma once

#include "FileFormat.hpp"
#include "VectorBase.hpp"
#include <memory>
#include <string>

namespace pycauset {

class VectorFactory {
public:
    static std::unique_ptr<VectorBase> create(
        uint64_t n, 
        DataType dtype, 
        MatrixType mtype, 
        const std::string& backing_file = ""
    );

    static std::unique_ptr<VectorBase> load(
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
};

}
