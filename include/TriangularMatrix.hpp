#pragma once

#include "MatrixBase.hpp"
#include <vector>

class TriangularMatrix : public MatrixBase {
public:
    using MatrixBase::MatrixBase;

    // Helper to get the byte offset for a specific row
    uint64_t get_row_offset(uint64_t i) const { 
        if (i >= row_offsets_.size()) return 0; // Should not happen if initialized correctly
        return row_offsets_[i]; 
    }

protected:
    TriangularMatrix(uint64_t n, std::unique_ptr<MemoryMapper> mapper) 
        : MatrixBase(n, std::move(mapper)) {}

    std::vector<uint64_t> row_offsets_;

    // Calculate offsets for strictly upper triangular matrix
    // element_bits: size of one element in bits
    // alignment_bits: row alignment in bits (e.g., 64 for CausalMatrix)
    // Returns total size in bytes
    uint64_t calculate_triangular_offsets(uint64_t element_bits, uint64_t alignment_bits);
};
