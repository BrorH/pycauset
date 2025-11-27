#pragma once

#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include "MemoryMapper.hpp"

// Forward declaration
class IntegerMatrix;

class CausalMatrix {
public:
    // Constructor: Creates a new matrix of size N*N
    // If backingFile is provided, it uses disk storage.
    CausalMatrix(uint64_t N, const std::string& backingFile = "");
    
    // Destructor
    ~CausalMatrix() = default;

    // Set bit at (i, j). Only valid for i < j.
    void set(uint64_t i, uint64_t j, bool value);

    // Get bit at (i, j). Returns 0 if i >= j.
    bool get(uint64_t i, uint64_t j) const;

    // Standard Matrix Multiplication: Result = this * other
    // Returns an IntegerMatrix (strictly upper triangular)
    // Result[i][j] = sum(this[i][k] * other[k][j])
    std::unique_ptr<IntegerMatrix> multiply(const CausalMatrix& other, const std::string& resultFile) const;

    // Get the dimension N
    uint64_t size() const { return N_; }

    // Get the raw data pointer (useful for bulk operations)
    const uint64_t* data() const { return static_cast<const uint64_t*>(mapper_->getData()); }
    uint64_t* data() { return static_cast<uint64_t*>(mapper_->getData()); }

    // Helper to get the byte offset for a specific row
    uint64_t getRowOffset(uint64_t i) const { return row_offsets_[i]; }

    // Factory method for random matrices
    static std::unique_ptr<CausalMatrix> random(uint64_t N, double density, const std::string& backingFile = "");

private:
    uint64_t N_;
    std::unique_ptr<MemoryMapper> mapper_;
    std::vector<uint64_t> row_offsets_; // Byte offsets for each row
    
    void calculateOffsets();
};
