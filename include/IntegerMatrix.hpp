#pragma once

#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include "MemoryMapper.hpp"

class IntegerMatrix {
public:
    // Constructor: Creates a new matrix of size N*N
    // Stores 32-bit integers. Strictly Upper Triangular.
    IntegerMatrix(uint64_t N, const std::string& backingFile);
    
    ~IntegerMatrix() = default;

    // Set value at (i, j). Only valid for i < j.
    void set(uint64_t i, uint64_t j, uint32_t value);

    // Get value at (i, j). Returns 0 if i >= j.
    uint32_t get(uint64_t i, uint64_t j) const;

    uint64_t size() const { return N_; }

    const uint32_t* data() const { return static_cast<const uint32_t*>(mapper_->getData()); }
    uint32_t* data() { return static_cast<uint32_t*>(mapper_->getData()); }

private:
    uint64_t N_;
    std::unique_ptr<MemoryMapper> mapper_;
    std::vector<uint64_t> row_offsets_;
    
    void calculateOffsets();
};
