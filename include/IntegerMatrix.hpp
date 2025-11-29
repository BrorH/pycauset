#pragma once

#include <cstdint>
#include <string>
#include <memory>
#include <vector>

#include "MatrixBase.hpp"

class IntegerMatrix : public MatrixBase {
public:
    // Constructor: Creates a new matrix of size n*n
    // Stores 32-bit integers. Strictly Upper Triangular.
    IntegerMatrix(uint64_t n, const std::string& backingFile);

    // Constructor for loading from existing mapper
    IntegerMatrix(uint64_t n, std::unique_ptr<MemoryMapper> mapper);

    // Set value at (i, j). Only valid for i < j.
    void set(uint64_t i, uint64_t j, uint32_t value);

    // Get value at (i, j). Returns 0 if i >= j.
    uint32_t get(uint64_t i, uint64_t j) const;

    const uint32_t* data() const { return static_cast<const uint32_t*>(require_mapper()->get_data()); }
    uint32_t* data() { return static_cast<uint32_t*>(require_mapper()->get_data()); }

private:
    std::vector<uint64_t> row_offsets_;
    
    void calculate_offsets();
};
