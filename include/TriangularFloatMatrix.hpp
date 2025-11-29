#pragma once

#include "TriangularMatrix.hpp"
#include <string>

class TriangularFloatMatrix : public TriangularMatrix {
public:
    TriangularFloatMatrix(uint64_t n, const std::string& backingFile = "");
    
    // Constructor for loading
    TriangularFloatMatrix(uint64_t n, std::unique_ptr<MemoryMapper> mapper);

    // Set value at (i, j). Only valid for i < j.
    void set(uint64_t i, uint64_t j, double value);
    
    // Get value at (i, j). Returns 0.0 if i >= j.
    double get(uint64_t i, uint64_t j) const;

    // Get raw data pointer
    double* data() { return static_cast<double*>(require_mapper()->get_data()); }
    const double* data() const { return static_cast<const double*>(require_mapper()->get_data()); }
};
