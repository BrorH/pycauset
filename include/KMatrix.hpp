#pragma once

#include "CausalMatrix.hpp"
#include "MatrixBase.hpp"
#include <string>

class FloatMatrix : public MatrixBase {
public:
    FloatMatrix(uint64_t n, const std::string& backingFile = "");

    // Constructor for loading
    FloatMatrix(uint64_t n, std::unique_ptr<MemoryMapper> mapper);
    
    // Set value at (i, j)
    void set(uint64_t i, uint64_t j, double value);
    
    // Get value at (i, j)
    double get(uint64_t i, uint64_t j) const;

    // Get raw data pointer
    double* data() { return static_cast<double*>(require_mapper()->get_data()); }
    const double* data() const { return static_cast<const double*>(require_mapper()->get_data()); }
};

void compute_k_matrix(
    const CausalMatrix& C, 
    double a, 
    const std::string& output_path, 
    int num_threads
);
