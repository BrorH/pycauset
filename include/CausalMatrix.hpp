#pragma once

#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include <optional>

#include "TriangularMatrix.hpp"

// Forward declaration
class IntegerMatrix;

class CausalMatrix : public TriangularMatrix {
public:
    // Constructor: Creates a new matrix of size n*n
    // If backingFile is provided, it uses disk storage.
    CausalMatrix(uint64_t n, const std::string& backingFile = "", bool populate = false,
                 std::optional<uint64_t> seed = std::nullopt);

    // Constructor for loading from existing mapper
    CausalMatrix(uint64_t n, std::unique_ptr<MemoryMapper> mapper);

    // Set bit at (i, j). Only valid for i < j.
    void set(uint64_t i, uint64_t j, bool value);

    // Get bit at (i, j). Returns 0 if i >= j.
    bool get(uint64_t i, uint64_t j) const;

    // Standard Matrix Multiplication: Result = this * other
    // Returns an IntegerMatrix (strictly upper triangular)
    // Result[i][j] = sum(this[i][k] * other[k][j])
    std::unique_ptr<IntegerMatrix> multiply(const CausalMatrix& other, const std::string& result_file) const;
    // Logical AND between two boolean matrices; mirrors numpy.multiply for bool arrays.
    std::unique_ptr<CausalMatrix> elementwise_multiply(const CausalMatrix& other, const std::string& result_file) const;

    // Get the raw data pointer (useful for bulk operations)
    const uint64_t* data() const { return static_cast<const uint64_t*>(require_mapper()->get_data()); }
    uint64_t* data() { return static_cast<uint64_t*>(require_mapper()->get_data()); }

    // Factory method for random matrices
    static std::unique_ptr<CausalMatrix> random(uint64_t n, double density,
                                                const std::string& backing_file = "",
                                                std::optional<uint64_t> seed = std::nullopt);

private:
    void fill_random(double density, std::optional<uint64_t> seed = std::nullopt);
};
