#pragma once

#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include <optional>

#include "TriangularMatrix.hpp"
#include "DenseMatrix.hpp"

// Forward declarations
// using IntegerMatrix = DenseMatrix<int32_t>; // We can't forward declare typedefs easily without including
// using TriangularFloatMatrix = TriangularMatrix<double>;

template <>
class TriangularMatrix<bool> : public TriangularMatrixBase {
public:
    // Constructor: Creates a new matrix of size n*n
    // If backingFile is provided, it uses disk storage.
    TriangularMatrix(uint64_t n, const std::string& backingFile = "");

    // Constructor for loading with explicit metadata
    TriangularMatrix(uint64_t n, 
                     const std::string& backing_file,
                     size_t offset,
                     uint64_t seed,
                     double scalar,
                     bool is_transposed);

    // Constructor for loading from existing mapper
    TriangularMatrix(uint64_t n, std::unique_ptr<MemoryMapper> mapper);

    // Set bit at (i, j). Only valid for i < j.
    void set(uint64_t i, uint64_t j, bool value);

    // Get bit at (i, j). Returns 0 if i >= j.
    bool get(uint64_t i, uint64_t j) const;
    
    // Generic accessor
    double get_element_as_double(uint64_t i, uint64_t j) const override;

    // Standard Matrix Multiplication: Result = this * other
    // Returns a TriangularMatrix<int32_t> (strictly upper triangular)
    std::unique_ptr<TriangularMatrix<int32_t>> multiply(const TriangularMatrix<bool>& other, const std::string& result_file) const;
    
    // Logical AND between two boolean matrices; mirrors numpy.multiply for bool arrays.
    std::unique_ptr<TriangularMatrix<bool>> elementwise_multiply(const TriangularMatrix<bool>& other, const std::string& result_file) const;
    
    // Scalar multiplication
    std::unique_ptr<MatrixBase> multiply_scalar(double factor, const std::string& result_file = "") const override;

    std::unique_ptr<MatrixBase> multiply_scalar(int64_t factor, const std::string& result_file = "") const override {
        return multiply_scalar(static_cast<double>(factor), result_file);
    }

    std::unique_ptr<MatrixBase> transpose(const std::string& result_file = "") const override;

    // Bitwise inversion (NOT)
    // Returns a new matrix where every bit in the upper triangle is flipped.
    std::unique_ptr<TriangularMatrix<bool>> bitwise_not(const std::string& result_file = "") const;

    // Matrix Inversion (Linear Algebra)
    // Throws exception as strictly upper triangular matrices are singular.
    std::unique_ptr<TriangularMatrix<double>> inverse(const std::string& result_file = "") const;

    // Get the raw data pointer (useful for bulk operations)
    const uint64_t* data() const { return static_cast<const uint64_t*>(require_mapper()->get_data()); }
    uint64_t* data() { return static_cast<uint64_t*>(require_mapper()->get_data()); }

    // Factory method for random matrices
    static std::unique_ptr<TriangularMatrix<bool>> random(uint64_t n, double density,
                                                const std::string& backing_file = "",
                                                std::optional<uint64_t> seed = std::nullopt);

private:
    void fill_random(double density, std::optional<uint64_t> seed = std::nullopt);
};

using TriangularBitMatrix = TriangularMatrix<bool>;
