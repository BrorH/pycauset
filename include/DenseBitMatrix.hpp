#pragma once

#include "DenseMatrix.hpp"
#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include <optional>

template <>
class DenseMatrix<bool> : public MatrixBase {
public:
    // Constructor: Creates a new matrix of size n*n
    // If backingFile is provided, it uses disk storage.
    DenseMatrix(uint64_t n, const std::string& backingFile = "");

    // Constructor for loading from existing mapper
    DenseMatrix(uint64_t n, std::unique_ptr<MemoryMapper> mapper);

    // Set bit at (i, j).
    void set(uint64_t i, uint64_t j, bool value);

    // Get bit at (i, j).
    bool get(uint64_t i, uint64_t j) const;
    
    // Generic accessor
    double get_element_as_double(uint64_t i, uint64_t j) const override;

    // Standard Matrix Multiplication: Result = this * other
    // Returns a DenseMatrix<int32_t> (integer result)
    // Note: We return int32 because boolean matmul is essentially counting paths.
    // If we want boolean semiring multiplication, that's different.
    // Standard matmul over Z_2 or Z? Usually Z for "counting".
    // But if it's "BitMatrix", maybe we want boolean semiring (OR-AND)?
    // NumPy matmul on bools does integer multiplication (0/1) and sums them, returning integers (or floats).
    // So returning IntegerMatrix is correct for standard matmul.
    std::unique_ptr<DenseMatrix<int32_t>> multiply(const DenseMatrix<bool>& other, const std::string& result_file) const;
    
    // Scalar multiplication
    std::unique_ptr<MatrixBase> multiply_scalar(double factor, const std::string& result_file = "") const override;
    
    std::unique_ptr<MatrixBase> multiply_scalar(int64_t factor, const std::string& result_file = "") const override {
        return multiply_scalar(static_cast<double>(factor), result_file);
    }

    // Bitwise inversion (NOT)
    std::unique_ptr<DenseMatrix<bool>> bitwise_not(const std::string& result_file = "") const;

    // Get the raw data pointer
    const uint64_t* data() const { return static_cast<const uint64_t*>(require_mapper()->get_data()); }
    uint64_t* data() { return static_cast<uint64_t*>(require_mapper()->get_data()); }

    // Factory method for random matrices
    static std::unique_ptr<DenseMatrix<bool>> random(uint64_t n, double density,
                                                const std::string& backing_file = "",
                                                std::optional<uint64_t> seed = std::nullopt);

    uint64_t stride_bytes() const { return stride_bytes_; }

private:
    uint64_t stride_bytes_; // Bytes per row (padded to 64-bit boundary)
    
    void calculate_stride();
    void fill_random(double density, std::optional<uint64_t> seed = std::nullopt);
};

using DenseBitMatrix = DenseMatrix<bool>;
