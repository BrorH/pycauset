#pragma once

#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include <optional>

#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"

namespace pycauset {

// Forward declarations
// using IntegerMatrix = DenseMatrix<int32_t>; // We can not forward declare typedefs easily without including
// using TriangularFloatMatrix = TriangularMatrix<double>;

template <>
class TriangularMatrix<bool> : public TriangularMatrixBase {
public:
    // Bit-packed upper triangle: strictly upper (a causal matrix) or, with has_diagonal, upper
    // including the diagonal (e.g. the reflexive closure C + I). Row i holds the bits of
    // columns first_col(i) .. n-1, padded to 64-bit words.
    TriangularMatrix(uint64_t n, const std::string& backingFile = "", bool has_diagonal = false);

    // Constructor for loading with explicit metadata
    TriangularMatrix(uint64_t n, 
                     const std::string& backing_file,
                     size_t offset,
                     uint64_t seed,
                     std::complex<double> scalar,
                     bool is_transposed,
                     bool has_diagonal = false);

    // Constructor for loading from existing mapper
    TriangularMatrix(uint64_t n, std::shared_ptr<MemoryMapper> mapper, bool has_diagonal = false);

    // First stored column of row i, and the number of stored bits in it.
    uint64_t first_col(uint64_t i) const { return has_diagonal_ ? i : i + 1; }
    uint64_t row_bits(uint64_t i) const {
        const uint64_t n = base_rows();
        return first_col(i) < n ? n - first_col(i) : 0;
    }

    // Set bit at (i, j); only stored positions (j > i, or j >= i with a diagonal).
    void set(uint64_t i, uint64_t j, bool value);

    // Get bit at (i, j); 0 outside the stored triangle.
    bool get(uint64_t i, uint64_t j) const;
    
    // Generic accessor
    double element_as_double(uint64_t i, uint64_t j) const override;

    // Generic setter
    void set_element_double(uint64_t i, uint64_t j, double value) override {
        set(i, j, value != 0.0);
    }

    // Path counts: strictly upper x strictly upper -> strictly upper int32 (word-level popcounts).
    // Both operands must be strict (dispatch_matmul's bit rule checks; others go dense).
    std::unique_ptr<TriangularMatrix<int32_t>> multiply(const TriangularMatrix<bool>& other, const std::string& result_file) const;

    // Scalar multiplication
    std::unique_ptr<MatrixBase> multiply_scalar(double factor, const std::string& result_file = "") const override;

    std::unique_ptr<MatrixBase> multiply_scalar(int64_t factor, const std::string& result_file = "") const override {
        return multiply_scalar(static_cast<double>(factor), result_file);
    }


    std::unique_ptr<MatrixBase> transpose(const std::string& result_file = "") const override;

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

} // namespace pycauset
