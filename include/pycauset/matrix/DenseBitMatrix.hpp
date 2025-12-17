#pragma once

#include "pycauset/matrix/DenseMatrix.hpp"
#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include <optional>

namespace pycauset {

template <>
class DenseMatrix<bool> : public MatrixBase {
public:
    // Constructors: create a new dense bit-packed matrix.
    // If backingFile is provided, it uses disk storage.
    DenseMatrix(uint64_t n, const std::string& backingFile = "");
    DenseMatrix(uint64_t rows, uint64_t cols, const std::string& backingFile = "");

    // Constructor for loading with explicit metadata
    DenseMatrix(uint64_t n, 
                const std::string& backing_file,
                size_t offset,
                uint64_t seed,
                std::complex<double> scalar,
                bool is_transposed);

    DenseMatrix(uint64_t rows,
                uint64_t cols,
                const std::string& backing_file,
                size_t offset,
                uint64_t seed,
                std::complex<double> scalar,
                bool is_transposed);

    // Constructor for loading from existing mapper
    DenseMatrix(uint64_t n, std::shared_ptr<MemoryMapper> mapper);
    DenseMatrix(uint64_t rows, uint64_t cols, std::shared_ptr<MemoryMapper> mapper);

    // Set bit at (i, j).
    void set(uint64_t i, uint64_t j, bool value);

    // Get bit at (i, j).
    bool get(uint64_t i, uint64_t j) const;
    
    // Generic accessor
    double get_element_as_double(uint64_t i, uint64_t j) const override;

    // Standard Matrix Multiplication: Result = this * other
    // Returns a DenseMatrix<int32_t>
    std::unique_ptr<DenseMatrix<int32_t>> multiply(const DenseMatrix<bool>& other, const std::string& result_file) const;
    
    // Logical AND between two boolean matrices
    std::unique_ptr<DenseMatrix<bool>> elementwise_multiply(const DenseMatrix<bool>& other, const std::string& result_file) const;
    
    // Scalar multiplication
    std::unique_ptr<MatrixBase> multiply_scalar(double factor, const std::string& result_file = "") const override;

    std::unique_ptr<MatrixBase> multiply_scalar(int64_t factor, const std::string& result_file = "") const override {
        return multiply_scalar(static_cast<double>(factor), result_file);
    }

    std::unique_ptr<MatrixBase> add_scalar(double scalar, const std::string& result_file = "") const override;
    std::unique_ptr<MatrixBase> add_scalar(int64_t scalar, const std::string& result_file = "") const override;

    std::unique_ptr<MatrixBase> transpose(const std::string& result_file = "") const override;

    // Bitwise inversion (NOT)
    std::unique_ptr<DenseMatrix<bool>> bitwise_not(const std::string& result_file = "") const;

    // Matrix Inversion (Linear Algebra)
    // Throws exception as boolean matrices are not invertible in the standard sense here.
    std::unique_ptr<DenseMatrix<double>> inverse(const std::string& result_file = "") const;

    uint64_t stride_bytes() const { return stride_bytes_; }

    // Get the raw data pointer (useful for bulk operations)
    const uint64_t* data() const { return static_cast<const uint64_t*>(require_mapper()->get_data()); }
    uint64_t* data() { return static_cast<uint64_t*>(require_mapper()->get_data()); }

    // Fill all logical entries with a value.
    void fill(bool value);

    // Factory method for random matrices
    static std::unique_ptr<DenseMatrix<bool>> random(uint64_t n, double density,
                                                const std::string& backing_file = "",
                                                std::optional<uint64_t> seed = std::nullopt);

    static std::unique_ptr<DenseMatrix<bool>> random(uint64_t rows, uint64_t cols, double density,
                                                const std::string& backing_file = "",
                                                std::optional<uint64_t> seed = std::nullopt);

private:
    void fill_random(double density, std::optional<uint64_t> seed = std::nullopt);
    void calculate_stride();
    uint64_t stride_bytes_ = 0;
};

using DenseBitMatrix = DenseMatrix<bool>;

} // namespace pycauset
