#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <optional>
#include <vector>
#include <complex>
#include <iostream>
#include <algorithm>

#include "pycauset/core/PersistentObject.hpp"
#include "pycauset/matrix/expression/MatrixExpression.hpp"

namespace pycauset {

class MatrixBase : public PersistentObject {
public:
    // Constructor for creating new matrix (square)
    MatrixBase(uint64_t n,
               pycauset::MatrixType matrix_type,
               pycauset::DataType data_type);

    // Constructor for creating new matrix (rectangular)
    MatrixBase(uint64_t rows,
               uint64_t cols,
               pycauset::MatrixType matrix_type,
               pycauset::DataType data_type);

    // Constructor for loading/wrapping existing storage (square)
    MatrixBase(uint64_t n,
               std::shared_ptr<MemoryMapper> mapper,
               pycauset::MatrixType matrix_type,
               pycauset::DataType data_type,
               uint64_t seed = 0,
               std::complex<double> scalar = 1.0,
               bool is_transposed = false,
               bool is_temporary = false);

    // Constructor for loading/wrapping existing storage (rectangular)
    MatrixBase(uint64_t rows,
               uint64_t cols,
               std::shared_ptr<MemoryMapper> mapper,
               pycauset::MatrixType matrix_type,
               pycauset::DataType data_type,
               uint64_t seed = 0,
               std::complex<double> scalar = 1.0,
               bool is_transposed = false,
               bool is_temporary = false);

    virtual ~MatrixBase() {
    }

    std::unique_ptr<PersistentObject> clone() const override;

    // NumPy-aligned: size is total elements.
    uint64_t size() const { return rows() * cols(); }

    // Logical dimensions (transpose is a metadata view).
    uint64_t rows() const { return is_transposed() ? logical_cols_ : logical_rows_; }
    uint64_t cols() const { return is_transposed() ? logical_rows_ : logical_cols_; }

    // Base (storage) dimensions.
    uint64_t base_rows() const { return get_rows(); }
    uint64_t base_cols() const { return get_cols(); }

    uint64_t row_offset() const { return row_offset_; }
    uint64_t col_offset() const { return col_offset_; }

    uint64_t logical_rows() const { return logical_rows_; }
    uint64_t logical_cols() const { return logical_cols_; }

    std::shared_ptr<MemoryMapper> shared_mapper() const { return mapper_; }

    bool has_view_offset() const { return row_offset_ != 0 || col_offset_ != 0; }

    virtual double get_element_as_double(uint64_t i, uint64_t j) const = 0;
    
    virtual std::complex<double> get_element_as_complex(uint64_t i, uint64_t j) const {
        return std::complex<double>(get_element_as_double(i, j), 0.0);
    }

    // Virtual setter for generic assignment (Phase 2.2)
    virtual void set_element_as_double(uint64_t i, uint64_t j, double value) = 0;

    // Phase 2.5: Batch Write
    // Allows writing a block of data at once, enabling vectorization in derived classes.
    virtual void write_block(uint64_t start_row, uint64_t start_col, 
                             uint64_t num_rows, uint64_t num_cols, 
                             const double* buffer, uint64_t stride) {
        // Default implementation: loop and set
        for (uint64_t i = 0; i < num_rows; ++i) {
            for (uint64_t j = 0; j < num_cols; ++j) {
                set_element_as_double(start_row + i, start_col + j, buffer[i * stride + j]);
            }
        }
    }

    // Lazy Assignment Operator (Phase 2.2)
    template <typename E>
    MatrixBase& operator=(const MatrixExpression<E>& expr) {
        // 1. Aliasing Check (Phase 2.3)
        if (expr.aliases(this)) {
            // Safe fallback: Evaluate to temporary to prevent data corruption.
            // This handles cases like A = A + B or A = A.transpose().
            auto temp_obj = this->clone();
            MatrixBase* temp = dynamic_cast<MatrixBase*>(temp_obj.get());
            if (!temp) throw std::runtime_error("Internal Error: Clone returned non-MatrixBase");
            
            *temp = expr;  // Evaluate expression into temporary
            *this = *temp; // Copy temporary back to this
            return *this;
        }

        // 2. Dimension Check
        if (rows() != expr.rows() || cols() != expr.cols()) {
            throw std::runtime_error("Dimension mismatch in assignment");
        }

        // 3. Touch Operands (Phase 4)
        // Ensure operands are marked as recently used so they aren't evicted
        // if we need to spill something else during evaluation.
        expr.touch_operands();

        // 4. Evaluation Loop (Phase 2.5: Blocked Evaluation)
        const uint64_t BLOCK_SIZE = 256; 
        std::vector<double> buffer(BLOCK_SIZE * BLOCK_SIZE);

        for (uint64_t i = 0; i < rows(); i += BLOCK_SIZE) {
            uint64_t r_chunk = std::min(BLOCK_SIZE, rows() - i);
            for (uint64_t j = 0; j < cols(); j += BLOCK_SIZE) {
                uint64_t c_chunk = std::min(BLOCK_SIZE, cols() - j);
                
                // Fill buffer from expression
                expr.fill_buffer(buffer.data(), i, j, r_chunk, c_chunk, BLOCK_SIZE);
                
                // Write buffer to matrix
                this->write_block(i, j, r_chunk, c_chunk, buffer.data(), BLOCK_SIZE);
            }
        }

        return *this;
    }

    // Copy assignment from another MatrixBase (via expression engine)
    MatrixBase& operator=(const MatrixBase& other);

    // In-place operators (Declarations)
    MatrixBase& operator+=(const MatrixBase& other);
    MatrixBase& operator-=(const MatrixBase& other);
    MatrixBase& operator*=(double scalar);
    MatrixBase& operator/=(double scalar);

    template <typename E>
    MatrixBase& operator+=(const MatrixExpression<E>& expr);

    template <typename E>
    MatrixBase& operator-=(const MatrixExpression<E>& expr);

    // Phase 4.3: Spill Mechanism
    // Moves data from RAM (Anonymous) to a file-backed mapping.
    virtual void spill_to_disk(const std::string& filename);

    virtual std::unique_ptr<MatrixBase> multiply_scalar(double scalar, const std::string& result_file = "") const = 0;
    virtual std::unique_ptr<MatrixBase> multiply_scalar(int64_t scalar, const std::string& result_file = "") const = 0;
    virtual std::unique_ptr<MatrixBase> multiply_scalar(std::complex<double> scalar, const std::string& result_file = "") const {
        throw std::runtime_error("Complex scalar multiplication not implemented for this matrix type");
    }

    virtual std::unique_ptr<MatrixBase> add_scalar(double scalar, const std::string& result_file = "") const = 0;
    virtual std::unique_ptr<MatrixBase> add_scalar(int64_t scalar, const std::string& result_file = "") const = 0;

    virtual std::unique_ptr<MatrixBase> transpose(const std::string& result_file = "") const = 0;

protected:
    // Intentionally no in-object caches here.
    // Pre-alpha policy: cached derived values live in the unified properties/cached.*
    // system (Python layer + persistence), not in ad-hoc native members.

    void set_view(uint64_t logical_rows, uint64_t logical_cols, uint64_t row_offset, uint64_t col_offset) {
        logical_rows_ = logical_rows;
        logical_cols_ = logical_cols;
        row_offset_ = row_offset;
        col_offset_ = col_offset;
    }

    uint64_t logical_rows_ = 0;
    uint64_t logical_cols_ = 0;
    uint64_t row_offset_ = 0;
    uint64_t col_offset_ = 0;
};

} // namespace pycauset
