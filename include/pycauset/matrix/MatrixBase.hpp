#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <optional>
#include <vector>
#include <complex>
#include <iostream>

#include "pycauset/core/PersistentObject.hpp"

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
