#pragma once

#include "pycauset/core/AccessCounter.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <optional>
#include <vector>
#include <complex>
#include <iostream>
#include <algorithm>

#include "pycauset/core/PersistentObject.hpp"

namespace pycauset {

class MatrixBase : public PersistentObject {
public:
    // Constructor for creating new matrix (square)
    MatrixBase(uint64_t n,
               pycauset::Structure structure,
               pycauset::DataType data_type);

    // Constructor for creating new matrix (rectangular)
    MatrixBase(uint64_t rows,
               uint64_t cols,
               pycauset::Structure structure,
               pycauset::DataType data_type);

    // Constructor for loading/wrapping existing storage (square)
    MatrixBase(uint64_t n,
               std::shared_ptr<MemoryMapper> mapper,
               pycauset::Structure structure,
               pycauset::DataType data_type,
               uint64_t seed = 0,
               std::complex<double> scalar = 1.0,
               bool is_transposed = false,
               bool is_temporary = false);

    // Constructor for loading/wrapping existing storage (rectangular)
    MatrixBase(uint64_t rows,
               uint64_t cols,
               std::shared_ptr<MemoryMapper> mapper,
               pycauset::Structure structure,
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

    // A window into larger storage: offset, or fewer rows/cols than the storage. Kernels that
    // read storage directly must treat a subview as non-plain (a [0:3, 0:3] window has no
    // offset but a different row stride: F-062).
    bool is_subview() const {
        return row_offset_ != 0 || col_offset_ != 0 || logical_rows_ != get_rows() || logical_cols_ != get_cols();
    }

    // Generic per-element access (logical coordinates, view metadata applied). Non-virtual
    // wrappers count calls (access_counter, debug only) and dispatch to the element_* virtuals
    // that each storage class implements. Kernels should not loop over these.
    double get_element_as_double(uint64_t i, uint64_t j) const {
        access_counter::hit();
        return element_as_double(i, j);
    }
    std::complex<double> get_element_as_complex(uint64_t i, uint64_t j) const {
        access_counter::hit();
        return element_as_complex(i, j);
    }
    void set_element_as_double(uint64_t i, uint64_t j, double value) {
        access_counter::hit();
        set_element_double(i, j, value);
    }

protected:
    virtual double element_as_double(uint64_t i, uint64_t j) const = 0;
    virtual std::complex<double> element_as_complex(uint64_t i, uint64_t j) const {
        return std::complex<double>(element_as_double(i, j), 0.0);
    }
    virtual void set_element_double(uint64_t i, uint64_t j, double value) = 0;

public:

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

    // Phase 4.3: Spill Mechanism
    // Moves data from RAM (Anonymous) to a file-backed mapping.
    virtual void spill_to_disk(const std::string& filename);

    virtual std::unique_ptr<MatrixBase> multiply_scalar(double scalar, const std::string& result_file = "") const = 0;
    virtual std::unique_ptr<MatrixBase> multiply_scalar(int64_t scalar, const std::string& result_file = "") const = 0;
    virtual std::unique_ptr<MatrixBase> multiply_scalar(std::complex<double> scalar, const std::string& result_file = "") const {
        throw std::runtime_error("Complex scalar multiplication not implemented for this matrix type");
    }

    // O(1) scaled view sharing storage: clone() keeps every view flag (transpose, conjugation,
    // window), then the factor is scaled. value = conj?(raw * s), so scaling the value by f
    // scales s by conj(f) under a pending conjugation (F-071). Structured classes use this
    // instead of hand-built copies, which dropped the conjugation flag (A.conj() * 0.5).
    std::unique_ptr<MatrixBase> scaled_clone(double f) const {
        auto out = std::unique_ptr<MatrixBase>(static_cast<MatrixBase*>(clone().release()));
        out->set_scalar(get_scalar() * f);
        return out;
    }
    // O(1) views sharing this object's storage, with its whole view state (window, offsets,
    // transposition, conjugation, factor, temporary flag) kept: one implementation for every
    // class (hand-built copies passed the transposed shape as the storage shape and dropped
    // windows, F-097).
    std::unique_ptr<MatrixBase> scaled_clone(std::complex<double> f) const {
        auto out = std::unique_ptr<MatrixBase>(static_cast<MatrixBase*>(clone().release()));
        out->set_scalar(get_scalar() * (is_conjugated() ? std::conj(f) : f));
        out->set_temporary(is_temporary());
        return out;
    }
    std::unique_ptr<MatrixBase> transposed_clone() const {
        auto out = std::unique_ptr<MatrixBase>(static_cast<MatrixBase*>(clone().release()));
        out->set_transposed(!is_transposed());
        out->set_temporary(is_temporary());
        return out;
    }

    virtual std::unique_ptr<MatrixBase> transpose(const std::string& result_file = "") const = 0;

protected:
    // No in-object result caches (F-081); the planned derived-value cache lives on the storage
    // (review/features/DERIVED_CACHE.md).

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

// ---------------------------------------------------------------------------
// Structure bases. One per user-visible structure; no data, no behaviour. They exist
// so Python can expose ONE class per structure (DenseMatrix, DiagonalMatrix, ...) with
// the dtype as an attribute: the polymorphic type hook in
// src/bindings/bindings_common.hpp maps every concrete template instantiation to its
// structure class. (review/RESTRUCTURE.md 3.1 item 4.)
// ---------------------------------------------------------------------------
class DenseMatrixBase : public MatrixBase {
public:
    using MatrixBase::MatrixBase;
};

class DiagonalMatrixBase : public MatrixBase {
public:
    using MatrixBase::MatrixBase;
};

class IdentityMatrixBase : public MatrixBase {
public:
    using MatrixBase::MatrixBase;
};

} // namespace pycauset
