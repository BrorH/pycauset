#pragma once

#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/core/MatrixTraits.hpp"
#include <algorithm>
#include <stdexcept>

namespace pycauset {

// Structure bases (see MatrixBase.hpp). Symmetric and antisymmetric matrices share one
// implementation (SymmetricMatrix<T> with a runtime flag), so every packed (anti)symmetric
// object derives BOTH bases; the Python type hook picks the class from the flag.
class SymmetricMatrixBase : public TriangularMatrixBase {
public:
    using TriangularMatrixBase::TriangularMatrixBase;
};

class AntiSymmetricMatrixBase : public SymmetricMatrixBase {
public:
    using SymmetricMatrixBase::SymmetricMatrixBase;
};

template <typename T>
class SymmetricMatrix : public AntiSymmetricMatrixBase {
public:
    SymmetricMatrix(uint64_t n, const std::string& backing_file = "", bool is_antisymmetric = false)
        : AntiSymmetricMatrixBase(n, is_antisymmetric ? Structure::ANTISYMMETRIC : Structure::SYMMETRIC, MatrixTraits<T>::data_type),
          is_antisymmetric_(is_antisymmetric) {
        
        has_diagonal_ = true; // Symmetric matrices always have a diagonal (even if 0 for antisymmetric)
        
        // Calculate offsets for T (sizeof(T)*8 bits per element), aligned to 64 bits
        // We store the upper triangle including diagonal.
        // Number of elements = n*(n+1)/2
        uint64_t size_in_bytes = calculate_triangular_offsets(sizeof(T) * 8, 64);
        
        initialize_storage(size_in_bytes, backing_file, 
                         std::string(is_antisymmetric ? "antisymmetric_" : "symmetric_") + MatrixTraits<T>::name, 
                         8,
                         is_antisymmetric ? Structure::ANTISYMMETRIC : Structure::SYMMETRIC, 
                         MatrixTraits<T>::data_type,
                         n, n);
    }

    // Constructor for loading with explicit metadata
    SymmetricMatrix(uint64_t n, 
                     const std::string& backing_file,
                     size_t offset,
                     uint64_t seed,
                     std::complex<double> scalar,
                     bool is_transposed,
                     bool is_antisymmetric)
        : AntiSymmetricMatrixBase(n, is_antisymmetric ? Structure::ANTISYMMETRIC : Structure::SYMMETRIC, MatrixTraits<T>::data_type),
          is_antisymmetric_(is_antisymmetric) {
        
        has_diagonal_ = true;
        uint64_t size_in_bytes = calculate_triangular_offsets(sizeof(T) * 8, 64);
        
        initialize_storage(size_in_bytes, backing_file, 
                         "", 
                         8,
                         is_antisymmetric ? Structure::ANTISYMMETRIC : Structure::SYMMETRIC, 
                         MatrixTraits<T>::data_type,
                         n, n,
                         offset,
                         false);
        
        set_seed(seed);
        set_scalar(scalar);
        set_transposed(is_transposed);
    }

    SymmetricMatrix(uint64_t n, std::shared_ptr<MemoryMapper> mapper, bool is_antisymmetric = false)
        : AntiSymmetricMatrixBase(n, std::move(mapper), is_antisymmetric ? Structure::ANTISYMMETRIC : Structure::SYMMETRIC, MatrixTraits<T>::data_type),
          is_antisymmetric_(is_antisymmetric) {
        has_diagonal_ = true;
        calculate_triangular_offsets(sizeof(T) * 8, 64);
    }

    bool is_antisymmetric() const { return is_antisymmetric_; }

    void set(uint64_t i, uint64_t j, T value) {
        ensure_unique();
        
        if (i >= rows() || j >= cols()) throw std::out_of_range("Index out of bounds");

        bool swapped = false;
        if (i > j) {
            std::swap(i, j);
            swapped = true;
        }

        if (is_antisymmetric_) {
            if (i == j) {
                if (value != T(0)) {
                    throw std::invalid_argument("Diagonal of anti-symmetric matrix must be zero");
                }
                // We still set it to 0 in storage to be safe/consistent
            } else if (swapped) {
                // User set lower triangle: A[j, i] = val => A[i, j] = -val
                value = -value;
            }
        }

        // Access storage (Upper Triangular)
        // Row i offset + j
        // Note: TriangularMatrixBase::calculate_triangular_offsets assumes a specific packing.
        // Usually row_offsets_[i] gives the start of row i.
        // In packed upper triangular, row i has length (N-i) or something similar depending on packing.
        // Let's assume standard packed storage where row i starts at row_offsets_[i].
        // Since we are storing upper triangle, for a given row i, the columns are j=i, i+1, ..., N-1.
        // So the index within the row is (j - i).
        
        uint64_t byte_offset = get_row_offset(i) + (j - i) * sizeof(T);
        
        // Write value
        T* ptr = reinterpret_cast<T*>(static_cast<uint8_t*>(mapper_->get_data()) + byte_offset);
        *ptr = value;
    }

    T get(uint64_t i, uint64_t j) const {
        if (i >= rows() || j >= cols()) throw std::out_of_range("Index out of bounds");

        bool swapped = false;
        if (i > j) {
            std::swap(i, j);
            swapped = true;
        }

        // Read from storage (Upper Triangular)
        uint64_t byte_offset = get_row_offset(i) + (j - i) * sizeof(T);
        const T* ptr = reinterpret_cast<const T*>(static_cast<const uint8_t*>(mapper_->get_data()) + byte_offset);
        T val = *ptr;

        if (is_antisymmetric_ && swapped) {
            return -val;
        }
        return val;
    }

    T* data() { return static_cast<T*>(this->require_mapper()->get_data()); }
    const T* data() const { return static_cast<const T*>(this->require_mapper()->get_data()); }

    // Implement MatrixBase virtuals
    double get_element_as_double(uint64_t i, uint64_t j) const override {
        if (scalar_ == 1.0) {
            return static_cast<double>(get(i, j));
        }
        return (static_cast<double>(get(i, j)) * scalar_).real();
    }

    std::complex<double> get_element_as_complex(uint64_t i, uint64_t j) const override {
        if (scalar_ == 1.0) {
            return std::complex<double>(static_cast<double>(get(i, j)), 0.0);
        }
        return static_cast<double>(get(i, j)) * scalar_;
    }

    void set_element_as_double(uint64_t i, uint64_t j, double value) override {
        set(i, j, static_cast<T>(value));
    }

    std::unique_ptr<MatrixBase> multiply_scalar(double scalar, const std::string& result_file = "") const override {
        // Metadata update
        auto clone_ptr = this->clone();
        auto mat_ptr = std::unique_ptr<MatrixBase>(dynamic_cast<MatrixBase*>(clone_ptr.release()));
        mat_ptr->set_scalar(mat_ptr->get_scalar() * scalar);
        if (!result_file.empty()) {
             // If result file is specified, we might need to copy storage, but here we just clone and update scalar.
             // If deep copy is needed, clone() usually does CoW or shallow copy.
             // For now, follow pattern.
        }
        return mat_ptr;
    }

    std::unique_ptr<MatrixBase> multiply_scalar(int64_t scalar, const std::string& result_file = "") const override {
        return multiply_scalar(static_cast<double>(scalar), result_file);
    }

    // S + s stays symmetric; an antisymmetric A + s is not antisymmetric, so it becomes dense.
    // Values are read through the view (factor applied).
    std::unique_ptr<MatrixBase> add_scalar(double scalar, const std::string& result_file = "") const override {
        const uint64_t n = this->rows();
        if (!is_antisymmetric_) {
            auto out = std::make_unique<SymmetricMatrix<T>>(n, result_file, false);
            for (uint64_t i = 0; i < n; ++i) {
                for (uint64_t j = i; j < n; ++j) {
                    out->set(i, j, static_cast<T>(this->get_element_as_double(i, j) + scalar));
                }
            }
            return out;
        }
        auto out = std::make_unique<DenseMatrix<T>>(n, n, result_file);
        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t j = 0; j < n; ++j) {
                out->set(i, j, static_cast<T>(this->get_element_as_double(i, j) + scalar));
            }
        }
        return out;
    }

    std::unique_ptr<MatrixBase> add_scalar(int64_t scalar, const std::string& result_file = "") const override {
        return add_scalar(static_cast<double>(scalar), result_file);
    }

    std::unique_ptr<MatrixBase> transpose(const std::string& result_file = "") const override {
        auto clone_ptr = this->clone();
        auto mat_ptr = std::unique_ptr<MatrixBase>(dynamic_cast<MatrixBase*>(clone_ptr.release()));
        
        if (is_antisymmetric_) {
            // Transpose of anti-symmetric is negation.
            // A^T = -A.
            // We can represent this by flipping the scalar?
            mat_ptr->set_scalar(mat_ptr->get_scalar() * -1.0);
        } else {
            // Transpose of symmetric is itself.
            // No change needed.
        }
        return mat_ptr;
    }

    std::unique_ptr<PersistentObject> clone() const override {
        // Create a new object sharing the same mapper (CoW handled by PersistentObject/MemoryMapper)
        // Keep the view metadata: transpose/negation and scaling are stored in it (the factor
        // used to be dropped here, so A.T * 3 gave 3A and A.T.T gave -A for antisymmetric A).
        auto out = std::make_unique<SymmetricMatrix<T>>(base_rows(), mapper_, is_antisymmetric_);
        out->set_scalar(this->get_scalar());
        out->set_conjugated(this->is_conjugated());
        out->set_transposed(this->is_transposed());
        out->set_seed(this->get_seed());
        return out;
    }

    static std::unique_ptr<SymmetricMatrix<T>> from_triangular(const TriangularMatrix<T>& source, const std::string& backing_file = "") {
        auto n = source.rows();
        auto mat = std::make_unique<SymmetricMatrix<T>>(n, backing_file, false);
        
        // Copy upper triangle including diagonal
        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t j = i; j < n; ++j) {
                mat->set(i, j, source.get(i, j));
            }
        }
        return mat;
    }

protected:
    bool is_antisymmetric_;
};

template <typename T>
class AntiSymmetricMatrix : public SymmetricMatrix<T> {
public:
    AntiSymmetricMatrix(uint64_t n, const std::string& backing_file = "")
        : SymmetricMatrix<T>(n, backing_file, true) {}

    AntiSymmetricMatrix(uint64_t n, 
                     const std::string& backing_file,
                     size_t offset,
                     uint64_t seed,
                     std::complex<double> scalar,
                     bool is_transposed)
        : SymmetricMatrix<T>(n, backing_file, offset, seed, scalar, is_transposed, true) {}

    AntiSymmetricMatrix(uint64_t n, std::shared_ptr<MemoryMapper> mapper)
        : SymmetricMatrix<T>(n, std::move(mapper), true) {}

    std::unique_ptr<PersistentObject> clone() const override {
        auto out = std::make_unique<AntiSymmetricMatrix<T>>(this->base_rows(), this->mapper_);
        out->set_scalar(this->get_scalar());
        out->set_conjugated(this->is_conjugated());
        out->set_transposed(this->is_transposed());
        out->set_seed(this->get_seed());
        return out;
    }

    static std::unique_ptr<AntiSymmetricMatrix<T>> from_triangular(const TriangularMatrix<T>& source, const std::string& backing_file = "") {
        auto n = source.rows();
        auto mat = std::make_unique<AntiSymmetricMatrix<T>>(n, backing_file);
        
        // Copy upper triangle excluding diagonal (diagonal is 0)
        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t j = i + 1; j < n; ++j) {
                mat->set(i, j, source.get(i, j));
            }
        }
        return mat;
    }
};

} // namespace pycauset
