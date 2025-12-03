#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <optional>
#include <vector>
#include <complex>

#include "PersistentObject.hpp"

class MatrixBase : public PersistentObject {
public:
    // Constructor for creating new matrix
    MatrixBase(uint64_t n, 
               pycauset::MatrixType matrix_type,
               pycauset::DataType data_type);

    // Constructor for loading/wrapping existing storage
    MatrixBase(uint64_t n, 
               std::unique_ptr<MemoryMapper> mapper,
               pycauset::MatrixType matrix_type,
               pycauset::DataType data_type,
               uint64_t seed = 0,
               double scalar = 1.0,
               bool is_transposed = false,
               bool is_temporary = false);

    virtual ~MatrixBase() = default;

    uint64_t size() const { return n_; }

    virtual double get_element_as_double(uint64_t i, uint64_t j) const = 0;

    virtual std::unique_ptr<MatrixBase> multiply_scalar(double scalar, const std::string& result_file = "") const = 0;
    virtual std::unique_ptr<MatrixBase> multiply_scalar(int64_t scalar, const std::string& result_file = "") const = 0;

    virtual std::unique_ptr<MatrixBase> add_scalar(double scalar, const std::string& result_file = "") const = 0;
    virtual std::unique_ptr<MatrixBase> add_scalar(int64_t scalar, const std::string& result_file = "") const = 0;

    virtual std::unique_ptr<MatrixBase> transpose(const std::string& result_file = "") const = 0;

    // Caching support
    std::optional<double> get_cached_trace() const { return cached_trace_; }
    void set_cached_trace(double val) const { cached_trace_ = val; }

    std::optional<double> get_cached_determinant() const { return cached_determinant_; }
    void set_cached_determinant(double val) const { cached_determinant_ = val; }

    std::optional<std::vector<std::complex<double>>> get_cached_eigenvalues() const { return cached_eigenvalues_; }
    void set_cached_eigenvalues(const std::vector<std::complex<double>>& val) const { cached_eigenvalues_ = val; }
    void clear_cached_eigenvalues() const { cached_eigenvalues_ = std::nullopt; }

protected:
    uint64_t n_;
    
    // Cache members
    mutable std::optional<double> cached_trace_;
    mutable std::optional<double> cached_determinant_;
    mutable std::optional<std::vector<std::complex<double>>> cached_eigenvalues_;
};
