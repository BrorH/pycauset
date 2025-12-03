#pragma once

#include <cstdint>
#include <memory>
#include <string>

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

    virtual std::unique_ptr<MatrixBase> multiply_scalar(double factor, const std::string& result_file = "") const = 0;
    virtual std::unique_ptr<MatrixBase> multiply_scalar(int64_t factor, const std::string& result_file = "") const = 0;

    virtual std::unique_ptr<MatrixBase> transpose(const std::string& result_file = "") const = 0;

protected:
    uint64_t n_;
};
