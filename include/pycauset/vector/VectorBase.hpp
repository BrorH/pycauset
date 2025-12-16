#pragma once

#include <cstdint>
#include <complex>
#include <memory>
#include <stdexcept>
#include <string>

#include "pycauset/core/PersistentObject.hpp"

namespace pycauset {

class VectorBase : public PersistentObject {
public:
    explicit VectorBase(uint64_t n);

    VectorBase(uint64_t n, 
               std::shared_ptr<::MemoryMapper> mapper,
               pycauset::MatrixType matrix_type,
               pycauset::DataType data_type);

    VectorBase(uint64_t n, 
               uint64_t size_in_bytes, 
               const std::string& backing_file, 
               const std::string& default_prefix, 
               pycauset::MatrixType matrix_type, 
               pycauset::DataType data_type);

    VectorBase(uint64_t n, 
               uint64_t size_in_bytes, 
               const std::string& backing_file, 
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed,
               pycauset::MatrixType matrix_type, 
               pycauset::DataType data_type);

    virtual ~VectorBase() = default;

    std::unique_ptr<PersistentObject> clone() const override;

    uint64_t size() const { return n_; }

    virtual double get_element_as_double(uint64_t i) const = 0;

    virtual std::complex<double> get_element_as_complex(uint64_t i) const {
        return std::complex<double>(get_element_as_double(i), 0.0);
    }

    virtual std::unique_ptr<VectorBase> transpose(const std::string& saveas = "") const = 0;

    virtual std::unique_ptr<VectorBase> multiply_scalar(double factor, const std::string& result_file = "") const = 0;
    virtual std::unique_ptr<VectorBase> multiply_scalar(std::complex<double> factor, const std::string& result_file = "") const {
        throw std::runtime_error("Complex scalar multiplication not implemented for this vector type");
    }
    virtual std::unique_ptr<VectorBase> add_scalar(double scalar, const std::string& result_file = "") const = 0;

protected:
    uint64_t n_;
};

} // namespace pycauset
