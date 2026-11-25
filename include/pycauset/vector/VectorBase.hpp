#pragma once

#include "pycauset/core/AccessCounter.hpp"

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
               pycauset::Structure structure,
               pycauset::DataType data_type);

    VectorBase(uint64_t n, 
               uint64_t size_in_bytes, 
               const std::string& backing_file, 
               const std::string& default_prefix, 
               pycauset::Structure structure, 
               pycauset::DataType data_type);

    VectorBase(uint64_t n, 
               uint64_t size_in_bytes, 
               const std::string& backing_file, 
               size_t offset,
               uint64_t seed,
               std::complex<double> scalar,
               bool is_transposed,
               pycauset::Structure structure, 
               pycauset::DataType data_type);

    virtual ~VectorBase() = default;

    std::unique_ptr<PersistentObject> clone() const override;

    uint64_t size() const { return n_; }

    // Generic per-element access; see MatrixBase (counted wrappers over element_* virtuals).
    double get_element_as_double(uint64_t i) const {
        access_counter::hit();
        return element_as_double(i);
    }
    std::complex<double> get_element_as_complex(uint64_t i) const {
        access_counter::hit();
        return element_as_complex(i);
    }

protected:
    virtual double element_as_double(uint64_t i) const = 0;
    virtual std::complex<double> element_as_complex(uint64_t i) const {
        return std::complex<double>(element_as_double(i), 0.0);
    }

public:

    virtual std::unique_ptr<VectorBase> transpose(const std::string& saveas = "") const = 0;


protected:
    uint64_t n_;
};

// Structure base for dense vectors of every dtype (see the note in MatrixBase.hpp).
class DenseVectorBase : public VectorBase {
public:
    using VectorBase::VectorBase;
};

} // namespace pycauset
