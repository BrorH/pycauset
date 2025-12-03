#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "PersistentObject.hpp"

class VectorBase : public PersistentObject {
public:
    explicit VectorBase(uint64_t n);
    virtual ~VectorBase() = default;

    uint64_t size() const { return n_; }

    virtual double get_element_as_double(uint64_t i) const = 0;

    virtual std::unique_ptr<VectorBase> transpose(const std::string& saveas = "") const = 0;

protected:
    // Constructor for loading from existing mapper
    VectorBase(uint64_t n, 
               std::unique_ptr<MemoryMapper> mapper,
               pycauset::MatrixType matrix_type,
               pycauset::DataType data_type);

    uint64_t n_;
};
