#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "PersistentObject.hpp"

class MatrixBase : public PersistentObject {
public:
    explicit MatrixBase(uint64_t n);
    virtual ~MatrixBase() = default;

    uint64_t size() const { return n_; }

    virtual double get_element_as_double(uint64_t i, uint64_t j) const = 0;

protected:
    // Constructor for loading from existing mapper
    MatrixBase(uint64_t n, std::unique_ptr<MemoryMapper> mapper);

    uint64_t n_;
};
