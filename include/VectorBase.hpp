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
    // Constructor for subclasses that initialize storage themselves (like UnitVector)
    VectorBase(uint64_t n, 
               uint64_t size_in_bytes, 
               const std::string& backing_file,
               const std::string& fallback_name,
               pycauset::MatrixType mtype,
               pycauset::DataType dtype)
        : PersistentObject(), n_(n) {
        initialize_storage(size_in_bytes, backing_file, 
                         fallback_name, 
                         size_in_bytes > 0 ? 8 : 0, // Min size
                         mtype, 
                         dtype,
                         n, 1);
    }

    // Constructor for loading subclasses with explicit metadata
    VectorBase(uint64_t n, 
               uint64_t size_in_bytes,
               const std::string& backing_file,
               size_t offset,
               uint64_t seed,
               double scalar,
               bool is_transposed,
               pycauset::MatrixType mtype,
               pycauset::DataType dtype)
        : PersistentObject(), n_(n) {
            
        initialize_storage(size_in_bytes, backing_file, 
                         "", 
                         size_in_bytes > 0 ? 8 : 0,
                         mtype, 
                         dtype,
                         n, 1,
                         offset,
                         false);
        
        set_seed(seed);
        set_scalar(scalar);
        set_transposed(is_transposed);
    }

    // Constructor for loading from existing mapper
    VectorBase(uint64_t n, 
               std::unique_ptr<MemoryMapper> mapper,
               pycauset::MatrixType matrix_type,
               pycauset::DataType data_type);

    uint64_t n_;
};
