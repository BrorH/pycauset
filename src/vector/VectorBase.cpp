#include "pycauset/vector/VectorBase.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/core/ObjectFactory.hpp"
#include <memory>

namespace pycauset {

VectorBase::VectorBase(uint64_t n) : PersistentObject(), n_(n) {}

VectorBase::VectorBase(uint64_t n, 
                       std::shared_ptr<::MemoryMapper> mapper,
                       pycauset::MatrixType matrix_type,
                       pycauset::DataType data_type) 
    : PersistentObject(std::move(mapper), matrix_type, data_type, n, 1), n_(n) {}

VectorBase::VectorBase(uint64_t n, 
                       uint64_t size_in_bytes, 
                       const std::string& backing_file, 
                       const std::string& default_prefix, 
                       pycauset::MatrixType matrix_type, 
                       pycauset::DataType data_type)
    : PersistentObject(), n_(n) {
    // element_size = 0 if size_in_bytes is 0? Or maybe 8 for double?
    // If size_in_bytes is 0, it's likely virtual/no storage.
    uint64_t element_size = (size_in_bytes > 0) ? (size_in_bytes / n) : 0;
    initialize_storage(size_in_bytes, backing_file, default_prefix, element_size, matrix_type, data_type, n, 1);
}

VectorBase::VectorBase(uint64_t n, 
                       uint64_t size_in_bytes, 
                       const std::string& backing_file, 
                       size_t offset,
                       uint64_t seed,
                       std::complex<double> scalar,
                       bool is_transposed,
                       pycauset::MatrixType matrix_type, 
                       pycauset::DataType data_type)
    : PersistentObject(), n_(n) {
    uint64_t element_size = (size_in_bytes > 0) ? (size_in_bytes / n) : 0;
    initialize_storage(size_in_bytes, backing_file, "", element_size, matrix_type, data_type, n, 1, offset, false);
    set_seed(seed);
    set_scalar(scalar);
    set_transposed(is_transposed);
}

std::unique_ptr<PersistentObject> VectorBase::clone() const {
    return ObjectFactory::clone_vector(mapper_, rows_, cols_, data_type_, matrix_type_, seed_, scalar_, is_transposed_);
}

} // namespace pycauset
