#include "VectorBase.hpp"

VectorBase::VectorBase(uint64_t n) : PersistentObject(), n_(n) {}

VectorBase::VectorBase(uint64_t n, 
                       std::unique_ptr<MemoryMapper> mapper,
                       pycauset::MatrixType matrix_type,
                       pycauset::DataType data_type) 
    : PersistentObject(std::move(mapper), matrix_type, data_type, n, 1), n_(n) {}
