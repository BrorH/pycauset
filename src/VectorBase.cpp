#include "VectorBase.hpp"

VectorBase::VectorBase(uint64_t n) : PersistentObject(), n_(n) {}

VectorBase::VectorBase(uint64_t n, std::unique_ptr<MemoryMapper> mapper) 
    : PersistentObject(std::move(mapper)), n_(n) {}
