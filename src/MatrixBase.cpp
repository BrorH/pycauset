#include "MatrixBase.hpp"

MatrixBase::MatrixBase(uint64_t n) : PersistentObject(), n_(n) {}

MatrixBase::MatrixBase(uint64_t n, std::unique_ptr<MemoryMapper> mapper) 
    : PersistentObject(std::move(mapper)), n_(n) {}
