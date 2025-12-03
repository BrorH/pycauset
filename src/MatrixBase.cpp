#include "MatrixBase.hpp"

MatrixBase::MatrixBase(uint64_t n, 
                       pycauset::MatrixType matrix_type,
                       pycauset::DataType data_type)
    : PersistentObject(), n_(n) {
    // Initialize metadata members in PersistentObject
    matrix_type_ = matrix_type;
    data_type_ = data_type;
    rows_ = n;
    cols_ = n;
}

MatrixBase::MatrixBase(uint64_t n, 
                       std::unique_ptr<MemoryMapper> mapper,
                       pycauset::MatrixType matrix_type,
                       pycauset::DataType data_type,
                       uint64_t seed,
                       double scalar,
                       bool is_transposed,
                       bool is_temporary)
    : PersistentObject(std::move(mapper), matrix_type, data_type, n, n, seed, scalar, is_transposed, is_temporary), 
      n_(n) {}
