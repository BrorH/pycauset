#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/core/ObjectFactory.hpp"
#include <memory>

namespace pycauset {

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
                       std::shared_ptr<MemoryMapper> mapper,
                       pycauset::MatrixType matrix_type,
                       pycauset::DataType data_type,
                       uint64_t seed,
                       std::complex<double> scalar,
                       bool is_transposed,
                       bool is_temporary)
    : PersistentObject(std::move(mapper), matrix_type, data_type, n, n, seed, scalar, is_transposed, is_temporary), 
      n_(n) {}

std::unique_ptr<PersistentObject> MatrixBase::clone() const {
    auto out = ObjectFactory::clone_matrix(mapper_, rows(), cols(), data_type_, matrix_type_, seed_, scalar_, is_transposed_);
    out->set_conjugated(is_conjugated());
    return out;
}

} // namespace pycauset
