#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/core/ObjectFactory.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include <memory>
#include <cstring>

namespace pycauset {

MatrixBase::MatrixBase(uint64_t n, 
                       pycauset::Structure structure,
                       pycauset::DataType data_type)
    : MatrixBase(n, n, structure, data_type) {}

MatrixBase::MatrixBase(uint64_t rows,
                       uint64_t cols,
                       pycauset::Structure structure,
                       pycauset::DataType data_type)
    : PersistentObject() {
    structure_ = structure;
    data_type_ = data_type;
    rows_ = rows;
    cols_ = cols;
    logical_rows_ = rows;
    logical_cols_ = cols;
    row_offset_ = 0;
    col_offset_ = 0;
}

MatrixBase::MatrixBase(uint64_t n, 
                       std::shared_ptr<MemoryMapper> mapper,
                       pycauset::Structure structure,
                       pycauset::DataType data_type,
                       uint64_t seed,
                       std::complex<double> scalar,
                       bool is_transposed)
    : MatrixBase(n, n, std::move(mapper), structure, data_type, seed, scalar, is_transposed) {}

MatrixBase::MatrixBase(uint64_t rows,
                       uint64_t cols,
                       std::shared_ptr<MemoryMapper> mapper,
                       pycauset::Structure structure,
                       pycauset::DataType data_type,
                       uint64_t seed,
                       std::complex<double> scalar,
                       bool is_transposed)
    : PersistentObject(std::move(mapper), structure, data_type, rows, cols, seed, scalar, is_transposed) {
    logical_rows_ = rows;
    logical_cols_ = cols;
    row_offset_ = 0;
    col_offset_ = 0;
}

std::unique_ptr<PersistentObject> MatrixBase::clone() const {
    // The triangular layout (row lengths) depends on has_diagonal: a clone without it read
    // shifted values (F-064; the bit triangle with a diagonal hit it again, N.5.6).
    const auto* tri = dynamic_cast<const TriangularMatrixBase*>(this);
    const bool has_diagonal = tri != nullptr && structure_ == Structure::TRIANGULAR && tri->has_diagonal();
    auto out = ObjectFactory::clone_matrix(mapper_, base_rows(), base_cols(), data_type_, structure_, seed_, scalar_,
                                           is_transposed_, has_diagonal);
    out->set_conjugated(is_conjugated());
    if (auto* m = dynamic_cast<MatrixBase*>(out.get())) {
        m->set_view(logical_rows_, logical_cols_, row_offset_, col_offset_);
    }
    return out;
}






} // namespace pycauset

