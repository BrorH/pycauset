#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/core/ObjectFactory.hpp"
#include "pycauset/matrix/expression/MatrixRefExpression.hpp"
#include "pycauset/matrix/MatrixOps.hpp"
#include "pycauset/compute/ComputeContext.hpp"
#include "pycauset/compute/AutoSolver.hpp"
#include <memory>
#include <cstring>

namespace pycauset {

MatrixBase::MatrixBase(uint64_t n, 
                       pycauset::MatrixType matrix_type,
                       pycauset::DataType data_type)
    : MatrixBase(n, n, matrix_type, data_type) {}

MatrixBase::MatrixBase(uint64_t rows,
                       uint64_t cols,
                       pycauset::MatrixType matrix_type,
                       pycauset::DataType data_type)
    : PersistentObject() {
    matrix_type_ = matrix_type;
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
                       pycauset::MatrixType matrix_type,
                       pycauset::DataType data_type,
                       uint64_t seed,
                       std::complex<double> scalar,
                       bool is_transposed,
                       bool is_temporary)
    : MatrixBase(n, n, std::move(mapper), matrix_type, data_type, seed, scalar, is_transposed, is_temporary) {}

MatrixBase::MatrixBase(uint64_t rows,
                       uint64_t cols,
                       std::shared_ptr<MemoryMapper> mapper,
                       pycauset::MatrixType matrix_type,
                       pycauset::DataType data_type,
                       uint64_t seed,
                       std::complex<double> scalar,
                       bool is_transposed,
                       bool is_temporary)
    : PersistentObject(std::move(mapper), matrix_type, data_type, rows, cols, seed, scalar, is_transposed, is_temporary) {
    logical_rows_ = rows;
    logical_cols_ = cols;
    row_offset_ = 0;
    col_offset_ = 0;
}

std::unique_ptr<PersistentObject> MatrixBase::clone() const {
    auto out = ObjectFactory::clone_matrix(mapper_, base_rows(), base_cols(), data_type_, matrix_type_, seed_, scalar_, is_transposed_);
    out->set_conjugated(is_conjugated());
    if (auto* m = dynamic_cast<MatrixBase*>(out.get())) {
        m->set_view(logical_rows_, logical_cols_, row_offset_, col_offset_);
    }
    return out;
}

MatrixBase& MatrixBase::operator=(const MatrixBase& other) {
    if (this == &other) return *this;
    return *this = MatrixRefExpression(other);
}

MatrixBase& MatrixBase::operator+=(const MatrixBase& other) {
    return *this = *this + other;
}

MatrixBase& MatrixBase::operator-=(const MatrixBase& other) {
    return *this = *this - other;
}

MatrixBase& MatrixBase::operator*=(double scalar) {
    return *this = *this * scalar;
}

MatrixBase& MatrixBase::operator/=(double scalar) {
    return *this = *this / scalar;
}

void MatrixBase::spill_to_disk(const std::string& filename) {
    if (!mapper_) return;
    if (mapper_->get_filename() != ":memory:") {
        // Already file-backed.
        return;
    }

    // 1. Create new file-backed mapper
    size_t size_bytes = mapper_->get_data_size();
    auto new_mapper = std::make_shared<MemoryMapper>(filename, size_bytes, 0, true);

    // 2. Copy data
    std::memcpy(new_mapper->get_data(), mapper_->get_data(), size_bytes);

    // 3. Flush to ensure data hits disk
    new_mapper->flush();

    // 4. Swap mapper
    mapper_ = new_mapper;
}

} // namespace pycauset

