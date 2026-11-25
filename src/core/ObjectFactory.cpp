#include "pycauset/core/ObjectFactory.hpp"
#include "pycauset/core/DTypeDispatch.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/matrix/IdentityMatrix.hpp"
#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/matrix/TriangularBitMatrix.hpp"
#include "pycauset/matrix/SymmetricMatrix.hpp"
#include "pycauset/matrix/ComplexFloat16Matrix.hpp"
#include "pycauset/vector/DenseVector.hpp"
#include "pycauset/vector/ComplexFloat16Vector.hpp"
#include "pycauset/vector/UnitVector.hpp"
#include "pycauset/core/Float16.hpp"
#include <complex>
#include <stdexcept>

namespace pycauset {

// --- Matrix Creation ---

std::unique_ptr<MatrixBase> ObjectFactory::create_matrix(
    uint64_t n, 
    DataType dtype, 
    Structure mtype, 
    const std::string& backing_file
) {
    return create_matrix(n, n, dtype, mtype, backing_file);
}

std::unique_ptr<MatrixBase> ObjectFactory::create_matrix(
    uint64_t rows,
    uint64_t cols,
    DataType dtype,
    Structure mtype,
    const std::string& backing_file
) {
    // Which (structure, dtype) pairs exist is decided by the catalogue (DTypeDispatch.hpp).
    return make_matrix(mtype, dtype, rows, cols, backing_file);
}

// --- Vector Creation ---

std::unique_ptr<VectorBase> ObjectFactory::create_vector(
    uint64_t n, 
    DataType dtype, 
    Structure mtype, 
    const std::string& backing_file
) {
    if (mtype == Structure::VECTOR) {
        switch (dtype) {
            case DataType::BIT:
                return std::make_unique<DenseVector<bool>>(n, backing_file);
            case DataType::INT8:
                return std::make_unique<DenseVector<int8_t>>(n, backing_file);
            case DataType::INT16:
                return std::make_unique<DenseVector<int16_t>>(n, backing_file);
            case DataType::INT32:
                return std::make_unique<DenseVector<int32_t>>(n, backing_file);
            case DataType::INT64:
                return std::make_unique<DenseVector<int64_t>>(n, backing_file);
            case DataType::UINT8:
                return std::make_unique<DenseVector<uint8_t>>(n, backing_file);
            case DataType::UINT16:
                return std::make_unique<DenseVector<uint16_t>>(n, backing_file);
            case DataType::UINT32:
                return std::make_unique<DenseVector<uint32_t>>(n, backing_file);
            case DataType::UINT64:
                return std::make_unique<DenseVector<uint64_t>>(n, backing_file);
            case DataType::FLOAT16:
                return std::make_unique<DenseVector<pycauset::float16_t>>(n, backing_file);
            case DataType::FLOAT32:
                return std::make_unique<DenseVector<float>>(n, backing_file);
            case DataType::FLOAT64:
                return std::make_unique<DenseVector<double>>(n, backing_file);
            case DataType::COMPLEX_FLOAT16:
                return std::make_unique<ComplexFloat16Vector>(n, backing_file);
            case DataType::COMPLEX_FLOAT32:
                return std::make_unique<DenseVector<std::complex<float>>>(n, backing_file);
            case DataType::COMPLEX_FLOAT64:
                return std::make_unique<DenseVector<std::complex<double>>>(n, backing_file);
            default:
                throw std::runtime_error("Unsupported DataType for DenseVector");
        }
    }
    
    if (mtype == Structure::UNIT_VECTOR) {
        throw std::runtime_error("UnitVector requires an active index for creation");
    }

    throw std::runtime_error("Unknown Vector Type");
}

std::unique_ptr<MatrixBase> ObjectFactory::clone_matrix(
    std::shared_ptr<MemoryMapper> mapper,
    uint64_t rows,
    uint64_t cols,
    DataType dtype,
    Structure mtype,
    uint64_t seed,
    std::complex<double> scalar,
    bool is_transposed
) {
    // Classes whose storage depends on more than (structure, dtype) (triangular has_diagonal,
    // the symmetric kind) override clone() themselves; this serves the others.
    auto mat = adopt_matrix(mtype, dtype, std::move(mapper), rows, cols);
    mat->set_seed(seed);
    mat->set_scalar(scalar);
    mat->set_transposed(is_transposed);
    return mat;
}

std::unique_ptr<VectorBase> ObjectFactory::clone_vector(
    std::shared_ptr<MemoryMapper> mapper,
    uint64_t rows,
    uint64_t cols,
    DataType dtype,
    Structure mtype,
    uint64_t seed,
    std::complex<double> scalar,
    bool is_transposed
) {
    uint64_t n = rows;
    std::unique_ptr<VectorBase> vec;

    if (mtype == Structure::UNIT_VECTOR) {
        vec = std::make_unique<UnitVector>(n, mapper);
    } else {
        switch (dtype) {
            case DataType::BIT: vec = std::make_unique<DenseVector<bool>>(n, mapper); break;
            case DataType::INT8: vec = std::make_unique<DenseVector<int8_t>>(n, mapper); break;
            case DataType::INT16: vec = std::make_unique<DenseVector<int16_t>>(n, mapper); break;
            case DataType::INT32: vec = std::make_unique<DenseVector<int32_t>>(n, mapper); break;
            case DataType::INT64: vec = std::make_unique<DenseVector<int64_t>>(n, mapper); break;
            case DataType::UINT8: vec = std::make_unique<DenseVector<uint8_t>>(n, mapper); break;
            case DataType::UINT16: vec = std::make_unique<DenseVector<uint16_t>>(n, mapper); break;
            case DataType::UINT32: vec = std::make_unique<DenseVector<uint32_t>>(n, mapper); break;
            case DataType::UINT64: vec = std::make_unique<DenseVector<uint64_t>>(n, mapper); break;
            case DataType::FLOAT16: vec = std::make_unique<DenseVector<pycauset::float16_t>>(n, mapper); break;
            case DataType::FLOAT64: vec = std::make_unique<DenseVector<double>>(n, mapper); break;
            case DataType::FLOAT32: vec = std::make_unique<DenseVector<float>>(n, mapper); break;
            case DataType::COMPLEX_FLOAT16: vec = std::make_unique<ComplexFloat16Vector>(n, mapper); break;
            case DataType::COMPLEX_FLOAT32: vec = std::make_unique<DenseVector<std::complex<float>>>(n, mapper); break;
            case DataType::COMPLEX_FLOAT64: vec = std::make_unique<DenseVector<std::complex<double>>>(n, mapper); break;
            default: throw std::runtime_error("Unsupported DataType for DenseVector clone");
        }
    }

    if (vec) {
        vec->set_seed(seed);
        vec->set_scalar(scalar);
        vec->set_transposed(is_transposed);
    }
    return vec;
}

}
