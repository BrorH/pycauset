#include "pycauset/core/ObjectFactory.hpp"
#include "pycauset/matrix/DenseMatrix.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/matrix/IdentityMatrix.hpp"
#include "pycauset/matrix/DiagonalMatrix.hpp"
#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/matrix/TriangularBitMatrix.hpp"
#include "pycauset/vector/DenseVector.hpp"
#include "pycauset/vector/UnitVector.hpp"
#include "pycauset/core/Float16.hpp"
#include <stdexcept>

namespace pycauset {

// --- Matrix Creation ---

std::unique_ptr<MatrixBase> ObjectFactory::create_matrix(
    uint64_t n, 
    DataType dtype, 
    MatrixType mtype, 
    const std::string& backing_file
) {
    if (mtype == MatrixType::IDENTITY) {
        switch (dtype) {
            case DataType::BIT:
                return std::make_unique<IdentityMatrix<bool>>(n);
            case DataType::INT32:
                return std::make_unique<IdentityMatrix<int32_t>>(n);
            case DataType::FLOAT64:
                return std::make_unique<IdentityMatrix<double>>(n);
            default:
                throw std::runtime_error("Unsupported DataType for IdentityMatrix");
        }
    }

    if (mtype == MatrixType::DIAGONAL) {
        switch (dtype) {
            case DataType::BIT:
                return std::make_unique<DiagonalMatrix<bool>>(n, backing_file);
            case DataType::INT32:
                return std::make_unique<DiagonalMatrix<int32_t>>(n, backing_file);
            case DataType::FLOAT64:
                return std::make_unique<DiagonalMatrix<double>>(n, backing_file);
            default:
                throw std::runtime_error("Unsupported DataType for DiagonalMatrix");
        }
    }

    bool is_triangular = (mtype == MatrixType::TRIANGULAR_FLOAT || mtype == MatrixType::CAUSAL);
    
    if (is_triangular) {
        switch (dtype) {
            case DataType::BIT:
                return std::make_unique<TriangularMatrix<bool>>(n, backing_file);
            case DataType::INT32:
                return std::make_unique<TriangularMatrix<int32_t>>(n, backing_file);
            case DataType::FLOAT64:
                return std::make_unique<TriangularMatrix<double>>(n, backing_file);
            default:
                throw std::runtime_error("Unsupported DataType for TriangularMatrix");
        }
    } else {
        // Dense (DENSE_FLOAT, INTEGER, etc.)
        switch (dtype) {
            case DataType::BIT:
                return std::make_unique<DenseMatrix<bool>>(n, backing_file);
            case DataType::INT32:
                return std::make_unique<DenseMatrix<int32_t>>(n, backing_file);
            case DataType::FLOAT64:
                return std::make_unique<DenseMatrix<double>>(n, backing_file);
            case DataType::FLOAT32:
                return std::make_unique<DenseMatrix<float>>(n, backing_file);
            case DataType::FLOAT16:
                return std::make_unique<DenseMatrix<pycauset::Float16>>(n, backing_file);
            default:
                throw std::runtime_error("Unsupported DataType for DenseMatrix");
        }
    }
}

std::unique_ptr<MatrixBase> ObjectFactory::load_matrix(
    const std::string& backing_file,
    size_t offset,
    uint64_t rows,
    uint64_t cols,
    DataType dtype,
    MatrixType mtype,
    uint64_t seed,
    double scalar,
    bool is_transposed
) {
    uint64_t n = rows; // Assuming square for now, or handled by specific classes

    if (mtype == MatrixType::IDENTITY) {
        switch (dtype) {
            case DataType::BIT:
                return std::make_unique<IdentityMatrix<bool>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::INT32:
                return std::make_unique<IdentityMatrix<int32_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::FLOAT64:
                return std::make_unique<IdentityMatrix<double>>(n, backing_file, offset, seed, scalar, is_transposed);
            default:
                throw std::runtime_error("Unsupported DataType for IdentityMatrix load");
        }
    }

    if (mtype == MatrixType::DIAGONAL) {
        switch (dtype) {
            case DataType::BIT:
                return std::make_unique<DiagonalMatrix<bool>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::INT32:
                return std::make_unique<DiagonalMatrix<int32_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::FLOAT64:
                return std::make_unique<DiagonalMatrix<double>>(n, backing_file, offset, seed, scalar, is_transposed);
            default:
                throw std::runtime_error("Unsupported DataType for DiagonalMatrix load");
        }
    }

    bool is_triangular = (mtype == MatrixType::TRIANGULAR_FLOAT || mtype == MatrixType::CAUSAL);
    
    if (is_triangular) {
        switch (dtype) {
            case DataType::BIT:
                return std::make_unique<TriangularMatrix<bool>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::INT32:
                return std::make_unique<TriangularMatrix<int32_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::FLOAT64:
                return std::make_unique<TriangularMatrix<double>>(n, backing_file, offset, seed, scalar, is_transposed);
            default:
                throw std::runtime_error("Unsupported DataType for TriangularMatrix load");
        }
    } else {
        switch (dtype) {
            case DataType::BIT:
                return std::make_unique<DenseMatrix<bool>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::INT32:
                return std::make_unique<DenseMatrix<int32_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::FLOAT64:
                return std::make_unique<DenseMatrix<double>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::FLOAT32:
                return std::make_unique<DenseMatrix<float>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::FLOAT16:
                return std::make_unique<DenseMatrix<pycauset::Float16>>(n, backing_file, offset, seed, scalar, is_transposed);
            default:
                throw std::runtime_error("Unsupported DataType for DenseMatrix load");
        }
    }
}

// --- Vector Creation ---

std::unique_ptr<VectorBase> ObjectFactory::create_vector(
    uint64_t n, 
    DataType dtype, 
    MatrixType mtype, 
    const std::string& backing_file
) {
    if (mtype == MatrixType::VECTOR) {
        switch (dtype) {
            case DataType::BIT:
                return std::make_unique<DenseVector<bool>>(n, backing_file);
            case DataType::INT32:
                return std::make_unique<DenseVector<int32_t>>(n, backing_file);
            case DataType::FLOAT64:
                return std::make_unique<DenseVector<double>>(n, backing_file);
            default:
                throw std::runtime_error("Unsupported DataType for DenseVector");
        }
    }
    
    if (mtype == MatrixType::UNIT_VECTOR) {
        throw std::runtime_error("UnitVector requires an active index for creation");
    }

    throw std::runtime_error("Unknown Vector Type");
}

std::unique_ptr<VectorBase> ObjectFactory::load_vector(
    const std::string& backing_file,
    size_t offset,
    uint64_t rows,
    uint64_t cols,
    DataType dtype,
    MatrixType mtype,
    uint64_t seed,
    double scalar,
    bool is_transposed
) {
    uint64_t n = (rows > cols) ? rows : cols;

    if (mtype == MatrixType::VECTOR) {
        switch (dtype) {
            case DataType::BIT:
                return std::make_unique<DenseVector<bool>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::INT32:
                return std::make_unique<DenseVector<int32_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::FLOAT64:
                return std::make_unique<DenseVector<double>>(n, backing_file, offset, seed, scalar, is_transposed);
            default:
                throw std::runtime_error("Unsupported DataType for DenseVector load");
        }
    }

    if (mtype == MatrixType::UNIT_VECTOR) {
        return std::make_unique<UnitVector>(n, seed, backing_file, offset, seed, scalar, is_transposed);
    }

    throw std::runtime_error("Unknown Vector Type for load");
}

// --- Type Resolution ---

DataType ObjectFactory::resolve_result_type(DataType a, DataType b) {
    if (a == DataType::FLOAT64 || b == DataType::FLOAT64) return DataType::FLOAT64;
    if (a == DataType::FLOAT32 || b == DataType::FLOAT32) return DataType::FLOAT32;
    if (a == DataType::FLOAT16 || b == DataType::FLOAT16) return DataType::FLOAT16;
    if (a == DataType::INT32 || b == DataType::INT32) return DataType::INT32;
    return DataType::BIT;
}

MatrixType ObjectFactory::resolve_result_matrix_type(MatrixType a, MatrixType b) {
    if (a == MatrixType::DENSE_FLOAT || b == MatrixType::DENSE_FLOAT) return MatrixType::DENSE_FLOAT;
    if (a == MatrixType::TRIANGULAR_FLOAT && b == MatrixType::TRIANGULAR_FLOAT) return MatrixType::TRIANGULAR_FLOAT;
    if (a == MatrixType::CAUSAL && b == MatrixType::CAUSAL) return MatrixType::CAUSAL;
    return MatrixType::DENSE_FLOAT; // Default fallback
}

}
