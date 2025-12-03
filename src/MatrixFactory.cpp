#include "MatrixFactory.hpp"
#include "DenseMatrix.hpp"
#include "TriangularMatrix.hpp"
#include "IdentityMatrix.hpp"
#include "DiagonalMatrix.hpp"
#include "DenseBitMatrix.hpp"
#include "TriangularBitMatrix.hpp"
#include <stdexcept>

namespace pycauset {

std::unique_ptr<MatrixBase> MatrixFactory::create(
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

// New load function that accepts explicit metadata
std::unique_ptr<MatrixBase> MatrixFactory::load(
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
    if (rows != cols) {
        // Currently only square matrices supported in this factory path
        // Vectors handled separately or need extension
    }
    uint64_t n = rows;

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

DataType MatrixFactory::resolve_result_type(DataType a, DataType b) {
    if (a == DataType::FLOAT64 || b == DataType::FLOAT64) {
        return DataType::FLOAT64;
    }
    if (a == DataType::INT32 || b == DataType::INT32) {
        return DataType::INT32;
    }
    return DataType::BIT;
}

MatrixType MatrixFactory::resolve_result_matrix_type(MatrixType a, MatrixType b) {
    // If either is Dense, result is Dense
    if (a == MatrixType::DENSE_FLOAT || b == MatrixType::DENSE_FLOAT ||
        a == MatrixType::INTEGER || b == MatrixType::INTEGER) {
        return MatrixType::DENSE_FLOAT; // Or generic Dense
    }
    
    // If both are Diagonal, result is Diagonal
    if (a == MatrixType::DIAGONAL && b == MatrixType::DIAGONAL) {
        return MatrixType::DIAGONAL;
    }

    // If one is Diagonal and other is Triangular, result is Triangular
    // (Diag + Tri = Tri)
    if ((a == MatrixType::DIAGONAL && (b == MatrixType::TRIANGULAR_FLOAT || b == MatrixType::CAUSAL)) ||
        (b == MatrixType::DIAGONAL && (a == MatrixType::TRIANGULAR_FLOAT || a == MatrixType::CAUSAL))) {
        return MatrixType::TRIANGULAR_FLOAT; // Or generic Triangular
    }

    // Identity behaves like Diagonal for structure
    if (a == MatrixType::IDENTITY && b == MatrixType::IDENTITY) {
        return MatrixType::IDENTITY;
    }
    if (a == MatrixType::IDENTITY) return b;
    if (b == MatrixType::IDENTITY) return a;

    // Default fallback
    return MatrixType::DENSE_FLOAT;
}

}
