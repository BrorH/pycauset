#include "pycauset/core/ObjectFactory.hpp"
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
#include "pycauset/core/MatrixTypeResolver.hpp"
#include "pycauset/core/Float16.hpp"
#include <complex>
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
            case DataType::INT8:
                return std::make_unique<IdentityMatrix<int8_t>>(n);
            case DataType::INT16:
                return std::make_unique<IdentityMatrix<int16_t>>(n);
            case DataType::INT32:
                return std::make_unique<IdentityMatrix<int32_t>>(n);
            case DataType::INT64:
                return std::make_unique<IdentityMatrix<int64_t>>(n);
            case DataType::UINT8:
                return std::make_unique<IdentityMatrix<uint8_t>>(n);
            case DataType::UINT16:
                return std::make_unique<IdentityMatrix<uint16_t>>(n);
            case DataType::UINT32:
                return std::make_unique<IdentityMatrix<uint32_t>>(n);
            case DataType::UINT64:
                return std::make_unique<IdentityMatrix<uint64_t>>(n);
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
            case DataType::INT8:
                return std::make_unique<DiagonalMatrix<int8_t>>(n, backing_file);
            case DataType::INT16:
                return std::make_unique<DiagonalMatrix<int16_t>>(n, backing_file);
            case DataType::INT32:
                return std::make_unique<DiagonalMatrix<int32_t>>(n, backing_file);
            case DataType::INT64:
                return std::make_unique<DiagonalMatrix<int64_t>>(n, backing_file);
            case DataType::UINT8:
                return std::make_unique<DiagonalMatrix<uint8_t>>(n, backing_file);
            case DataType::UINT16:
                return std::make_unique<DiagonalMatrix<uint16_t>>(n, backing_file);
            case DataType::UINT32:
                return std::make_unique<DiagonalMatrix<uint32_t>>(n, backing_file);
            case DataType::UINT64:
                return std::make_unique<DiagonalMatrix<uint64_t>>(n, backing_file);
            case DataType::FLOAT64:
                return std::make_unique<DiagonalMatrix<double>>(n, backing_file);
            default:
                throw std::runtime_error("Unsupported DataType for DiagonalMatrix");
        }
    }

    if (mtype == MatrixType::SYMMETRIC || mtype == MatrixType::ANTISYMMETRIC) {
        bool is_anti = (mtype == MatrixType::ANTISYMMETRIC);
        switch (dtype) {
            case DataType::BIT:
                return std::make_unique<SymmetricMatrix<bool>>(n, backing_file, is_anti);
            case DataType::INT8:
                return std::make_unique<SymmetricMatrix<int8_t>>(n, backing_file, is_anti);
            case DataType::INT16:
                return std::make_unique<SymmetricMatrix<int16_t>>(n, backing_file, is_anti);
            case DataType::INT32:
                return std::make_unique<SymmetricMatrix<int32_t>>(n, backing_file, is_anti);
            case DataType::INT64:
                return std::make_unique<SymmetricMatrix<int64_t>>(n, backing_file, is_anti);
            case DataType::UINT8:
                return std::make_unique<SymmetricMatrix<uint8_t>>(n, backing_file, is_anti);
            case DataType::UINT16:
                return std::make_unique<SymmetricMatrix<uint16_t>>(n, backing_file, is_anti);
            case DataType::UINT32:
                return std::make_unique<SymmetricMatrix<uint32_t>>(n, backing_file, is_anti);
            case DataType::UINT64:
                return std::make_unique<SymmetricMatrix<uint64_t>>(n, backing_file, is_anti);
            case DataType::FLOAT64:
                return std::make_unique<SymmetricMatrix<double>>(n, backing_file, is_anti);
            case DataType::FLOAT16:
                return std::make_unique<SymmetricMatrix<float16_t>>(n, backing_file, is_anti);
            case DataType::FLOAT32:
                return std::make_unique<SymmetricMatrix<float>>(n, backing_file, is_anti);
            default:
                throw std::runtime_error("Unsupported DataType for SymmetricMatrix");
        }
    }

    bool is_triangular = (mtype == MatrixType::TRIANGULAR_FLOAT || mtype == MatrixType::CAUSAL);
    
    if (is_triangular) {
        switch (dtype) {
            case DataType::BIT:
                return std::make_unique<TriangularMatrix<bool>>(n, backing_file);
            case DataType::INT8:
                return std::make_unique<TriangularMatrix<int8_t>>(n, backing_file);
            case DataType::INT16:
                return std::make_unique<TriangularMatrix<int16_t>>(n, backing_file);
            case DataType::INT32:
                return std::make_unique<TriangularMatrix<int32_t>>(n, backing_file);
            case DataType::INT64:
                return std::make_unique<TriangularMatrix<int64_t>>(n, backing_file);
            case DataType::UINT8:
                return std::make_unique<TriangularMatrix<uint8_t>>(n, backing_file);
            case DataType::UINT16:
                return std::make_unique<TriangularMatrix<uint16_t>>(n, backing_file);
            case DataType::UINT32:
                return std::make_unique<TriangularMatrix<uint32_t>>(n, backing_file);
            case DataType::UINT64:
                return std::make_unique<TriangularMatrix<uint64_t>>(n, backing_file);
            case DataType::FLOAT64:
                return std::make_unique<TriangularMatrix<double>>(n, backing_file);
            case DataType::FLOAT16:
                return std::make_unique<TriangularMatrix<float16_t>>(n, backing_file);
            case DataType::FLOAT32:
                return std::make_unique<TriangularMatrix<float>>(n, backing_file);
            default:
                throw std::runtime_error("Unsupported DataType for TriangularMatrix");
        }
    } else {
        // Dense (DENSE_FLOAT, INTEGER, etc.)
        switch (dtype) {
            case DataType::BIT:
                return std::make_unique<DenseMatrix<bool>>(n, backing_file);
            case DataType::INT8:
                return std::make_unique<DenseMatrix<int8_t>>(n, backing_file);
            case DataType::INT16:
                return std::make_unique<DenseMatrix<int16_t>>(n, backing_file);
            case DataType::INT32:
                return std::make_unique<DenseMatrix<int32_t>>(n, backing_file);
            case DataType::INT64:
                return std::make_unique<DenseMatrix<int64_t>>(n, backing_file);
            case DataType::UINT8:
                return std::make_unique<DenseMatrix<uint8_t>>(n, backing_file);
            case DataType::UINT16:
                return std::make_unique<DenseMatrix<uint16_t>>(n, backing_file);
            case DataType::UINT32:
                return std::make_unique<DenseMatrix<uint32_t>>(n, backing_file);
            case DataType::UINT64:
                return std::make_unique<DenseMatrix<uint64_t>>(n, backing_file);
            case DataType::FLOAT64:
                return std::make_unique<DenseMatrix<double>>(n, backing_file);
            case DataType::FLOAT16:
                return std::make_unique<DenseMatrix<float16_t>>(n, backing_file);
            case DataType::FLOAT32:
                return std::make_unique<DenseMatrix<float>>(n, backing_file);
            case DataType::COMPLEX_FLOAT16:
                return std::make_unique<ComplexFloat16Matrix>(n, backing_file);
            case DataType::COMPLEX_FLOAT32:
                return std::make_unique<DenseMatrix<std::complex<float>>>(n, backing_file);
            case DataType::COMPLEX_FLOAT64:
                return std::make_unique<DenseMatrix<std::complex<double>>>(n, backing_file);
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
    std::complex<double> scalar,
    bool is_transposed
) {
    uint64_t n = rows; // Assuming square for now, or handled by specific classes

    if (mtype == MatrixType::IDENTITY) {
        switch (dtype) {
            case DataType::BIT:
                return std::make_unique<IdentityMatrix<bool>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::INT16:
                return std::make_unique<IdentityMatrix<int16_t>>(n, backing_file, offset, seed, scalar, is_transposed);
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
            case DataType::INT16:
                return std::make_unique<DiagonalMatrix<int16_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::INT32:
                return std::make_unique<DiagonalMatrix<int32_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::FLOAT64:
                return std::make_unique<DiagonalMatrix<double>>(n, backing_file, offset, seed, scalar, is_transposed);
            default:
                throw std::runtime_error("Unsupported DataType for DiagonalMatrix load");
        }
    }

    if (mtype == MatrixType::SYMMETRIC || mtype == MatrixType::ANTISYMMETRIC) {
        bool is_anti = (mtype == MatrixType::ANTISYMMETRIC);
        switch (dtype) {
            case DataType::BIT:
                return std::make_unique<SymmetricMatrix<bool>>(n, backing_file, offset, seed, scalar, is_transposed, is_anti);
            case DataType::INT16:
                return std::make_unique<SymmetricMatrix<int16_t>>(n, backing_file, offset, seed, scalar, is_transposed, is_anti);
            case DataType::INT32:
                return std::make_unique<SymmetricMatrix<int32_t>>(n, backing_file, offset, seed, scalar, is_transposed, is_anti);
            case DataType::FLOAT64:
                return std::make_unique<SymmetricMatrix<double>>(n, backing_file, offset, seed, scalar, is_transposed, is_anti);
            case DataType::FLOAT16:
                return std::make_unique<SymmetricMatrix<float16_t>>(n, backing_file, offset, seed, scalar, is_transposed, is_anti);
            case DataType::FLOAT32:
                return std::make_unique<SymmetricMatrix<float>>(n, backing_file, offset, seed, scalar, is_transposed, is_anti);
            default:
                throw std::runtime_error("Unsupported DataType for SymmetricMatrix load");
        }
    }

    bool is_triangular = (mtype == MatrixType::TRIANGULAR_FLOAT || mtype == MatrixType::CAUSAL);
    
    if (is_triangular) {
        switch (dtype) {
            case DataType::BIT:
                return std::make_unique<TriangularMatrix<bool>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::INT16:
                return std::make_unique<TriangularMatrix<int16_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::INT32:
                return std::make_unique<TriangularMatrix<int32_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::FLOAT64:
                return std::make_unique<TriangularMatrix<double>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::FLOAT16:
                return std::make_unique<TriangularMatrix<float16_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::FLOAT32:
                return std::make_unique<TriangularMatrix<float>>(n, backing_file, offset, seed, scalar, is_transposed);
            default:
                throw std::runtime_error("Unsupported DataType for TriangularMatrix load");
        }
    } else {
        switch (dtype) {
            case DataType::BIT:
                return std::make_unique<DenseMatrix<bool>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::INT16:
                return std::make_unique<DenseMatrix<int16_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::INT32:
                return std::make_unique<DenseMatrix<int32_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::FLOAT64:
                return std::make_unique<DenseMatrix<double>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::FLOAT16:
                return std::make_unique<DenseMatrix<float16_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::FLOAT32:
                return std::make_unique<DenseMatrix<float>>(n, backing_file, offset, seed, scalar, is_transposed);

            case DataType::COMPLEX_FLOAT16:
                return std::make_unique<ComplexFloat16Matrix>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::COMPLEX_FLOAT32:
                return std::make_unique<DenseMatrix<std::complex<float>>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::COMPLEX_FLOAT64:
                return std::make_unique<DenseMatrix<std::complex<double>>>(n, backing_file, offset, seed, scalar, is_transposed);

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
                return std::make_unique<DenseVector<float16_t>>(n, backing_file);
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
    std::complex<double> scalar,
    bool is_transposed
) {
    uint64_t n = (rows > cols) ? rows : cols;

    if (mtype == MatrixType::VECTOR) {
        switch (dtype) {
            case DataType::BIT:
                return std::make_unique<DenseVector<bool>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::INT8:
                return std::make_unique<DenseVector<int8_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::INT16:
                return std::make_unique<DenseVector<int16_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::INT32:
                return std::make_unique<DenseVector<int32_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::INT64:
                return std::make_unique<DenseVector<int64_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::UINT8:
                return std::make_unique<DenseVector<uint8_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::UINT16:
                return std::make_unique<DenseVector<uint16_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::UINT32:
                return std::make_unique<DenseVector<uint32_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::UINT64:
                return std::make_unique<DenseVector<uint64_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::FLOAT16:
                return std::make_unique<DenseVector<float16_t>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::FLOAT32:
                return std::make_unique<DenseVector<float>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::FLOAT64:
                return std::make_unique<DenseVector<double>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::COMPLEX_FLOAT16:
                return std::make_unique<ComplexFloat16Vector>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::COMPLEX_FLOAT32:
                return std::make_unique<DenseVector<std::complex<float>>>(n, backing_file, offset, seed, scalar, is_transposed);
            case DataType::COMPLEX_FLOAT64:
                return std::make_unique<DenseVector<std::complex<double>>>(n, backing_file, offset, seed, scalar, is_transposed);
            default:
                throw std::runtime_error("Unsupported DataType for DenseVector load");
        }
    }

    if (mtype == MatrixType::UNIT_VECTOR) {
        return std::make_unique<UnitVector>(n, seed, backing_file, offset, seed, scalar, is_transposed);
    }

    throw std::runtime_error("Unknown Vector Type for load");
}

std::unique_ptr<MatrixBase> ObjectFactory::clone_matrix(
    std::shared_ptr<MemoryMapper> mapper,
    uint64_t rows,
    uint64_t cols,
    DataType dtype,
    MatrixType mtype,
    uint64_t seed,
    std::complex<double> scalar,
    bool is_transposed
) {
    uint64_t n = rows;
    std::unique_ptr<MatrixBase> mat;

    if (mtype == MatrixType::IDENTITY) {
        switch (dtype) {
            case DataType::BIT: mat = std::make_unique<IdentityMatrix<bool>>(n, mapper); break;
            case DataType::INT8: mat = std::make_unique<IdentityMatrix<int8_t>>(n, mapper); break;
            case DataType::INT16: mat = std::make_unique<IdentityMatrix<int16_t>>(n, mapper); break;
            case DataType::INT32: mat = std::make_unique<IdentityMatrix<int32_t>>(n, mapper); break;
            case DataType::INT64: mat = std::make_unique<IdentityMatrix<int64_t>>(n, mapper); break;
            case DataType::UINT8: mat = std::make_unique<IdentityMatrix<uint8_t>>(n, mapper); break;
            case DataType::UINT16: mat = std::make_unique<IdentityMatrix<uint16_t>>(n, mapper); break;
            case DataType::UINT32: mat = std::make_unique<IdentityMatrix<uint32_t>>(n, mapper); break;
            case DataType::UINT64: mat = std::make_unique<IdentityMatrix<uint64_t>>(n, mapper); break;
            case DataType::FLOAT64: mat = std::make_unique<IdentityMatrix<double>>(n, mapper); break;
            default: throw std::runtime_error("Unsupported DataType for IdentityMatrix clone");
        }
    } else if (mtype == MatrixType::DIAGONAL) {
        switch (dtype) {
            case DataType::BIT: mat = std::make_unique<DiagonalMatrix<bool>>(n, mapper); break;
            case DataType::INT8: mat = std::make_unique<DiagonalMatrix<int8_t>>(n, mapper); break;
            case DataType::INT16: mat = std::make_unique<DiagonalMatrix<int16_t>>(n, mapper); break;
            case DataType::INT32: mat = std::make_unique<DiagonalMatrix<int32_t>>(n, mapper); break;
            case DataType::INT64: mat = std::make_unique<DiagonalMatrix<int64_t>>(n, mapper); break;
            case DataType::UINT8: mat = std::make_unique<DiagonalMatrix<uint8_t>>(n, mapper); break;
            case DataType::UINT16: mat = std::make_unique<DiagonalMatrix<uint16_t>>(n, mapper); break;
            case DataType::UINT32: mat = std::make_unique<DiagonalMatrix<uint32_t>>(n, mapper); break;
            case DataType::UINT64: mat = std::make_unique<DiagonalMatrix<uint64_t>>(n, mapper); break;
            case DataType::FLOAT64: mat = std::make_unique<DiagonalMatrix<double>>(n, mapper); break;
            default: throw std::runtime_error("Unsupported DataType for DiagonalMatrix clone");
        }
    } else if (mtype == MatrixType::TRIANGULAR_FLOAT || mtype == MatrixType::CAUSAL) {
        switch (dtype) {
            case DataType::BIT: mat = std::make_unique<TriangularMatrix<bool>>(n, mapper); break;
            case DataType::INT8: mat = std::make_unique<TriangularMatrix<int8_t>>(n, mapper); break;
            case DataType::INT16: mat = std::make_unique<TriangularMatrix<int16_t>>(n, mapper); break;
            case DataType::INT32: mat = std::make_unique<TriangularMatrix<int32_t>>(n, mapper); break;
            case DataType::INT64: mat = std::make_unique<TriangularMatrix<int64_t>>(n, mapper); break;
            case DataType::UINT8: mat = std::make_unique<TriangularMatrix<uint8_t>>(n, mapper); break;
            case DataType::UINT16: mat = std::make_unique<TriangularMatrix<uint16_t>>(n, mapper); break;
            case DataType::UINT32: mat = std::make_unique<TriangularMatrix<uint32_t>>(n, mapper); break;
            case DataType::UINT64: mat = std::make_unique<TriangularMatrix<uint64_t>>(n, mapper); break;
            case DataType::FLOAT64: mat = std::make_unique<TriangularMatrix<double>>(n, mapper); break;
            case DataType::FLOAT16: mat = std::make_unique<TriangularMatrix<float16_t>>(n, mapper); break;
            case DataType::FLOAT32: mat = std::make_unique<TriangularMatrix<float>>(n, mapper); break;
            default: throw std::runtime_error("Unsupported DataType for TriangularMatrix clone");
        }
    } else if (mtype == MatrixType::SYMMETRIC || mtype == MatrixType::ANTISYMMETRIC) {
        bool is_anti = (mtype == MatrixType::ANTISYMMETRIC);
        switch (dtype) {
            case DataType::BIT: mat = std::make_unique<SymmetricMatrix<bool>>(n, mapper, is_anti); break;
            case DataType::INT8: mat = std::make_unique<SymmetricMatrix<int8_t>>(n, mapper, is_anti); break;
            case DataType::INT16: mat = std::make_unique<SymmetricMatrix<int16_t>>(n, mapper, is_anti); break;
            case DataType::INT32: mat = std::make_unique<SymmetricMatrix<int32_t>>(n, mapper, is_anti); break;
            case DataType::INT64: mat = std::make_unique<SymmetricMatrix<int64_t>>(n, mapper, is_anti); break;
            case DataType::UINT8: mat = std::make_unique<SymmetricMatrix<uint8_t>>(n, mapper, is_anti); break;
            case DataType::UINT16: mat = std::make_unique<SymmetricMatrix<uint16_t>>(n, mapper, is_anti); break;
            case DataType::UINT32: mat = std::make_unique<SymmetricMatrix<uint32_t>>(n, mapper, is_anti); break;
            case DataType::UINT64: mat = std::make_unique<SymmetricMatrix<uint64_t>>(n, mapper, is_anti); break;
            case DataType::FLOAT64: mat = std::make_unique<SymmetricMatrix<double>>(n, mapper, is_anti); break;
            case DataType::FLOAT16: mat = std::make_unique<SymmetricMatrix<float16_t>>(n, mapper, is_anti); break;
            case DataType::FLOAT32: mat = std::make_unique<SymmetricMatrix<float>>(n, mapper, is_anti); break;
            default: throw std::runtime_error("Unsupported DataType for SymmetricMatrix clone");
        }
    } else {
        switch (dtype) {
            case DataType::BIT: mat = std::make_unique<DenseMatrix<bool>>(n, mapper); break;
            case DataType::INT8: mat = std::make_unique<DenseMatrix<int8_t>>(n, mapper); break;
            case DataType::INT16: mat = std::make_unique<DenseMatrix<int16_t>>(n, mapper); break;
            case DataType::INT32: mat = std::make_unique<DenseMatrix<int32_t>>(n, mapper); break;
            case DataType::INT64: mat = std::make_unique<DenseMatrix<int64_t>>(n, mapper); break;
            case DataType::UINT8: mat = std::make_unique<DenseMatrix<uint8_t>>(n, mapper); break;
            case DataType::UINT16: mat = std::make_unique<DenseMatrix<uint16_t>>(n, mapper); break;
            case DataType::UINT32: mat = std::make_unique<DenseMatrix<uint32_t>>(n, mapper); break;
            case DataType::UINT64: mat = std::make_unique<DenseMatrix<uint64_t>>(n, mapper); break;
            case DataType::FLOAT64: mat = std::make_unique<DenseMatrix<double>>(n, mapper); break;
            case DataType::FLOAT16: mat = std::make_unique<DenseMatrix<float16_t>>(n, mapper); break;
            case DataType::FLOAT32: mat = std::make_unique<DenseMatrix<float>>(n, mapper); break;
            case DataType::COMPLEX_FLOAT16: mat = std::make_unique<ComplexFloat16Matrix>(n, mapper); break;
            case DataType::COMPLEX_FLOAT32: mat = std::make_unique<DenseMatrix<std::complex<float>>>(n, mapper); break;
            case DataType::COMPLEX_FLOAT64: mat = std::make_unique<DenseMatrix<std::complex<double>>>(n, mapper); break;
            default: throw std::runtime_error("Unsupported DataType for DenseMatrix clone");
        }
    }

    if (mat) {
        mat->set_seed(seed);
        mat->set_scalar(scalar);
        mat->set_transposed(is_transposed);
    }
    return mat;
}

std::unique_ptr<VectorBase> ObjectFactory::clone_vector(
    std::shared_ptr<MemoryMapper> mapper,
    uint64_t rows,
    uint64_t cols,
    DataType dtype,
    MatrixType mtype,
    uint64_t seed,
    std::complex<double> scalar,
    bool is_transposed
) {
    uint64_t n = rows;
    std::unique_ptr<VectorBase> vec;

    if (mtype == MatrixType::UNIT_VECTOR) {
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
            case DataType::FLOAT16: vec = std::make_unique<DenseVector<float16_t>>(n, mapper); break;
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
