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
    uint64_t rows,
    uint64_t cols,
    DataType dtype,
    Structure mtype,
    const std::string& backing_file,
    const MatrixOptions& opts
) {
    // Which (structure, dtype) pairs exist is decided by the catalogue (DTypeDispatch.hpp).
    return make_matrix(mtype, dtype, rows, cols, backing_file, opts);
}

// --- Vector Creation ---

std::unique_ptr<VectorBase> ObjectFactory::create_vector(
    uint64_t n, 
    DataType dtype, 
    Structure mtype, 
    const std::string& backing_file
) {
    if (mtype == Structure::VECTOR) {
        // The catalogue's dtype -> class map (DenseVectorOf), as for matrices (review 3.3 C-12).
        return visit_dtype(dtype, [&](auto tag) -> std::unique_ptr<VectorBase> {
            using V = typename DenseVectorOf<typename decltype(tag)::type>::type;
            return std::make_unique<V>(n, backing_file);
        });
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
    bool is_transposed,
    bool has_diagonal
) {
    MatrixOptions opts;
    opts.has_diagonal = has_diagonal;
    auto mat = adopt_matrix(mtype, dtype, std::move(mapper), rows, cols, opts);
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
        vec = std::make_unique<UnitVector>(n, mapper, dtype);
    } else {
        vec = visit_dtype(dtype, [&](auto tag) -> std::unique_ptr<VectorBase> {
            using V = typename DenseVectorOf<typename decltype(tag)::type>::type;
            return std::make_unique<V>(n, mapper);
        });
    }

    if (vec) {
        vec->set_seed(seed);
        vec->set_scalar(scalar);
        vec->set_transposed(is_transposed);
    }
    return vec;
}

}
