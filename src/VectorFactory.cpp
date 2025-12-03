#include "VectorFactory.hpp"
#include "DenseVector.hpp"
#include "UnitVector.hpp"
#include <stdexcept>

namespace pycauset {

std::unique_ptr<VectorBase> VectorFactory::create(
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
    
    // UnitVector creation usually requires an index, which isn't in this signature.
    // This create method might need to be overloaded or UnitVector created directly.
    // For now, we'll throw if someone tries to create a generic UnitVector without index.
    if (mtype == MatrixType::UNIT_VECTOR) {
        throw std::runtime_error("UnitVector requires an active index for creation");
    }

    throw std::runtime_error("Unknown Vector Type");
}

std::unique_ptr<VectorBase> VectorFactory::load(
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
    uint64_t n = (rows > cols) ? rows : cols; // Assuming vector is N x 1 or 1 x N

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
        // For UnitVector, the 'seed' field is repurposed to store the active_index
        // This is a bit of a hack, but efficient.
        return std::make_unique<UnitVector>(n, seed, backing_file, offset, seed, scalar, is_transposed);
    }

    throw std::runtime_error("Unknown Vector Type for load");
}

}
