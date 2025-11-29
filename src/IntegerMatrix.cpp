#include "IntegerMatrix.hpp"

#include <stdexcept>

IntegerMatrix::IntegerMatrix(uint64_t n, const std::string& backingFile) : MatrixBase(n) {
    calculate_offsets();
    
    // Calculate total size
    uint64_t currentOffset = 0;
    for (uint64_t i = 0; i < n_; ++i) {
        uint64_t rowLen = (n_ - 1) - i;
        if (rowLen > 0) {
            currentOffset += rowLen * sizeof(uint32_t);
        }
    }
    uint64_t sizeInBytes = currentOffset;
    initialize_storage(sizeInBytes, backingFile, "integer_matrix", sizeof(uint32_t),
                      pycauset::MatrixType::INTEGER, pycauset::DataType::INT32);
}

IntegerMatrix::IntegerMatrix(uint64_t n, std::unique_ptr<MemoryMapper> mapper) 
    : MatrixBase(n, std::move(mapper)) {
    calculate_offsets();
}

void IntegerMatrix::calculate_offsets() {
    row_offsets_.resize(n_);
    uint64_t currentOffset = 0;
    for (uint64_t i = 0; i < n_; ++i) {
        row_offsets_[i] = currentOffset;
        uint64_t rowLen = (n_ - 1) - i;
        if (rowLen > 0) {
            currentOffset += rowLen * sizeof(uint32_t);
        }
    }
}

void IntegerMatrix::set(uint64_t i, uint64_t j, uint32_t value) {
    if (i >= j) throw std::invalid_argument("Strictly upper triangular");
    if (j >= n_) throw std::out_of_range("Index out of bounds");

    uint64_t colIndex = j - (i + 1);
    uint64_t byteOffset = row_offsets_[i] + colIndex * sizeof(uint32_t);
    
    auto* basePtr = static_cast<char*>(require_mapper()->get_data());
    uint32_t* dataPtr = reinterpret_cast<uint32_t*>(basePtr + byteOffset);
    *dataPtr = value;
}

uint32_t IntegerMatrix::get(uint64_t i, uint64_t j) const {
    if (i >= j || j >= n_) return 0;
    
    uint64_t colIndex = j - (i + 1);
    uint64_t byteOffset = row_offsets_[i] + colIndex * sizeof(uint32_t);
    
    auto* basePtr = static_cast<const char*>(require_mapper()->get_data());
    const uint32_t* dataPtr = reinterpret_cast<const uint32_t*>(basePtr + byteOffset);
    return *dataPtr;
}
