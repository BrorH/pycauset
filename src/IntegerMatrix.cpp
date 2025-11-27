#include "IntegerMatrix.hpp"
#include <stdexcept>

IntegerMatrix::IntegerMatrix(uint64_t N, const std::string& backingFile) : N_(N) {
    calculateOffsets();
    
    // Calculate total size
    uint64_t currentOffset = 0;
    for (uint64_t i = 0; i < N; ++i) {
        uint64_t rowLen = (N - 1) - i;
        if (rowLen > 0) {
            currentOffset += rowLen * sizeof(uint32_t);
        }
    }
    uint64_t sizeInBytes = currentOffset;
    if (sizeInBytes == 0) sizeInBytes = 4;

    std::string path = backingFile;
    if (path.empty()) {
        path = "temp_int_matrix.bin";
    }

    mapper_ = std::make_unique<MemoryMapper>(path, sizeInBytes);
}

void IntegerMatrix::calculateOffsets() {
    row_offsets_.resize(N_);
    uint64_t currentOffset = 0;
    for (uint64_t i = 0; i < N_; ++i) {
        row_offsets_[i] = currentOffset;
        uint64_t rowLen = (N_ - 1) - i;
        if (rowLen > 0) {
            currentOffset += rowLen * sizeof(uint32_t);
        }
    }
}

void IntegerMatrix::set(uint64_t i, uint64_t j, uint32_t value) {
    if (i >= j) throw std::invalid_argument("Strictly upper triangular");
    if (j >= N_) throw std::out_of_range("Index out of bounds");

    uint64_t colIndex = j - (i + 1);
    uint64_t byteOffset = row_offsets_[i] + colIndex * sizeof(uint32_t);
    
    uint32_t* dataPtr = (uint32_t*)((char*)mapper_->getData() + byteOffset);
    *dataPtr = value;
}

uint32_t IntegerMatrix::get(uint64_t i, uint64_t j) const {
    if (i >= j || j >= N_) return 0;
    
    uint64_t colIndex = j - (i + 1);
    uint64_t byteOffset = row_offsets_[i] + colIndex * sizeof(uint32_t);
    
    const uint32_t* dataPtr = (const uint32_t*)((const char*)mapper_->getData() + byteOffset);
    return *dataPtr;
}
