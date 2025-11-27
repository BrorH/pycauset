#include "CausalMatrix.hpp"
#include "IntegerMatrix.hpp"
#include <stdexcept>
#include <bit>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

CausalMatrix::CausalMatrix(uint64_t N, const std::string& backingFile) : N_(N) {
    calculateOffsets();
    
    // Total size is the offset of the (non-existent) N-th row
    // row_offsets_ has N entries. The end of the last row is needed.
    // Let's calculate total bytes.
    // Last row (index N-1) has length 0.
    // Row N-2 has length 1.
    
    uint64_t totalBytes = 0;
    if (N > 0) {
        // The offset of the byte after the last row
        uint64_t lastRowLen = 1; // Row N-2 has 1 bit
        // Actually, let's just use the loop in calculateOffsets logic
        // Re-run logic here or trust calculateOffsets?
        // calculateOffsets populates row_offsets_.
        // We need the total size to initialize mapper.
        
        // Let's do a quick sum or just use the last offset + last row size
        // But calculateOffsets is called before mapper_ init? No, mapper_ needs size.
        // So we must calculate size first.
    }
    
    // Re-implement logic to get size first
    uint64_t currentOffset = 0;
    for (uint64_t i = 0; i < N; ++i) {
        uint64_t rowLen = (N - 1) - i;
        if (rowLen > 0) {
            uint64_t words = (rowLen + 63) / 64;
            currentOffset += words * 8;
        }
    }
    uint64_t sizeInBytes = currentOffset;
    if (sizeInBytes == 0) sizeInBytes = 8; // Minimum size to avoid empty mapping errors

    std::string path = backingFile;
    if (path.empty()) {
        path = "temp_matrix.bin"; 
    }

    mapper_ = std::make_unique<MemoryMapper>(path, sizeInBytes);
}

void CausalMatrix::calculateOffsets() {
    row_offsets_.resize(N_);
    uint64_t currentOffset = 0;
    for (uint64_t i = 0; i < N_; ++i) {
        row_offsets_[i] = currentOffset;
        uint64_t rowLen = (N_ - 1) - i;
        if (rowLen > 0) {
            uint64_t words = (rowLen + 63) / 64;
            currentOffset += words * 8;
        }
    }
}

void CausalMatrix::set(uint64_t i, uint64_t j, bool value) {
    if (i >= j) throw std::invalid_argument("Strictly upper triangular");
    if (j >= N_) throw std::out_of_range("Index out of bounds");

    // Row i starts at row_offsets_[i]
    // It represents columns i+1 to N-1
    // The bit for column j is at index (j - (i + 1)) in this row
    
    uint64_t bitOffset = j - (i + 1);
    uint64_t wordIndex = bitOffset / 64;
    uint64_t bitIndex = bitOffset % 64;

    uint64_t byteOffset = row_offsets_[i] + wordIndex * 8;
    uint64_t* dataPtr = (uint64_t*)((char*)mapper_->getData() + byteOffset);
    
    if (value) {
        *dataPtr |= (1ULL << bitIndex);
    } else {
        *dataPtr &= ~(1ULL << bitIndex);
    }
}

bool CausalMatrix::get(uint64_t i, uint64_t j) const {
    if (i >= j || j >= N_) return false;

    uint64_t bitOffset = j - (i + 1);
    uint64_t wordIndex = bitOffset / 64;
    uint64_t bitIndex = bitOffset % 64;

    uint64_t byteOffset = row_offsets_[i] + wordIndex * 8;
    const uint64_t* dataPtr = (const uint64_t*)((const char*)mapper_->getData() + byteOffset);
    
    return (*dataPtr >> bitIndex) & 1ULL;
}

std::unique_ptr<CausalMatrix> CausalMatrix::random(uint64_t N, double density, const std::string& backingFile) {
    auto mat = std::make_unique<CausalMatrix>(N, backingFile);
    
    std::random_device rd;
    std::mt19937_64 gen(rd());

    // Optimization for 50% density: fill with random words
    if (std::abs(density - 0.5) < 0.0001) {
        uint64_t* rawData = mat->data();
        for (uint64_t i = 0; i < N; ++i) {
            uint64_t rowLen = (N - 1) - i;
            if (rowLen == 0) continue;
            uint64_t words = (rowLen + 63) / 64;
            uint64_t* rowPtr = (uint64_t*)((char*)rawData + mat->getRowOffset(i));
            for (uint64_t w = 0; w < words; ++w) {
                rowPtr[w] = gen();
            }
            // Mask the last word to ensure unused bits are 0? 
            // Not strictly necessary if we always access via get(), but good practice.
            // The last word has (rowLen % 64) valid bits. If 0, all 64 are valid.
            uint64_t validBits = rowLen % 64;
            if (validBits > 0) {
                uint64_t mask = (1ULL << validBits) - 1;
                rowPtr[words - 1] &= mask;
            }
        }
    } else {
        std::bernoulli_distribution d(density);
        // Iterate only upper triangle
        for (uint64_t i = 0; i < N; ++i) {
            for (uint64_t j = i + 1; j < N; ++j) {
                if (d(gen)) {
                    mat->set(i, j, true);
                }
            }
        }
    }
    return mat;
}

std::unique_ptr<IntegerMatrix> CausalMatrix::multiply(const CausalMatrix& other, const std::string& resultFile) const {
    if (N_ != other.size()) {
        throw std::invalid_argument("Matrix dimensions must match");
    }

    auto result = std::make_unique<IntegerMatrix>(N_, resultFile);
    
    // Optimized Row-Addition Algorithm
    // C[i][j] = sum_k (A[i][k] * B[k][j])
    
    // Buffer to accumulate one row of results
    std::vector<uint32_t> accumulator(N_);

    const char* a_base = (const char*)mapper_->getData();
    const char* b_base = (const char*)other.mapper_->getData();

    for (uint64_t i = 0; i < N_; ++i) {
        // Clear accumulator for Row i
        std::fill(accumulator.begin(), accumulator.end(), 0);
        
        // Iterate over Row i of A
        // Row i has length N - 1 - i
        uint64_t rowLenA = (N_ - 1) - i;
        if (rowLenA == 0) continue;

        uint64_t wordsA = (rowLenA + 63) / 64;
        const uint64_t* a_row_ptr = (const uint64_t*)(a_base + row_offsets_[i]);

        for (uint64_t w = 0; w < wordsA; ++w) {
            uint64_t word = a_row_ptr[w];
            if (word == 0) continue;

            // Iterate set bits in word
            while (word) {
                int bit = std::countr_zero(word); // Number of trailing zeros
                // The column index k in A corresponding to this bit
                // bit 0 corresponds to col i+1 + w*64
                uint64_t k = (i + 1) + w * 64 + bit;
                
                if (k < N_) {
                    // Add Row k of B to accumulator
                    // Row k of B starts at col k+1
                    uint64_t rowLenB = (N_ - 1) - k;
                    if (rowLenB > 0) {
                        uint64_t wordsB = (rowLenB + 63) / 64;
                        const uint64_t* b_row_ptr = (const uint64_t*)(b_base + other.getRowOffset(k));
                        
                        for (uint64_t wb = 0; wb < wordsB; ++wb) {
                            uint64_t wordB = b_row_ptr[wb];
                            if (wordB == 0) continue;
                            
                            // Add bits of B[k] to accumulator
                            // bit b in wordB corresponds to col (k+1) + wb*64 + b
                            uint64_t baseCol = (k + 1) + wb * 64;
                            
                            // Unroll loop for performance?
                            while (wordB) {
                                int bitB = std::countr_zero(wordB);
                                uint64_t col = baseCol + bitB;
                                if (col < N_) {
                                    accumulator[col]++;
                                }
                                wordB &= ~(1ULL << bitB);
                            }
                        }
                    }
                }
                
                // Clear the processed bit
                word &= ~(1ULL << bit);
            }
        }

        // Write accumulator to result matrix
        // Result is strictly upper triangular, so we only care about cols > i
        for (uint64_t j = i + 1; j < N_; ++j) {
            if (accumulator[j] > 0) {
                result->set(i, j, accumulator[j]);
            }
        }
    }

    return result;
}
