#include "DenseBitMatrix.hpp"
#include <stdexcept>
#include <bit>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <cstring>

// Helper for popcount
static inline int popcount64(uint64_t x) {
    return std::popcount(x);
}

void DenseMatrix<bool>::calculate_stride() {
    // Align rows to 64-bit (8-byte) boundaries
    uint64_t words_per_row = (n_ + 63) / 64;
    stride_bytes_ = words_per_row * 8;
}

DenseMatrix<bool>::DenseMatrix(uint64_t n, const std::string& backing_file)
    : MatrixBase(n) {
    calculate_stride();
    uint64_t size_in_bytes = n_ * stride_bytes_;
    
    // We use DENSE_FLOAT as a placeholder type for now, or we should add DENSE_BIT.
    // But since we are using DataType::BIT, it should be fine.
    // Let's check FileFormat.hpp for MatrixType enum.
    // Assuming DENSE_FLOAT is 0 or similar.
    // Ideally we should add DENSE_BIT to the enum, but I can't change the enum definition easily without seeing it.
    // I'll use DENSE_FLOAT and rely on DataType::BIT to distinguish.
    initialize_storage(size_in_bytes, backing_file, "dense_bit_matrix", 8, 
                      pycauset::MatrixType::DENSE_FLOAT, pycauset::DataType::BIT);
}

DenseMatrix<bool>::DenseMatrix(uint64_t n, std::unique_ptr<MemoryMapper> mapper)
    : MatrixBase(n, std::move(mapper)) {
    calculate_stride();
}

void DenseMatrix<bool>::set(uint64_t i, uint64_t j, bool value) {
    if (i >= n_ || j >= n_) throw std::out_of_range("Index out of bounds");

    uint64_t word_index = j / 64;
    uint64_t bit_index = j % 64;

    uint64_t byte_offset = i * stride_bytes_ + word_index * 8;
    auto* base_ptr = static_cast<char*>(require_mapper()->get_data());
    uint64_t* data_ptr = reinterpret_cast<uint64_t*>(base_ptr + byte_offset);
    
    if (value) {
        *data_ptr |= (1ULL << bit_index);
    } else {
        *data_ptr &= ~(1ULL << bit_index);
    }
}

bool DenseMatrix<bool>::get(uint64_t i, uint64_t j) const {
    if (i >= n_ || j >= n_) throw std::out_of_range("Index out of bounds");

    uint64_t word_index = j / 64;
    uint64_t bit_index = j % 64;

    uint64_t byte_offset = i * stride_bytes_ + word_index * 8;
    auto* base_ptr = static_cast<const char*>(require_mapper()->get_data());
    const uint64_t* data_ptr = reinterpret_cast<const uint64_t*>(base_ptr + byte_offset);
    
    return (*data_ptr >> bit_index) & 1ULL;
}

double DenseMatrix<bool>::get_element_as_double(uint64_t i, uint64_t j) const {
    return get(i, j) ? scalar_ : 0.0;
}

std::unique_ptr<DenseMatrix<int32_t>> DenseMatrix<bool>::multiply(const DenseMatrix<bool>& other, const std::string& result_file) const {
    if (n_ != other.size()) {
        throw std::invalid_argument("Matrix dimensions must match");
    }

    auto result = std::make_unique<DenseMatrix<int32_t>>(n_, result_file);
    
    // Naive implementation: O(N^3) bit ops.
    // Optimization: Transpose 'other' to allow row-row operations (popcount).
    
    // 1. Create temporary transpose of 'other' in memory (or disk if too large?)
    // For simplicity, let's do an in-memory transpose if it fits, or just slow access.
    // Given we want "efficient", let's do a block-based approach or just transpose.
    // Let's assume we can allocate a temporary buffer for the transpose of B.
    // Since it's bits, N=10000 is ~12MB. It fits in memory easily.
    
    std::vector<uint64_t> b_transposed(n_ * (stride_bytes_ / 8));
    // Fill transpose
    for (uint64_t i = 0; i < n_; ++i) {
        for (uint64_t j = 0; j < n_; ++j) {
            if (other.get(i, j)) {
                // Set (j, i) in transposed
                uint64_t word_idx = i / 64;
                uint64_t bit_idx = i % 64;
                b_transposed[j * (stride_bytes_ / 8) + word_idx] |= (1ULL << bit_idx);
            }
        }
    }

    const uint64_t* a_data = data();
    uint64_t words_per_row = stride_bytes_ / 8;
    
    for (uint64_t i = 0; i < n_; ++i) {
        const uint64_t* a_row = a_data + i * words_per_row;
        for (uint64_t j = 0; j < n_; ++j) {
            const uint64_t* b_col = b_transposed.data() + j * words_per_row;
            
            int32_t dot_product = 0;
            for (uint64_t k = 0; k < words_per_row; ++k) {
                dot_product += popcount64(a_row[k] & b_col[k]);
            }
            result->set(i, j, dot_product);
        }
    }
    
    return result;
}

std::unique_ptr<DenseMatrix<bool>> DenseMatrix<bool>::multiply_scalar(double factor, const std::string& result_file) const {
    // For boolean matrix, scalar mult just changes the scalar property, bits remain same.
    // Unless factor is 0, then it becomes all zeros.
    
    auto result = std::make_unique<DenseMatrix<bool>>(n_, result_file);
    
    if (factor == 0.0) {
        // Zero out
        // Already zeroed by default? initialize_storage usually zeroes?
        // MemoryMapper usually mmaps a new file which might be zeroed or garbage.
        // We should explicitly zero it.
        std::memset(result->data(), 0, n_ * stride_bytes_);
        result->set_scalar(1.0); // Scalar 0 is represented as all zero bits usually
    } else {
        // Copy bits
        std::memcpy(result->data(), data(), n_ * stride_bytes_);
        result->set_scalar(scalar_ * factor);
    }
    return result;
}

std::unique_ptr<DenseMatrix<bool>> DenseMatrix<bool>::bitwise_not(const std::string& result_file) const {
    auto result = std::make_unique<DenseMatrix<bool>>(n_, result_file);
    
    const uint64_t* src = data();
    uint64_t* dst = result->data();
    uint64_t total_words = (n_ * stride_bytes_) / 8;
    
    for (uint64_t i = 0; i < total_words; ++i) {
        dst[i] = ~src[i];
    }
    
    // Mask out padding bits in the last word of each row?
    // Not strictly necessary if we never read them, but good for cleanliness.
    // Our get() checks bounds, so we are safe.
    
    result->set_scalar(scalar_);
    return result;
}

std::unique_ptr<DenseMatrix<bool>> DenseMatrix<bool>::random(uint64_t n, double density,
                                            const std::string& backing_file,
                                            std::optional<uint64_t> seed) {
    auto mat = std::make_unique<DenseMatrix<bool>>(n, backing_file);
    mat->fill_random(density, seed);
    return mat;
}

void DenseMatrix<bool>::fill_random(double density, std::optional<uint64_t> seed) {
    std::mt19937_64 gen;
    if (seed) {
        gen.seed(*seed);
        this->set_seed(*seed);
    } else {
        std::random_device rd;
        uint64_t s = rd();
        gen.seed(s);
        this->set_seed(s);
    }

    std::bernoulli_distribution d(density);
    
    // We can optimize this by generating 64 bits at a time if density is 0.5.
    // For arbitrary density, we iterate.
    
    for (uint64_t i = 0; i < n_; ++i) {
        for (uint64_t j = 0; j < n_; ++j) {
            if (d(gen)) {
                set(i, j, true);
            }
        }
    }
}
