#include "CausalMatrix.hpp"
#include "IntegerMatrix.hpp"
#include <stdexcept>
#include <bit>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

CausalMatrix::CausalMatrix(uint64_t n, const std::string& backing_file, bool populate,
                           std::optional<uint64_t> seed)
    : TriangularMatrix(n) {
    // Calculate offsets for bits (1 bit per element), aligned to 64 bits
    uint64_t size_in_bytes = calculate_triangular_offsets(1, 64);
    initialize_storage(size_in_bytes, backing_file, "causal_matrix", 8, 
                      pycauset::MatrixType::CAUSAL, pycauset::DataType::BIT);

    if (populate) {
        fill_random(0.5, seed);
    }
}

CausalMatrix::CausalMatrix(uint64_t n, std::unique_ptr<MemoryMapper> mapper)
    : TriangularMatrix(n, std::move(mapper)) {
    // Calculate offsets for bits (1 bit per element), aligned to 64 bits
    calculate_triangular_offsets(1, 64);
}

void CausalMatrix::set(uint64_t i, uint64_t j, bool value) {
    if (i >= j) throw std::invalid_argument("Strictly upper triangular");
    if (j >= n_) throw std::out_of_range("Index out of bounds");

    // Row i starts at row_offsets_[i]
    // It represents columns i+1 to N-1
    // The bit for column j is at index (j - (i + 1)) in this row
    
    uint64_t bit_offset = j - (i + 1);
    uint64_t word_index = bit_offset / 64;
    uint64_t bit_index = bit_offset % 64;

    uint64_t byte_offset = row_offsets_[i] + word_index * 8;
    auto* base_ptr = static_cast<char*>(require_mapper()->get_data());
    uint64_t* data_ptr = reinterpret_cast<uint64_t*>(base_ptr + byte_offset);
    
    if (value) {
        *data_ptr |= (1ULL << bit_index);
    } else {
        *data_ptr &= ~(1ULL << bit_index);
    }
}

bool CausalMatrix::get(uint64_t i, uint64_t j) const {
    if (i >= j || j >= n_) return false;

    uint64_t bit_offset = j - (i + 1);
    uint64_t word_index = bit_offset / 64;
    uint64_t bit_index = bit_offset % 64;

    uint64_t byte_offset = row_offsets_[i] + word_index * 8;
    auto* base_ptr = static_cast<const char*>(require_mapper()->get_data());
    const uint64_t* data_ptr = reinterpret_cast<const uint64_t*>(base_ptr + byte_offset);
    
    return (*data_ptr >> bit_index) & 1ULL;
}

std::unique_ptr<CausalMatrix> CausalMatrix::random(uint64_t n, double density,
                                                   const std::string& backing_file,
                                                   std::optional<uint64_t> seed) {
    auto mat = std::make_unique<CausalMatrix>(n, backing_file, false);
    mat->fill_random(density, seed);
    return mat;
}

std::unique_ptr<IntegerMatrix> CausalMatrix::multiply(const CausalMatrix& other, const std::string& result_file) const {
    if (n_ != other.size()) {
        throw std::invalid_argument("Matrix dimensions must match");
    }

    auto result = std::make_unique<IntegerMatrix>(n_, result_file);
    
    // Optimized Row-Addition Algorithm
    // C[i][j] = sum_k (A[i][k] * B[k][j])
    
    // Buffer to accumulate one row of results
    std::vector<uint32_t> accumulator(n_);

    const char* a_base = static_cast<const char*>(require_mapper()->get_data());
    const char* b_base = static_cast<const char*>(other.require_mapper()->get_data());

    for (uint64_t i = 0; i < n_; ++i) {
        // Clear accumulator for Row i
        std::fill(accumulator.begin(), accumulator.end(), 0);
        
        // Iterate over Row i of A
        // Row i has length N - 1 - i
        uint64_t row_len_a = (n_ - 1) - i;
        if (row_len_a == 0) continue;

        uint64_t words_a = (row_len_a + 63) / 64;
        const uint64_t* a_row_ptr = (const uint64_t*)(a_base + row_offsets_[i]);

        for (uint64_t w = 0; w < words_a; ++w) {
            uint64_t word = a_row_ptr[w];
            if (word == 0) continue;

            // Iterate set bits in word
            while (word) {
                int bit = std::countr_zero(word); // Number of trailing zeros
                // The column index k in A corresponding to this bit
                // bit 0 corresponds to col i+1 + w*64
                uint64_t k = (i + 1) + w * 64 + bit;
                
                if (k < n_) {
                    // Add Row k of B to accumulator
                    // Row k of B starts at col k+1
                    uint64_t row_len_b = (n_ - 1) - k;
                    if (row_len_b > 0) {
                        uint64_t words_b = (row_len_b + 63) / 64;
                        const uint64_t* b_row_ptr = (const uint64_t*)(b_base + other.get_row_offset(k));
                        
                        for (uint64_t wb = 0; wb < words_b; ++wb) {
                            uint64_t word_b = b_row_ptr[wb];
                            if (word_b == 0) continue;
                            
                            // Add bits of B[k] to accumulator
                            // bit b in wordB corresponds to col (k+1) + wb*64 + b
                            uint64_t base_col = (k + 1) + wb * 64;
                            
                            // Unroll loop for performance?
                            while (word_b) {
                                int bit_b = std::countr_zero(word_b);
                                uint64_t col = base_col + bit_b;
                                if (col < n_) {
                                    accumulator[col]++;
                                }
                                word_b &= ~(1ULL << bit_b);
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
        for (uint64_t j = i + 1; j < n_; ++j) {
            if (accumulator[j] > 0) {
                result->set(i, j, accumulator[j]);
            }
        }
    }

    return result;
}

std::unique_ptr<CausalMatrix> CausalMatrix::elementwise_multiply(const CausalMatrix& other,
                                                                const std::string& result_file) const {
    if (n_ != other.size()) {
        throw std::invalid_argument("Matrix dimensions must match");
    }

    auto result = std::make_unique<CausalMatrix>(n_, result_file, false);
    const char* lhs_base = static_cast<const char*>(require_mapper()->get_data());
    const char* rhs_base = static_cast<const char*>(other.require_mapper()->get_data());
    char* dst_base = static_cast<char*>(result->require_mapper()->get_data());

    for (uint64_t i = 0; i < n_; ++i) {
        uint64_t row_len = (n_ - 1) - i;
        if (row_len == 0) {
            continue;
        }
        uint64_t words = (row_len + 63) / 64;

        const auto* lhs_row = reinterpret_cast<const uint64_t*>(lhs_base + row_offsets_[i]);
        const auto* rhs_row = reinterpret_cast<const uint64_t*>(rhs_base + other.get_row_offset(i));
        auto* dst_row = reinterpret_cast<uint64_t*>(dst_base + result->get_row_offset(i));

        for (uint64_t w = 0; w < words; ++w) {
            dst_row[w] = lhs_row[w] & rhs_row[w];
        }

        uint64_t valid_bits = row_len % 64;
        if (valid_bits > 0) {
            uint64_t mask = (1ULL << valid_bits) - 1;
            dst_row[words - 1] &= mask;
        }
    }

    return result;
}

void CausalMatrix::fill_random(double density, std::optional<uint64_t> seed) {
    std::mt19937_64 gen;
    if (seed.has_value()) {
        gen.seed(*seed);
    } else {
        std::random_device rd;
        gen.seed(rd());
    }

    if (std::abs(density - 0.5) < 0.0001) {
        uint64_t* raw_data = data();
        for (uint64_t i = 0; i < n_; ++i) {
            uint64_t row_len = (n_ - 1) - i;
            if (row_len == 0) {
                continue;
            }
            uint64_t words = (row_len + 63) / 64;
            auto* row_ptr = reinterpret_cast<uint64_t*>(
                reinterpret_cast<char*>(raw_data) + get_row_offset(i));
            for (uint64_t w = 0; w < words; ++w) {
                row_ptr[w] = gen();
            }
            uint64_t valid_bits = row_len % 64;
            if (valid_bits > 0) {
                uint64_t mask = (1ULL << valid_bits) - 1;
                row_ptr[words - 1] &= mask;
            }
        }
    } else {
        std::bernoulli_distribution dist(density);
        for (uint64_t i = 0; i < n_; ++i) {
            for (uint64_t j = i + 1; j < n_; ++j) {
                if (dist(gen)) {
                    set(i, j, true);
                }
            }
        }
    }
}



