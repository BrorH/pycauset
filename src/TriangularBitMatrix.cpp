#include "TriangularBitMatrix.hpp"
#include <stdexcept>
#include <bit>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <filesystem>

// Explicit template instantiation if needed, but we are defining the specialization here.

TriangularMatrix<bool>::TriangularMatrix(uint64_t n, const std::string& backing_file)
    : TriangularMatrixBase(n, nullptr) {
    // Calculate offsets for bits (1 bit per element), aligned to 64 bits
    uint64_t size_in_bytes = calculate_triangular_offsets(1, 64);
    initialize_storage(size_in_bytes, backing_file, "causal_matrix", 8, 
                      pycauset::MatrixType::CAUSAL, pycauset::DataType::BIT);
}

TriangularMatrix<bool>::TriangularMatrix(uint64_t n, std::unique_ptr<MemoryMapper> mapper)
    : TriangularMatrixBase(n, std::move(mapper)) {
    // Calculate offsets for bits (1 bit per element), aligned to 64 bits
    calculate_triangular_offsets(1, 64);
}

void TriangularMatrix<bool>::set(uint64_t i, uint64_t j, bool value) {
    if (i >= j) throw std::invalid_argument("Strictly upper triangular");
    if (j >= n_) throw std::out_of_range("Index out of bounds");

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

bool TriangularMatrix<bool>::get(uint64_t i, uint64_t j) const {
    if (i >= j || j >= n_) return false;

    uint64_t bit_offset = j - (i + 1);
    uint64_t word_index = bit_offset / 64;
    uint64_t bit_index = bit_offset % 64;

    uint64_t byte_offset = row_offsets_[i] + word_index * 8;
    auto* base_ptr = static_cast<const char*>(require_mapper()->get_data());
    const uint64_t* data_ptr = reinterpret_cast<const uint64_t*>(base_ptr + byte_offset);
    
    return (*data_ptr >> bit_index) & 1ULL;
}

double TriangularMatrix<bool>::get_element_as_double(uint64_t i, uint64_t j) const {
    if (scalar_ == 1.0) {
        return get(i, j) ? 1.0 : 0.0;
    }
    return get(i, j) ? scalar_ : 0.0;
}

std::unique_ptr<TriangularMatrix<bool>> TriangularMatrix<bool>::random(uint64_t n, double density,
                                                   const std::string& backing_file,
                                                   std::optional<uint64_t> seed) {
    auto mat = std::make_unique<TriangularMatrix<bool>>(n, backing_file);
    if (seed.has_value()) {
        mat->require_mapper()->get_header()->seed = seed.value();
    }
    mat->fill_random(density, seed);
    return mat;
}

std::unique_ptr<TriangularMatrix<bool>> TriangularMatrix<bool>::bitwise_not(const std::string& result_file) const {
    auto result = std::make_unique<TriangularMatrix<bool>>(n_, result_file);
    
    const uint64_t* src_data = data();
    uint64_t* dst_data = result->data();

    for (uint64_t i = 0; i < n_; ++i) {
        uint64_t row_len = (n_ - 1) - i; // Number of bits in this row
        if (row_len == 0) continue;

        uint64_t words = (row_len + 63) / 64;
        // Row offsets are guaranteed to be 64-bit aligned by calculate_triangular_offsets(1, 64)
        uint64_t word_offset = get_row_offset(i) / 8;

        for (uint64_t w = 0; w < words; ++w) {
            dst_data[word_offset + w] = ~src_data[word_offset + w];
        }

        // Mask padding bits in the last word to ensure they remain 0
        uint64_t valid_bits = row_len % 64;
        if (valid_bits > 0) {
            uint64_t mask = (1ULL << valid_bits) - 1;
            dst_data[word_offset + words - 1] &= mask;
        }
    }
    
    result->set_scalar(scalar_);
    return result;
}

std::unique_ptr<TriangularMatrix<double>> TriangularMatrix<bool>::inverse(const std::string& result_file) const {
    throw std::runtime_error("TriangularMatrix<bool> is strictly upper triangular and therefore singular (not invertible).");
}

std::unique_ptr<TriangularMatrix<int32_t>> TriangularMatrix<bool>::multiply(const TriangularMatrix<bool>& other, const std::string& result_file) const {
    if (n_ != other.size()) {
        throw std::invalid_argument("Matrix dimensions must match");
    }
    
    auto result = std::make_unique<TriangularMatrix<int32_t>>(n_, result_file);
    
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
    
    result->set_scalar(scalar_ * other.get_scalar());

    return result;
}

std::unique_ptr<TriangularMatrix<bool>> TriangularMatrix<bool>::elementwise_multiply(const TriangularMatrix<bool>& other,
                                                                const std::string& result_file) const {
    if (n_ != other.size()) {
        throw std::invalid_argument("Matrix dimensions must match");
    }

    auto result = std::make_unique<TriangularMatrix<bool>>(n_, result_file);
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
    
    // Result scalar = s1 * s2
    result->set_scalar(scalar_ * other.get_scalar());

    return result;
}

void TriangularMatrix<bool>::fill_random(double density, std::optional<uint64_t> seed) {
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

std::unique_ptr<TriangularMatrix<bool>> TriangularMatrix<bool>::multiply_scalar(double factor, const std::string& result_file) const {
    std::string new_path = copy_storage(result_file);
    
    uint64_t file_size = std::filesystem::file_size(new_path);
    uint64_t data_size = file_size - sizeof(pycauset::FileHeader);
    auto mapper = std::make_unique<MemoryMapper>(new_path, data_size, false);
    
    auto result = std::make_unique<TriangularMatrix<bool>>(n_, std::move(mapper));
    result->set_scalar(scalar_ * factor);
    
    return result;
}



