#include "pycauset/matrix/TriangularBitMatrix.hpp"
#include "pycauset/math/ViewValue.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/core/ParallelUtils.hpp"
#include <stdexcept>
#include <bit>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <filesystem>
#include <future>
#include <memory>

namespace pycauset {

// Explicit template instantiation if needed, but we are defining the specialization here.

TriangularMatrix<bool>::TriangularMatrix(uint64_t n, const std::string& backing_file, bool has_diagonal)
    : TriangularMatrixBase(n, pycauset::Structure::TRIANGULAR, pycauset::DataType::BIT) {
    has_diagonal_ = has_diagonal;
    // Calculate offsets for bits (1 bit per element), aligned to 64 bits
    uint64_t size_in_bytes = calculate_triangular_offsets(1, 64);
    initialize_storage(size_in_bytes, backing_file, "causal_matrix", 8, 
                      pycauset::Structure::TRIANGULAR, pycauset::DataType::BIT,
                      n, n);
}

TriangularMatrix<bool>::TriangularMatrix(uint64_t n, 
                                         const std::string& backing_file,
                                         size_t offset,
                                         uint64_t seed,
                                         std::complex<double> scalar,
                                         bool is_transposed,
                                         bool has_diagonal)
    : TriangularMatrixBase(n, pycauset::Structure::TRIANGULAR, pycauset::DataType::BIT) {
    has_diagonal_ = has_diagonal;
    
    uint64_t size_in_bytes = calculate_triangular_offsets(1, 64);
    initialize_storage(size_in_bytes, backing_file, "", 8, 
                      pycauset::Structure::TRIANGULAR, pycauset::DataType::BIT,
                      n, n, offset, false);
    
    set_seed(seed);
    set_scalar(scalar);
    set_transposed(is_transposed);
}

TriangularMatrix<bool>::TriangularMatrix(uint64_t n, std::shared_ptr<MemoryMapper> mapper, bool has_diagonal)
    : TriangularMatrixBase(n, std::move(mapper), pycauset::Structure::TRIANGULAR, pycauset::DataType::BIT) {
    has_diagonal_ = has_diagonal;
    // Calculate offsets for bits (1 bit per element), aligned to 64 bits
    calculate_triangular_offsets(1, 64);
}

void TriangularMatrix<bool>::set(uint64_t i, uint64_t j, bool value) {
    if (i >= rows() || j >= cols()) throw std::out_of_range("Index out of bounds");
    ensure_unique();
    if (is_transposed()) {
        std::swap(i, j);
    }
    if (j < first_col(i)) {
        if (!value) return;  // writing a zero outside the triangle is a no-op
        throw std::invalid_argument(has_diagonal_ ? "Upper triangular" : "Strictly upper triangular");
    }
    if (j >= cols()) throw std::out_of_range("Index out of bounds");

    uint64_t bit_offset = j - first_col(i);
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
    if (i >= rows() || j >= cols()) throw std::out_of_range("Index out of bounds");  // review 3.2 C-8
    if (is_transposed()) {
        std::swap(i, j);
    }
    if (j < first_col(i) || j >= cols()) return false;

    uint64_t bit_offset = j - first_col(i);
    uint64_t word_index = bit_offset / 64;
    uint64_t bit_index = bit_offset % 64;

    uint64_t byte_offset = row_offsets_[i] + word_index * 8;
    auto* base_ptr = static_cast<const char*>(require_mapper()->get_data());
    const uint64_t* data_ptr = reinterpret_cast<const uint64_t*>(base_ptr + byte_offset);
    
    return (*data_ptr >> bit_index) & 1ULL;
}

double TriangularMatrix<bool>::element_as_double(uint64_t i, uint64_t j) const {
    return pycauset::view_value(get(i, j), scalar_, false).real();
}

std::unique_ptr<TriangularMatrix<bool>> TriangularMatrix<bool>::random(uint64_t n, double density,
                                                   const std::string& backing_file,
                                                   std::optional<uint64_t> seed) {
    auto mat = std::make_unique<TriangularMatrix<bool>>(n, backing_file);
    // Seed storage in header is removed
    mat->fill_random(density, seed);
    return mat;
}

std::unique_ptr<MatrixBase> TriangularMatrix<bool>::transpose(const std::string& result_file) const {
    (void)result_file;
    return transposed_clone();
}

std::unique_ptr<TriangularMatrix<int32_t>> TriangularMatrix<bool>::multiply(const TriangularMatrix<bool>& other, const std::string& result_file) const {
    if (has_diagonal_ || other.has_diagonal()) {
        throw std::logic_error("TriangularMatrix<bool>::multiply: kernel contract is strictly upper operands");
    }
    const uint64_t n = rows();
    if (rows() != other.rows() || cols() != other.cols()) {
        throw std::invalid_argument("Matrix dimensions must match");
    }
    
    auto result = std::make_unique<TriangularMatrix<int32_t>>(n, result_file);
    
    const char* a_base = static_cast<const char*>(require_mapper()->get_data());
    const char* b_base = static_cast<const char*>(other.require_mapper()->get_data());

    std::vector<std::future<void>> futures;
    auto& pool = pycauset::ThreadPool::instance();
    TriangularMatrix<int32_t>* result_ptr = result.get();

    size_t num_threads = pycauset::ThreadPool::get_num_threads();
    uint64_t chunk_size = (n + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
        uint64_t start = t * chunk_size;
        uint64_t end = std::min(start + chunk_size, n);
        if (start >= end) break;

        futures.emplace_back(pool.enqueue([this, start, end, &other, result_ptr, a_base, b_base, n]() {
            // Buffer to accumulate one row of results
            std::vector<uint32_t> accumulator(n);

            for (uint64_t i = start; i < end; ++i) {
                std::fill(accumulator.begin(), accumulator.end(), 0);
                
                // Iterate over Row i of A
                // Row i has length N - 1 - i
                uint64_t row_len_a = (n - 1) - i;
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
                        
                        if (k < n) {
                            // Add Row k of B to accumulator
                            // Row k of B starts at col k+1
                            uint64_t row_len_b = (n - 1) - k;
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
                                        if (col < n) {
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
                for (uint64_t j = i + 1; j < n; ++j) {
                    if (accumulator[j] > 0) {
                        result_ptr->set(i, j, accumulator[j]);
                    }
                }
            }
        }));
    }
    
    for(auto& f : futures) {
        f.get();
    }
    
    result->set_scalar(compose_factors(scalar_, other.get_scalar()));

    return result;
}

void TriangularMatrix<bool>::fill_random(double density, std::optional<uint64_t> seed) {
    if (!(density >= 0.0 && density <= 1.0)) throw std::invalid_argument("p must be in [0, 1]");  // NaN too
    ensure_unique();

    // The seed used is recorded (as DenseMatrix.random does), so a run can be repeated.
    const uint64_t used = seed.has_value() ? *seed : static_cast<uint64_t>(std::random_device{}());
    std::mt19937_64 gen(used);
    set_seed(used);

    // Writes rows through the storage's own mapping. (It mapped 64 MB windows of the file
    // one at a time and never unmapped them on POSIX, review 3.1 C-8; shared file pages are
    // written back by the kernel, so the windows saved nothing.)
    char* const base = static_cast<char*>(require_mapper()->get_data());
    const uint64_t n = rows();
    const bool half = density == 0.5;  // fair coins 64 at a time; p = 0.50005 is not 0.5 (C-11)
    std::bernoulli_distribution dist(density);
    for (uint64_t i = 0; i < n; ++i) {
        const uint64_t row_len = (n - 1) - i;
        if (row_len == 0) continue;
        const uint64_t words = (row_len + 63) / 64;
        uint64_t* row = reinterpret_cast<uint64_t*>(base + get_row_offset(i));
        if (half) {
            // Density one half: every bit an independent fair coin, 64 at a time.
            for (uint64_t w = 0; w < words; ++w) row[w] = gen();
            if (const uint64_t valid = row_len % 64) row[words - 1] &= (1ULL << valid) - 1;
        } else {
            std::fill(row, row + words, uint64_t{0});
            for (uint64_t j = i + 1; j < n; ++j) {
                if (dist(gen)) {
                    const uint64_t bit = j - (i + 1);
                    row[bit / 64] |= (1ULL << (bit % 64));
                }
            }
        }
    }
}

std::unique_ptr<MatrixBase> TriangularMatrix<bool>::multiply_scalar(double factor, const std::string& result_file) const {
    (void)result_file;
    auto out = std::make_unique<TriangularMatrix<bool>>(base_rows(), mapper_, has_diagonal_);
    out->set_scalar(compose_factors(scalar_, factor));
    out->set_seed(seed_);
    out->set_transposed(is_transposed());
    out->set_conjugated(is_conjugated());
    return out;
}







} // namespace pycauset
