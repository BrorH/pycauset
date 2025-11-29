#include "KMatrix.hpp"
#include "TriangularFloatMatrix.hpp"
#include <vector>
#include <cmath>
#include <bit>
#include <omp.h>
#include <iostream>

FloatMatrix::FloatMatrix(uint64_t n, const std::string& backing_file)
    : MatrixBase(n) {
    // Size is N * N * sizeof(double)
    uint64_t size_in_bytes = n * n * sizeof(double);
    initialize_storage(size_in_bytes, backing_file, "k_matrix", 8,
                      pycauset::MatrixType::DENSE_FLOAT, pycauset::DataType::FLOAT64);
}

FloatMatrix::FloatMatrix(uint64_t n, std::unique_ptr<MemoryMapper> mapper)
    : MatrixBase(n, std::move(mapper)) {}

void FloatMatrix::set(uint64_t i, uint64_t j, double value) {
    if (i >= n_ || j >= n_) throw std::out_of_range("Index out of bounds");
    double* ptr = data();
    ptr[i * n_ + j] = value;
}

double FloatMatrix::get(uint64_t i, uint64_t j) const {
    if (i >= n_ || j >= n_) throw std::out_of_range("Index out of bounds");
    const double* ptr = data();
    return ptr[i * n_ + j];
}

void compute_k_matrix(
    const CausalMatrix& C, 
    double a, 
    const std::string& output_path, 
    int num_threads
) {
    uint64_t n = C.size();
    // Use TriangularFloatMatrix instead of dense FloatMatrix
    TriangularFloatMatrix K(n, output_path);
    
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    // Get base pointer for C
    const char* c_raw_bytes = reinterpret_cast<const char*>(C.data());
    // K data is not contiguous in the same way, so we can't get a single double* for the whole matrix
    // But we can get the base pointer for K's memory map
    // However, K.set() handles the offset calculation.
    // To optimize, we should calculate pointers manually inside the loop.
    
    char* k_base_ptr = reinterpret_cast<char*>(K.data());

    #pragma omp parallel for schedule(dynamic, 64)
    for (int64_t j = 0; j < (int64_t)n; ++j) {
        // Allocate temporary column buffer for K[0...j-1][j]
        // We use a vector to store the column values as we compute them.
        // col_j[i] will store K[i][j]
        std::vector<double> col_j(j + 1, 0.0);
        
        // Iterate i from j-1 down to 0
        for (int64_t i = j - 1; i >= 0; --i) {
            double sum = 0.0;
            
            // We need to iterate m from i+1 to j-1 where C[i][m] is 1.
            // C[i][m] corresponds to bit (m - (i + 1)) in row i.
            
            uint64_t row_offset = C.get_row_offset(i);
            const uint64_t* row_ptr = reinterpret_cast<const uint64_t*>(c_raw_bytes + row_offset);
            
            // We need to scan bits from 0 to (j - 1) - (i + 1) = j - i - 2.
            // If j <= i + 1, the range is empty.
            
            if (j > i + 1) {
                uint64_t max_bit_index = j - i - 2;
                uint64_t num_words = (max_bit_index / 64) + 1;
                
                for (uint64_t w = 0; w < num_words; ++w) {
                    uint64_t word = row_ptr[w];
                    if (word == 0) continue;
                    
                    // If this is the last word, mask out bits beyond max_bit_index
                    if (w == num_words - 1) {
                        uint64_t bits_in_last_word = (max_bit_index % 64) + 1;
                        if (bits_in_last_word < 64) {
                            uint64_t mask = (1ULL << bits_in_last_word) - 1;
                            word &= mask;
                        }
                    }

                    // Iterate set bits
                    while (word != 0) {
                        int bit = std::countr_zero(word);
                        uint64_t m = (i + 1) + (w * 64 + bit);
                        
                        // Double check m < j (should be guaranteed by mask)
                        if (m < j) {
                            sum += col_j[m];
                        }
                        
                        // Clear the bit
                        word &= (word - 1);
                    }
                }
            }
            
            // C[i][j] is bit (j - (i + 1)) in row i.
            bool c_ij = false;
            uint64_t bit_offset_j = j - (i + 1);
            uint64_t word_idx_j = bit_offset_j / 64;
            uint64_t bit_idx_j = bit_offset_j % 64;
            
            uint64_t word_j = row_ptr[word_idx_j];
            if ((word_j >> bit_idx_j) & 1ULL) {
                c_ij = true;
            }
            
            double val = ( (c_ij ? 1.0 : 0.0) - sum ) / a;
            col_j[i] = val;
            
            // Write to output K[i][j]
            // K is TriangularFloatMatrix.
            // Row i offset: K.get_row_offset(i)
            // Col index in row: j - (i + 1)
            uint64_t k_row_offset = K.get_row_offset(i);
            uint64_t k_col_idx = j - (i + 1);
            double* k_row_ptr = reinterpret_cast<double*>(k_base_ptr + k_row_offset);
            k_row_ptr[k_col_idx] = val;
        }
    }
}
