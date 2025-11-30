#include "MatrixOperations.hpp"
#include "TriangularMatrix.hpp"
#include "TriangularBitMatrix.hpp"
#include "DenseMatrix.hpp"
#include "IdentityMatrix.hpp"
#include <stdexcept>
#include <omp.h>
#include <bit>
#include <vector>

namespace pycauset {

bool is_triangular(const MatrixBase& m) {
    // Check if dynamic cast to TriangularMatrixBase works?
    // Or check matrix type from header.
    // Since we don't have easy access to MatrixType enum here without including FileFormat.hpp,
    // let's use dynamic_cast.
    return dynamic_cast<const TriangularMatrixBase*>(&m) != nullptr;
}

bool is_identity(const MatrixBase& m) {
    return dynamic_cast<const IdentityMatrix*>(&m) != nullptr;
}

std::unique_ptr<MatrixBase> add(const MatrixBase& a, const MatrixBase& b, const std::string& result_file) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Matrix dimensions must match");
    }
    uint64_t n = a.size();
    
    if (is_identity(a) && is_identity(b)) {
        const auto& a_id = static_cast<const IdentityMatrix&>(a);
        const auto& b_id = static_cast<const IdentityMatrix&>(b);
        return a_id.add(b_id, result_file);
    }

    bool both_triangular = is_triangular(a) && is_triangular(b);
    
    if (both_triangular) {
        auto result = std::make_unique<TriangularMatrix<double>>(n, result_file);
        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t j = i + 1; j < n; ++j) {
                double val = a.get_element_as_double(i, j) + b.get_element_as_double(i, j);
                if (val != 0.0) {
                    result->set(i, j, val);
                }
            }
        }
        return result;
    } else {
        auto result = std::make_unique<DenseMatrix<double>>(n, result_file);
        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t j = 0; j < n; ++j) {
                double val = a.get_element_as_double(i, j) + b.get_element_as_double(i, j);
                result->set(i, j, val);
            }
        }
        return result;
    }
}

std::unique_ptr<MatrixBase> subtract(const MatrixBase& a, const MatrixBase& b, const std::string& result_file) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Matrix dimensions must match");
    }
    uint64_t n = a.size();
    
    if (is_identity(a) && is_identity(b)) {
        const auto& a_id = static_cast<const IdentityMatrix&>(a);
        const auto& b_id = static_cast<const IdentityMatrix&>(b);
        return a_id.subtract(b_id, result_file);
    }

    bool both_triangular = is_triangular(a) && is_triangular(b);
    
    if (both_triangular) {
        auto result = std::make_unique<TriangularMatrix<double>>(n, result_file);
        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t j = i + 1; j < n; ++j) {
                double val = a.get_element_as_double(i, j) - b.get_element_as_double(i, j);
                if (val != 0.0) {
                    result->set(i, j, val);
                }
            }
        }
        return result;
    } else {
        auto result = std::make_unique<DenseMatrix<double>>(n, result_file);
        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t j = 0; j < n; ++j) {
                double val = a.get_element_as_double(i, j) - b.get_element_as_double(i, j);
                result->set(i, j, val);
            }
        }
        return result;
    }
}

std::unique_ptr<MatrixBase> elementwise_multiply(const MatrixBase& a, const MatrixBase& b, const std::string& result_file) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Matrix dimensions must match");
    }
    uint64_t n = a.size();
    
    if (is_identity(a) && is_identity(b)) {
        const auto& a_id = static_cast<const IdentityMatrix&>(a);
        const auto& b_id = static_cast<const IdentityMatrix&>(b);
        return a_id.elementwise_multiply(b_id, result_file);
    }

    bool both_triangular = is_triangular(a) && is_triangular(b);
    
    if (both_triangular) {
        auto result = std::make_unique<TriangularMatrix<double>>(n, result_file);
        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t j = i + 1; j < n; ++j) {
                double val = a.get_element_as_double(i, j) * b.get_element_as_double(i, j);
                if (val != 0.0) {
                    result->set(i, j, val);
                }
            }
        }
        return result;
    } else {
        auto result = std::make_unique<DenseMatrix<double>>(n, result_file);
        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t j = 0; j < n; ++j) {
                double val = a.get_element_as_double(i, j) * b.get_element_as_double(i, j);
                result->set(i, j, val);
            }
        }
        return result;
    }
}

} // namespace pycauset

void compute_k_matrix(
    const TriangularMatrix<bool>& C, 
    double a, 
    const std::string& output_path, 
    int num_threads
) {
    uint64_t n = C.size();
    TriangularMatrix<double> K(n, output_path);
    
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    // Get base pointer for C
    const char* c_raw_bytes = reinterpret_cast<const char*>(C.data());
    char* k_base_ptr = reinterpret_cast<char*>(K.data());

    #pragma omp parallel for schedule(dynamic, 64)
    for (int64_t j = 0; j < (int64_t)n; ++j) {
        // Allocate temporary column buffer for K[0...j-1][j]
        std::vector<double> col_j(j + 1, 0.0);
        
        // Iterate i from j-1 down to 0
        for (int64_t i = j - 1; i >= 0; --i) {
            double sum = 0.0;
            
            uint64_t row_offset = C.get_row_offset(i);
            const uint64_t* row_ptr = reinterpret_cast<const uint64_t*>(c_raw_bytes + row_offset);
            
            if (j > i + 1) {
                uint64_t max_bit_index = j - i - 2;
                uint64_t num_words = (max_bit_index / 64) + 1;
                
                for (uint64_t w = 0; w < num_words; ++w) {
                    uint64_t word = row_ptr[w];
                    if (word == 0) continue;
                    
                    if (w == num_words - 1) {
                        uint64_t bits_in_last_word = (max_bit_index % 64) + 1;
                        if (bits_in_last_word < 64) {
                            uint64_t mask = (1ULL << bits_in_last_word) - 1;
                            word &= mask;
                        }
                    }

                    while (word != 0) {
                        int bit = std::countr_zero(word);
                        uint64_t m = (i + 1) + (w * 64 + bit);
                        
                        if (m < j) {
                            sum += col_j[m];
                        }
                        word &= (word - 1);
                    }
                }
            }
            
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
            
            uint64_t k_row_offset = K.get_row_offset(i);
            uint64_t k_col_idx = j - (i + 1);
            double* k_row_ptr = reinterpret_cast<double*>(k_base_ptr + k_row_offset);
            k_row_ptr[k_col_idx] = val;
        }
    }
}
