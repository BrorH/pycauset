#include "MatrixOperations.hpp"
#include "TriangularMatrix.hpp"
#include "TriangularBitMatrix.hpp"
#include "DenseMatrix.hpp"
#include "DenseBitMatrix.hpp"
#include "IdentityMatrix.hpp"
#include "DiagonalMatrix.hpp"
#include "MatrixFactory.hpp"
#include "UnitVector.hpp"
#include "ParallelUtils.hpp"
#include "ComputeContext.hpp"
#include "ComputeDevice.hpp"
#include <stdexcept>
#include <bit>
#include <vector>
#include <functional>

namespace pycauset {

using IntegerMatrix = DenseMatrix<int32_t>;
using FloatMatrix = DenseMatrix<double>;

bool is_triangular(const MatrixBase& m) {
    // Check if dynamic cast to TriangularMatrixBase works?
    // Or check matrix type from header.
    // Since we don't have easy access to MatrixType enum here without including FileFormat.hpp,
    // let's use dynamic_cast.
    return dynamic_cast<const TriangularMatrixBase*>(&m) != nullptr;
}

bool is_identity(const MatrixBase& m) {
    return m.get_matrix_type() == MatrixType::IDENTITY;
}

std::unique_ptr<MatrixBase> add(const MatrixBase& a, const MatrixBase& b, const std::string& result_file) {
    if (is_identity(a) && is_identity(b)) {
        if (a.get_data_type() == DataType::FLOAT64 && b.get_data_type() == DataType::FLOAT64) {
            const auto& a_id = static_cast<const IdentityMatrix<double>&>(a);
            const auto& b_id = static_cast<const IdentityMatrix<double>&>(b);
            return a_id.add(b_id, result_file);
        }
        if (a.get_data_type() == DataType::INT32 && b.get_data_type() == DataType::INT32) {
            const auto& a_id = static_cast<const IdentityMatrix<int32_t>&>(a);
            const auto& b_id = static_cast<const IdentityMatrix<int32_t>&>(b);
            return a_id.add(b_id, result_file);
        }
    }

    DataType type_a = a.get_data_type();
    DataType type_b = b.get_data_type();
    MatrixType mtype_a = a.get_matrix_type();
    MatrixType mtype_b = b.get_matrix_type();

    DataType res_dtype = MatrixFactory::resolve_result_type(type_a, type_b);
    MatrixType res_mtype = MatrixFactory::resolve_result_matrix_type(mtype_a, mtype_b);

    auto result = MatrixFactory::create(a.size(), res_dtype, res_mtype, result_file);
    
    ComputeContext::instance().get_device()->add(a, b, *result);
    
    return result;
}

std::unique_ptr<MatrixBase> subtract(const MatrixBase& a, const MatrixBase& b, const std::string& result_file) {
    if (is_identity(a) && is_identity(b)) {
        if (a.get_data_type() == DataType::FLOAT64 && b.get_data_type() == DataType::FLOAT64) {
            const auto& a_id = static_cast<const IdentityMatrix<double>&>(a);
            const auto& b_id = static_cast<const IdentityMatrix<double>&>(b);
            return a_id.subtract(b_id, result_file);
        }
        if (a.get_data_type() == DataType::INT32 && b.get_data_type() == DataType::INT32) {
            const auto& a_id = static_cast<const IdentityMatrix<int32_t>&>(a);
            const auto& b_id = static_cast<const IdentityMatrix<int32_t>&>(b);
            return a_id.subtract(b_id, result_file);
        }
    }

    DataType type_a = a.get_data_type();
    DataType type_b = b.get_data_type();
    MatrixType mtype_a = a.get_matrix_type();
    MatrixType mtype_b = b.get_matrix_type();

    DataType res_dtype = MatrixFactory::resolve_result_type(type_a, type_b);
    MatrixType res_mtype = MatrixFactory::resolve_result_matrix_type(mtype_a, mtype_b);

    auto result = MatrixFactory::create(a.size(), res_dtype, res_mtype, result_file);
    
    ComputeContext::instance().get_device()->subtract(a, b, *result);
    
    return result;
}

std::unique_ptr<MatrixBase> elementwise_multiply(const MatrixBase& a, const MatrixBase& b, const std::string& result_file) {
    if (is_identity(a) && is_identity(b)) {
        if (a.get_data_type() == DataType::FLOAT64 && b.get_data_type() == DataType::FLOAT64) {
            const auto& a_id = static_cast<const IdentityMatrix<double>&>(a);
            const auto& b_id = static_cast<const IdentityMatrix<double>&>(b);
            return a_id.elementwise_multiply(b_id, result_file);
        }
        if (a.get_data_type() == DataType::INT32 && b.get_data_type() == DataType::INT32) {
            const auto& a_id = static_cast<const IdentityMatrix<int32_t>&>(a);
            const auto& b_id = static_cast<const IdentityMatrix<int32_t>&>(b);
            return a_id.elementwise_multiply(b_id, result_file);
        }
    }

    DataType type_a = a.get_data_type();
    DataType type_b = b.get_data_type();
    MatrixType mtype_a = a.get_matrix_type();
    MatrixType mtype_b = b.get_matrix_type();

    DataType res_dtype = MatrixFactory::resolve_result_type(type_a, type_b);
    MatrixType res_mtype = MatrixFactory::resolve_result_matrix_type(mtype_a, mtype_b);

    auto result = MatrixFactory::create(a.size(), res_dtype, res_mtype, result_file);
    
    ComputeContext::instance().get_device()->elementwise_multiply(a, b, *result);
    
    return result;
}

std::unique_ptr<MatrixBase> outer_product(const VectorBase& a, const VectorBase& b, const std::string& result_file) {
    if (a.size() != b.size()) throw std::invalid_argument("Vectors must be same size");
    uint64_t n = a.size();

    DataType type_a = a.get_data_type();
    DataType type_b = b.get_data_type();
    DataType res_dtype = MatrixFactory::resolve_result_type(type_a, type_b);
    
    // Resolve matrix type for outer product (usually Dense)
    // Note: DenseMatrix<T> currently uses DENSE_FLOAT for all types in this codebase
    MatrixType res_mtype = MatrixType::DENSE_FLOAT;

    auto result = MatrixFactory::create(n, res_dtype, res_mtype, result_file);
    
    ComputeContext::instance().get_device()->outer_product(a, b, *result);
    
    return result;
}

std::unique_ptr<VectorBase> matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, const std::string& result_file) {
    if (m.size() != v.size()) throw std::invalid_argument("Dimension mismatch");
    uint64_t n = m.size();

    DataType type_m = m.get_data_type();
    DataType type_v = v.get_data_type();
    DataType res_dtype = MatrixFactory::resolve_result_type(type_m, type_v);
    if (res_dtype == DataType::BIT) res_dtype = DataType::INT32;
    
    // Create result vector
    // VectorFactory? Or just make_unique?
    // We don't have VectorFactory exposed here easily, let's use direct creation or a helper.
    // But wait, VectorFactory is included.
    // Let's assume we can just create DenseVector.
    
    std::unique_ptr<VectorBase> result;
    if (res_dtype == DataType::INT32) {
        result = std::make_unique<DenseVector<int32_t>>(n, result_file);
    } else {
        result = std::make_unique<DenseVector<double>>(n, result_file);
    }
    
    ComputeContext::instance().get_device()->matrix_vector_multiply(m, v, *result);
    
    return result;
}

std::unique_ptr<VectorBase> vector_matrix_multiply(const VectorBase& v, const MatrixBase& m, const std::string& result_file) {
    if (m.size() != v.size()) throw std::invalid_argument("Dimension mismatch");
    uint64_t n = m.size();

    DataType type_m = m.get_data_type();
    DataType type_v = v.get_data_type();
    DataType res_dtype = MatrixFactory::resolve_result_type(type_m, type_v);
    if (res_dtype == DataType::BIT) res_dtype = DataType::INT32;

    std::unique_ptr<VectorBase> result;
    if (res_dtype == DataType::INT32) {
        result = std::make_unique<DenseVector<int32_t>>(n, result_file);
    } else {
        result = std::make_unique<DenseVector<double>>(n, result_file);
    }
    
    ComputeContext::instance().get_device()->vector_matrix_multiply(v, m, *result);
    result->set_transposed(true); // Result is row vector
    
    return result;
}

} // namespace pycauset

std::unique_ptr<TriangularMatrix<double>> compute_k_matrix(
    const TriangularMatrix<bool>& C, 
    double a, 
    const std::string& output_path, 
    int num_threads
) {
    uint64_t n = C.size();
    auto K = std::make_unique<TriangularMatrix<double>>(n, output_path);
    
    // Get base pointer for C
    const char* c_raw_bytes = reinterpret_cast<const char*>(C.data());
    char* k_base_ptr = reinterpret_cast<char*>(K->data());

    pycauset::ParallelFor(0, n, [&](size_t j) {
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
            
            uint64_t k_row_offset = K->get_row_offset(i);
            uint64_t k_col_idx = j - (i + 1);
            double* k_row_ptr = reinterpret_cast<double*>(k_base_ptr + k_row_offset);
            k_row_ptr[k_col_idx] = val;
        }
    });
    return K;
}