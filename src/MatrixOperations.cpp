#include "MatrixOperations.hpp"
#include "TriangularMatrix.hpp"
#include "TriangularBitMatrix.hpp"
#include "DenseMatrix.hpp"
#include "DenseBitMatrix.hpp"
#include "IdentityMatrix.hpp"
#include "MatrixFactory.hpp"
#include <stdexcept>
#include <omp.h>
#include <bit>
#include <vector>
#include <functional>

namespace pycauset {

using IntegerMatrix = DenseMatrix<int32_t>;
using FloatMatrix = DenseMatrix<double>;

// Helper template for executing binary operations
template <typename T, typename Op>
void execute_binary_op(const MatrixBase& a, const MatrixBase& b, MatrixBase& result, Op op) {
    uint64_t n = a.size();
    
    // Try to cast to TriangularMatrix<T>
    auto* tri_res = dynamic_cast<TriangularMatrix<T>*>(&result);
    if (tri_res) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)n; ++i) {
            for (uint64_t j = i + 1; j < n; ++j) {
                T val = op(static_cast<T>(a.get_element_as_double(i, j)), 
                           static_cast<T>(b.get_element_as_double(i, j)));
                if (val != static_cast<T>(0)) {
                    tri_res->set(i, j, val);
                }
            }
        }
        return;
    }

    // Try to cast to DenseMatrix<T>
    auto* dense_res = dynamic_cast<DenseMatrix<T>*>(&result);
    if (dense_res) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)n; ++i) {
            for (uint64_t j = 0; j < n; ++j) {
                T val = op(static_cast<T>(a.get_element_as_double(i, j)), 
                           static_cast<T>(b.get_element_as_double(i, j)));
                dense_res->set(i, j, val);
            }
        }
        return;
    }

    throw std::runtime_error("Unknown result matrix type in execute_binary_op");
}

template <typename Op>
std::unique_ptr<MatrixBase> dispatch_binary_op(
    const MatrixBase& a, 
    const MatrixBase& b, 
    const std::string& result_file,
    Op op
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Matrix dimensions must match");
    }

    DataType type_a = a.get_data_type();
    DataType type_b = b.get_data_type();
    MatrixType mtype_a = a.get_matrix_type();
    MatrixType mtype_b = b.get_matrix_type();

    DataType res_dtype = MatrixFactory::resolve_result_type(type_a, type_b);
    MatrixType res_mtype = MatrixFactory::resolve_result_matrix_type(mtype_a, mtype_b);

    auto result = MatrixFactory::create(a.size(), res_dtype, res_mtype, result_file);

    switch (res_dtype) {
        case DataType::INT32:
            execute_binary_op<int32_t>(a, b, *result, op);
            break;
        case DataType::FLOAT64:
            execute_binary_op<double>(a, b, *result, op);
            break;
        default:
            throw std::runtime_error("Unsupported result data type");
    }

    return result;
}

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
    if (is_identity(a) && is_identity(b)) {
        const auto& a_id = static_cast<const IdentityMatrix&>(a);
        const auto& b_id = static_cast<const IdentityMatrix&>(b);
        return a_id.add(b_id, result_file);
    }

    return dispatch_binary_op(a, b, result_file, std::plus<>());
}

std::unique_ptr<MatrixBase> subtract(const MatrixBase& a, const MatrixBase& b, const std::string& result_file) {
    if (is_identity(a) && is_identity(b)) {
        const auto& a_id = static_cast<const IdentityMatrix&>(a);
        const auto& b_id = static_cast<const IdentityMatrix&>(b);
        return a_id.subtract(b_id, result_file);
    }

    return dispatch_binary_op(a, b, result_file, std::minus<>());
}

std::unique_ptr<MatrixBase> elementwise_multiply(const MatrixBase& a, const MatrixBase& b, const std::string& result_file) {
    if (is_identity(a) && is_identity(b)) {
        const auto& a_id = static_cast<const IdentityMatrix&>(a);
        const auto& b_id = static_cast<const IdentityMatrix&>(b);
        return a_id.elementwise_multiply(b_id, result_file);
    }

    return dispatch_binary_op(a, b, result_file, std::multiplies<>());
}

std::unique_ptr<VectorBase> add_vectors(const VectorBase& a, const VectorBase& b, const std::string& result_file) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions must match");
    }
    uint64_t n = a.size();
    
    // Check if both are integer-like (IntegerVector or BitVector)
    bool a_is_int = (dynamic_cast<const DenseVector<int32_t>*>(&a) != nullptr) || (dynamic_cast<const DenseVector<bool>*>(&a) != nullptr);
    bool b_is_int = (dynamic_cast<const DenseVector<int32_t>*>(&b) != nullptr) || (dynamic_cast<const DenseVector<bool>*>(&b) != nullptr);

    if (a_is_int && b_is_int) {
        auto result = std::make_unique<DenseVector<int32_t>>(n, result_file);
        for (uint64_t i = 0; i < n; ++i) {
            result->set(i, (int32_t)a.get_element_as_double(i) + (int32_t)b.get_element_as_double(i));
        }
        return result;
    }

    auto result = std::make_unique<DenseVector<double>>(n, result_file);
    for (uint64_t i = 0; i < n; ++i) {
        result->set(i, a.get_element_as_double(i) + b.get_element_as_double(i));
    }
    return result;
}

std::unique_ptr<VectorBase> subtract_vectors(const VectorBase& a, const VectorBase& b, const std::string& result_file) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions must match");
    }
    uint64_t n = a.size();

    // Check if both are integer-like
    bool a_is_int = (dynamic_cast<const DenseVector<int32_t>*>(&a) != nullptr) || (dynamic_cast<const DenseVector<bool>*>(&a) != nullptr);
    bool b_is_int = (dynamic_cast<const DenseVector<int32_t>*>(&b) != nullptr) || (dynamic_cast<const DenseVector<bool>*>(&b) != nullptr);

    if (a_is_int && b_is_int) {
        auto result = std::make_unique<DenseVector<int32_t>>(n, result_file);
        for (uint64_t i = 0; i < n; ++i) {
            result->set(i, (int32_t)a.get_element_as_double(i) - (int32_t)b.get_element_as_double(i));
        }
        return result;
    }

    auto result = std::make_unique<DenseVector<double>>(n, result_file);
    for (uint64_t i = 0; i < n; ++i) {
        result->set(i, a.get_element_as_double(i) - b.get_element_as_double(i));
    }
    return result;
}

double dot_product(const VectorBase& a, const VectorBase& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions must match");
    }
    uint64_t n = a.size();
    
    // Check if both are integer-like (IntegerVector or BitVector)
    bool a_is_int = (dynamic_cast<const DenseVector<int32_t>*>(&a) != nullptr) || (dynamic_cast<const DenseVector<bool>*>(&a) != nullptr);
    bool b_is_int = (dynamic_cast<const DenseVector<int32_t>*>(&b) != nullptr) || (dynamic_cast<const DenseVector<bool>*>(&b) != nullptr);

    // If both are integer-like, we should ideally return an integer.
    // However, the return type of this function is double.
    // In Python bindings, we can cast it back to int if it's an integer result.
    // But here in C++, we are constrained by the return type.
    // For now, we return double, but the calculation is done in double to avoid overflow?
    // Actually, dot product of integers can overflow int32 easily.
    // So returning double (or int64) is safer.
    // Let's stick to double for now as the signature is double.
    
    double sum = 0.0;
    for (uint64_t i = 0; i < n; ++i) {
        sum += a.get_element_as_double(i) * b.get_element_as_double(i);
    }
    return sum;
}

std::unique_ptr<VectorBase> scalar_multiply_vector(const VectorBase& v, double scalar, const std::string& result_file) {
    uint64_t n = v.size();
    auto result = std::make_unique<DenseVector<double>>(n, result_file);
    for (uint64_t i = 0; i < n; ++i) {
        result->set(i, v.get_element_as_double(i) * scalar);
    }
    return result;
}

std::unique_ptr<VectorBase> scalar_multiply_vector(const VectorBase& v, int64_t scalar, const std::string& result_file) {
    uint64_t n = v.size();
    
    bool v_is_int = (dynamic_cast<const DenseVector<int32_t>*>(&v) != nullptr) || (dynamic_cast<const DenseVector<bool>*>(&v) != nullptr);

    if (v_is_int) {
        auto result = std::make_unique<DenseVector<int32_t>>(n, result_file);
        for (uint64_t i = 0; i < n; ++i) {
            result->set(i, (int32_t)v.get_element_as_double(i) * (int32_t)scalar);
        }
        return result;
    }

    auto result = std::make_unique<DenseVector<double>>(n, result_file);
    for (uint64_t i = 0; i < n; ++i) {
        result->set(i, v.get_element_as_double(i) * (double)scalar);
    }
    return result;
}

std::unique_ptr<VectorBase> scalar_add_vector(const VectorBase& v, double scalar, const std::string& result_file) {
    uint64_t n = v.size();
    auto result = std::make_unique<DenseVector<double>>(n, result_file);
    for (uint64_t i = 0; i < n; ++i) {
        result->set(i, v.get_element_as_double(i) + scalar);
    }
    return result;
}

std::unique_ptr<VectorBase> scalar_add_vector(const VectorBase& v, int64_t scalar, const std::string& result_file) {
    uint64_t n = v.size();
    
    bool v_is_int = (dynamic_cast<const DenseVector<int32_t>*>(&v) != nullptr) || (dynamic_cast<const DenseVector<bool>*>(&v) != nullptr);

    if (v_is_int) {
        auto result = std::make_unique<DenseVector<int32_t>>(n, result_file);
        for (uint64_t i = 0; i < n; ++i) {
            result->set(i, (int32_t)v.get_element_as_double(i) + (int32_t)scalar);
        }
        return result;
    }

    auto result = std::make_unique<DenseVector<double>>(n, result_file);
    for (uint64_t i = 0; i < n; ++i) {
        result->set(i, v.get_element_as_double(i) + (double)scalar);
    }
    return result;
}

std::unique_ptr<MatrixBase> outer_product(const VectorBase& a, const VectorBase& b, const std::string& result_file) {
    if (a.size() != b.size()) throw std::invalid_argument("Vectors must be same size");
    uint64_t n = a.size();

    DataType type_a = a.get_data_type();
    DataType type_b = b.get_data_type();
    DataType res_dtype = MatrixFactory::resolve_result_type(type_a, type_b);
    
    std::unique_ptr<MatrixBase> result;
    
    if (res_dtype == DataType::BIT) {
        auto m = std::make_unique<DenseBitMatrix>(n, result_file);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)n; ++i) {
            double val_a = a.get_element_as_double(i);
            if (val_a == 0.0) continue; 
            for (uint64_t j = 0; j < n; ++j) {
                double val_b = b.get_element_as_double(j);
                if (val_b != 0.0) {
                    m->set(i, j, true);
                }
            }
        }
        result = std::move(m);
    } else if (res_dtype == DataType::INT32) {
        auto m = std::make_unique<IntegerMatrix>(n, result_file);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)n; ++i) {
            int32_t val_a = (int32_t)a.get_element_as_double(i);
            for (uint64_t j = 0; j < n; ++j) {
                m->set(i, j, val_a * (int32_t)b.get_element_as_double(j));
            }
        }
        result = std::move(m);
    } else {
        auto m = std::make_unique<FloatMatrix>(n, result_file);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)n; ++i) {
            double val_a = a.get_element_as_double(i);
            for (uint64_t j = 0; j < n; ++j) {
                m->set(i, j, val_a * b.get_element_as_double(j));
            }
        }
        result = std::move(m);
    }
    return result;
}

std::unique_ptr<VectorBase> matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, const std::string& result_file) {
    if (m.size() != v.size()) throw std::invalid_argument("Dimension mismatch");
    uint64_t n = m.size();
    
    DataType type_m = m.get_data_type();
    DataType type_v = v.get_data_type();
    DataType res_dtype = MatrixFactory::resolve_result_type(type_m, type_v);
    
    if (res_dtype == DataType::BIT) res_dtype = DataType::INT32;
    
    if (res_dtype == DataType::INT32) {
        auto res = std::make_unique<DenseVector<int32_t>>(n, result_file);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)n; ++i) {
            int32_t sum = 0;
            for (uint64_t j = 0; j < n; ++j) {
                sum += (int32_t)(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res->set(i, sum);
        }
        return res;
    } else {
        auto res = std::make_unique<DenseVector<double>>(n, result_file);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)n; ++i) {
            double sum = 0.0;
            for (uint64_t j = 0; j < n; ++j) {
                sum += m.get_element_as_double(i, j) * v.get_element_as_double(j);
            }
            res->set(i, sum);
        }
        return res;
    }
}

std::unique_ptr<VectorBase> vector_matrix_multiply(const VectorBase& v, const MatrixBase& m, const std::string& result_file) {
    if (m.size() != v.size()) throw std::invalid_argument("Dimension mismatch");
    uint64_t n = m.size();
    
    DataType type_m = m.get_data_type();
    DataType type_v = v.get_data_type();
    DataType res_dtype = MatrixFactory::resolve_result_type(type_m, type_v);
    if (res_dtype == DataType::BIT) res_dtype = DataType::INT32;

    std::unique_ptr<VectorBase> res;
    if (res_dtype == DataType::INT32) {
        auto r = std::make_unique<DenseVector<int32_t>>(n, result_file);
        #pragma omp parallel for schedule(static)
        for (int64_t j = 0; j < (int64_t)n; ++j) {
            int32_t sum = 0;
            for (uint64_t i = 0; i < n; ++i) {
                sum += (int32_t)(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            r->set(j, sum);
        }
        res = std::move(r);
    } else {
        auto r = std::make_unique<DenseVector<double>>(n, result_file);
        #pragma omp parallel for schedule(static)
        for (int64_t j = 0; j < (int64_t)n; ++j) {
            double sum = 0.0;
            for (uint64_t i = 0; i < n; ++i) {
                sum += v.get_element_as_double(i) * m.get_element_as_double(i, j);
            }
            r->set(j, sum);
        }
        res = std::move(r);
    }
    
    res->set_transposed(true); 
    return res;
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