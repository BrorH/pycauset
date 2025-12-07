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

// Helper template for executing binary operations
template <typename T, typename Op>
void execute_binary_op(const MatrixBase& a, const MatrixBase& b, MatrixBase& result, Op op) {
    uint64_t n = a.size();
    
    // Optimization: Dense<T> op Dense<T> -> Dense<T>
    auto* a_dense = dynamic_cast<const DenseMatrix<T>*>(&a);
    auto* b_dense = dynamic_cast<const DenseMatrix<T>*>(&b);
    auto* res_dense = dynamic_cast<DenseMatrix<T>*>(&result);

    if (a_dense && b_dense && res_dense) {
        const T* a_data = a_dense->data();
        const T* b_data = b_dense->data();
        T* res_data = res_dense->data();
        
        bool ta = a_dense->is_transposed();
        bool tb = b_dense->is_transposed();
        
        // Handle scalars (cast to T to avoid promotion if T is float/int)
        T sa = static_cast<T>(a_dense->get_scalar());
        T sb = static_cast<T>(b_dense->get_scalar());

        if (!ta && !tb) {
            // Fast path: contiguous memory
            ParallelFor(0, n * n, [&](size_t k) {
                res_data[k] = op(a_data[k] * sa, b_data[k] * sb);
            });
        } else {
            // Transposed path
            ParallelFor(0, n, [&](size_t i) {
                for (size_t j = 0; j < n; ++j) {
                    T val_a = (ta ? a_data[j * n + i] : a_data[i * n + j]) * sa;
                    T val_b = (tb ? b_data[j * n + i] : b_data[i * n + j]) * sb;
                    res_data[i * n + j] = op(val_a, val_b);
                }
            });
        }
        return;
    }

    // Try to cast to TriangularMatrix<T>
    auto* tri_res = dynamic_cast<TriangularMatrix<T>*>(&result);
    if (tri_res) {
        ParallelFor(0, n, [&](size_t i) {
            for (uint64_t j = i + 1; j < n; ++j) {
                T val = op(static_cast<T>(a.get_element_as_double(i, j)), 
                           static_cast<T>(b.get_element_as_double(i, j)));
                if (val != static_cast<T>(0)) {
                    tri_res->set(i, j, val);
                }
            }
        });
        return;
    }

    // Try to cast to DiagonalMatrix<T>
    auto* diag_res = dynamic_cast<DiagonalMatrix<T>*>(&result);
    if (diag_res) {
        // Check if it is actually IdentityMatrix (which throws on set)
        if (result.get_matrix_type() == MatrixType::IDENTITY) {
             // Handle Identity specially: compute scalar from first element
             // We assume uniform operation for Identity
             if (n > 0) {
                 T val = op(static_cast<T>(a.get_element_as_double(0, 0)), 
                            static_cast<T>(b.get_element_as_double(0, 0)));
                 result.set_scalar(static_cast<double>(val));
             }
             return;
        }

        ParallelFor(0, n, [&](size_t i) {
            T val = op(static_cast<T>(a.get_element_as_double(i, i)), 
                       static_cast<T>(b.get_element_as_double(i, i)));
            diag_res->set(i, i, val);
        });
        return;
    }

    // Fallback for mixed types (Dense result)
    if (res_dense) {
        ParallelFor(0, n, [&](size_t i) {
            for (uint64_t j = 0; j < n; ++j) {
                T val = op(static_cast<T>(a.get_element_as_double(i, j)), 
                           static_cast<T>(b.get_element_as_double(i, j)));
                res_dense->set(i, j, val);
            }
        });
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
        case DataType::FLOAT32:
            execute_binary_op<float>(a, b, *result, op);
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

    // GPU Acceleration
    if (ComputeContext::instance().is_gpu_active()) {
        DataType type_a = a.get_data_type();
        DataType type_b = b.get_data_type();
        // Only support same-type dense float/double for now
        if (type_a == type_b && (type_a == DataType::FLOAT64 || type_a == DataType::FLOAT32)) {
             // Check if both are dense (or compatible)
             // For now, just try to create result and call device
             try {
                 DataType res_dtype = type_a;
                 MatrixType res_mtype = MatrixType::DENSE_FLOAT;
                 auto result = MatrixFactory::create(a.size(), res_dtype, res_mtype, result_file);
                 ComputeContext::instance().get_device()->add(a, b, *result);
                 return result;
             } catch (...) {
                 // Fallback to CPU
             }
        }
    }

    return dispatch_binary_op(a, b, result_file, std::plus<>());
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

    // GPU Acceleration
    if (ComputeContext::instance().is_gpu_active()) {
        DataType type_a = a.get_data_type();
        DataType type_b = b.get_data_type();
        if (type_a == type_b && (type_a == DataType::FLOAT64 || type_a == DataType::FLOAT32)) {
             try {
                 DataType res_dtype = type_a;
                 MatrixType res_mtype = MatrixType::DENSE_FLOAT;
                 auto result = MatrixFactory::create(a.size(), res_dtype, res_mtype, result_file);
                 ComputeContext::instance().get_device()->subtract(a, b, *result);
                 return result;
             } catch (...) {
                 // Fallback to CPU
             }
        }
    }

    return dispatch_binary_op(a, b, result_file, std::minus<>());
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

    return dispatch_binary_op(a, b, result_file, std::multiplies<>());
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
        ParallelFor(0, n, [&](size_t i) {
            double val_a = a.get_element_as_double(i);
            if (val_a != 0.0) { 
                for (uint64_t j = 0; j < n; ++j) {
                    double val_b = b.get_element_as_double(j);
                    if (val_b != 0.0) {
                        m->set(i, j, true);
                    }
                }
            }
        });
        result = std::move(m);
    } else if (res_dtype == DataType::INT32) {
        auto m = std::make_unique<IntegerMatrix>(n, result_file);
        ParallelFor(0, n, [&](size_t i) {
            int32_t val_a = (int32_t)a.get_element_as_double(i);
            for (uint64_t j = 0; j < n; ++j) {
                m->set(i, j, val_a * (int32_t)b.get_element_as_double(j));
            }
        });
        result = std::move(m);
    } else {
        auto m = std::make_unique<FloatMatrix>(n, result_file);
        ParallelFor(0, n, [&](size_t i) {
            double val_a = a.get_element_as_double(i);
            for (uint64_t j = 0; j < n; ++j) {
                m->set(i, j, val_a * b.get_element_as_double(j));
            }
        });
        result = std::move(m);
    }
    return result;
}

std::unique_ptr<VectorBase> matrix_vector_multiply(const MatrixBase& m, const VectorBase& v, const std::string& result_file) {
    if (m.size() != v.size()) throw std::invalid_argument("Dimension mismatch");
    uint64_t n = m.size();

    if (m.get_matrix_type() == pycauset::MatrixType::IDENTITY) {
        double scalar = m.get_scalar();
        DataType res_dtype = MatrixFactory::resolve_result_type(m.get_data_type(), v.get_data_type());
        if (res_dtype == DataType::BIT) res_dtype = DataType::INT32;

        if (res_dtype == DataType::INT32) {
            auto res = std::make_unique<DenseVector<int32_t>>(n, result_file);
            ParallelFor(0, n, [&](size_t i) {
                res->set(i, (int32_t)(v.get_element_as_double(i) * scalar));
            });
            return res;
        } else {
            auto res = std::make_unique<DenseVector<double>>(n, result_file);
            ParallelFor(0, n, [&](size_t i) {
                res->set(i, v.get_element_as_double(i) * scalar);
            });
            return res;
        }
    }
    
    DataType type_m = m.get_data_type();
    DataType type_v = v.get_data_type();
    DataType res_dtype = MatrixFactory::resolve_result_type(type_m, type_v);
    
    if (res_dtype == DataType::BIT) res_dtype = DataType::INT32;
    
    if (res_dtype == DataType::INT32) {
        auto res = std::make_unique<DenseVector<int32_t>>(n, result_file);
        ParallelFor(0, n, [&](size_t i) {
            int32_t sum = 0;
            for (uint64_t j = 0; j < n; ++j) {
                sum += (int32_t)(m.get_element_as_double(i, j) * v.get_element_as_double(j));
            }
            res->set(i, sum);
        });
        return res;
    } else {
        auto res = std::make_unique<DenseVector<double>>(n, result_file);
        ParallelFor(0, n, [&](size_t i) {
            double sum = 0.0;
            for (uint64_t j = 0; j < n; ++j) {
                sum += m.get_element_as_double(i, j) * v.get_element_as_double(j);
            }
            res->set(i, sum);
        });
        return res;
    }
}

std::unique_ptr<VectorBase> vector_matrix_multiply(const VectorBase& v, const MatrixBase& m, const std::string& result_file) {
    if (m.size() != v.size()) throw std::invalid_argument("Dimension mismatch");
    uint64_t n = m.size();

    if (m.get_matrix_type() == pycauset::MatrixType::IDENTITY) {
        double scalar = m.get_scalar();
        DataType res_dtype = MatrixFactory::resolve_result_type(m.get_data_type(), v.get_data_type());
        if (res_dtype == DataType::BIT) res_dtype = DataType::INT32;

        std::unique_ptr<VectorBase> res;
        if (res_dtype == DataType::INT32) {
            auto r = std::make_unique<DenseVector<int32_t>>(n, result_file);
            ParallelFor(0, n, [&](size_t i) {
                r->set(i, (int32_t)(v.get_element_as_double(i) * scalar));
            });
            res = std::move(r);
        } else {
            auto r = std::make_unique<DenseVector<double>>(n, result_file);
            ParallelFor(0, n, [&](size_t i) {
                r->set(i, v.get_element_as_double(i) * scalar);
            });
            res = std::move(r);
        }
        res->set_transposed(true);
        return res;
    }
    
    DataType type_m = m.get_data_type();
    DataType type_v = v.get_data_type();
    DataType res_dtype = MatrixFactory::resolve_result_type(type_m, type_v);
    if (res_dtype == DataType::BIT) res_dtype = DataType::INT32;

    std::unique_ptr<VectorBase> res;
    if (res_dtype == DataType::INT32) {
        auto r = std::make_unique<DenseVector<int32_t>>(n, result_file);
        ParallelFor(0, n, [&](size_t j) {
            int32_t sum = 0;
            for (uint64_t i = 0; i < n; ++i) {
                sum += (int32_t)(v.get_element_as_double(i) * m.get_element_as_double(i, j));
            }
            r->set(j, sum);
        });
        res = std::move(r);
    } else {
        auto r = std::make_unique<DenseVector<double>>(n, result_file);
        ParallelFor(0, n, [&](size_t j) {
            double sum = 0.0;
            for (uint64_t i = 0; i < n; ++i) {
                sum += v.get_element_as_double(i) * m.get_element_as_double(i, j);
            }
            r->set(j, sum);
        });
        res = std::move(r);
    }
    
    res->set_transposed(true); 
    return res;
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