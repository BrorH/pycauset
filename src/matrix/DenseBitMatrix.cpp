#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/core/MemoryMapper.hpp"
#include "pycauset/compute/ComputeContext.hpp"
#include "pycauset/compute/ComputeDevice.hpp"
#include <stdexcept>
#include <bit>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <cstring>
#include <memory>

namespace pycauset {

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
    : MatrixBase(n, pycauset::MatrixType::DENSE_FLOAT, pycauset::DataType::BIT) {
    calculate_stride();
    uint64_t size_in_bytes = n_ * stride_bytes_;
    
    initialize_storage(size_in_bytes, backing_file, "dense_bit_matrix", 8, 
                      pycauset::MatrixType::DENSE_FLOAT, pycauset::DataType::BIT,
                      n, n);
}

DenseMatrix<bool>::DenseMatrix(uint64_t n, 
                               const std::string& backing_file,
                               size_t offset,
                               uint64_t seed,
                               double scalar,
                               bool is_transposed)
    : MatrixBase(n, pycauset::MatrixType::DENSE_FLOAT, pycauset::DataType::BIT) {
    calculate_stride();
    uint64_t size_in_bytes = n_ * stride_bytes_;
    
    initialize_storage(size_in_bytes, backing_file, "", 8, 
                      pycauset::MatrixType::DENSE_FLOAT, pycauset::DataType::BIT,
                      n, n, offset, false);
    
    set_seed(seed);
    set_scalar(scalar);
    set_transposed(is_transposed);
}

DenseMatrix<bool>::DenseMatrix(uint64_t n, std::shared_ptr<MemoryMapper> mapper)
    : MatrixBase(n, std::move(mapper), pycauset::MatrixType::DENSE_FLOAT, pycauset::DataType::BIT) {
    calculate_stride();
}

void DenseMatrix<bool>::set(uint64_t i, uint64_t j, bool value) {
    ensure_unique();
    if (i >= n_ || j >= n_) throw std::out_of_range("Index out of bounds");

    if (is_transposed()) {
        std::swap(i, j);
    }

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

    if (is_transposed()) {
        std::swap(i, j);
    }

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

    // Delegate to ComputeContext (AutoSolver)
    // AutoSolver will choose between GPU and CPU (CpuSolver)
    // CpuSolver now contains the optimized bit matrix multiplication logic
    pycauset::ComputeContext::instance().get_device()->matmul(*this, other, *result);
    
    return result;
}

std::unique_ptr<MatrixBase> DenseMatrix<bool>::multiply_scalar(double factor, const std::string& result_file) const {
    std::string new_path = copy_storage(result_file);
    auto mapper = std::make_unique<MemoryMapper>(new_path, 0, false);
    auto new_matrix = std::make_unique<DenseMatrix<bool>>(n_, std::move(mapper));
    new_matrix->set_scalar(scalar_ * factor);
    if (result_file.empty()) {
        new_matrix->set_temporary(true);
    }
    return new_matrix;
}

std::unique_ptr<MatrixBase> DenseMatrix<bool>::transpose(const std::string& result_file) const {
    std::string new_path = copy_storage(result_file);
    auto mapper = std::make_unique<MemoryMapper>(new_path, 0, false);
    auto new_matrix = std::make_unique<DenseMatrix<bool>>(n_, std::move(mapper));
    
    // Flip the transposed bit
    new_matrix->set_transposed(!this->is_transposed());
    
    if (result_file.empty()) {
        new_matrix->set_temporary(true);
    }
    return new_matrix;
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

std::unique_ptr<MatrixBase> DenseMatrix<bool>::add_scalar(double scalar, const std::string& result_file) const {
    auto result = std::make_unique<DenseMatrix<double>>(n_, result_file);
    double* dst_data = result->data();
    
    for (uint64_t i = 0; i < n_; ++i) {
        for (uint64_t j = 0; j < n_; ++j) {
            double val = (get(i, j) ? scalar_ : 0.0) + scalar;
            dst_data[i * n_ + j] = val;
        }
    }
    
    if (result_file.empty()) {
        result->set_temporary(true);
    }
    return result;
}

std::unique_ptr<MatrixBase> DenseMatrix<bool>::add_scalar(int64_t scalar, const std::string& result_file) const {
    return add_scalar(static_cast<double>(scalar), result_file);
}


} // namespace pycauset
