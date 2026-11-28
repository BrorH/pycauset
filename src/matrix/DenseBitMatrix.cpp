#include "pycauset/matrix/DenseBitMatrix.hpp"
#include "pycauset/math/ViewValue.hpp"
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

void DenseMatrix<bool>::calculate_stride() {
    // Align rows to 512-bit (64-byte) boundaries for AVX-512 optimization.
    // Even if AVX-512 is not used, cache line alignment (64 bytes) is beneficial.
    uint64_t min_bytes = (base_cols() + 7) / 8;
    stride_bytes_ = (min_bytes + 63) & ~63ULL;
}

DenseMatrix<bool>::DenseMatrix(uint64_t n, const std::string& backing_file)
    : DenseMatrix(n, n, backing_file) {}

DenseMatrix<bool>::DenseMatrix(uint64_t rows, uint64_t cols, const std::string& backing_file)
    : DenseMatrixBase(rows, cols, pycauset::Structure::DENSE, pycauset::DataType::BIT) {
    calculate_stride();
    const uint64_t size_in_bytes = base_rows() * stride_bytes_;

    initialize_storage(size_in_bytes,
                       backing_file,
                       "dense_bit_matrix",
                       8,
                       pycauset::Structure::DENSE,
                       pycauset::DataType::BIT,
                       rows,
                       cols);
}

DenseMatrix<bool>::DenseMatrix(uint64_t n, 
                               const std::string& backing_file,
                               size_t offset,
                               uint64_t seed,
                               std::complex<double> scalar,
                               bool is_transposed)
    : DenseMatrix(n, n, backing_file, offset, seed, scalar, is_transposed) {}

DenseMatrix<bool>::DenseMatrix(uint64_t rows,
                               uint64_t cols,
                               const std::string& backing_file,
                               size_t offset,
                               uint64_t seed,
                               std::complex<double> scalar,
                               bool is_transposed)
    : DenseMatrixBase(rows, cols, pycauset::Structure::DENSE, pycauset::DataType::BIT) {
    calculate_stride();
    const uint64_t size_in_bytes = base_rows() * stride_bytes_;

    initialize_storage(size_in_bytes,
                       backing_file,
                       "",
                       8,
                       pycauset::Structure::DENSE,
                       pycauset::DataType::BIT,
                       rows,
                       cols,
                       offset,
                       false);

    set_seed(seed);
    set_scalar(scalar);
    set_transposed(is_transposed);
}

DenseMatrix<bool>::DenseMatrix(uint64_t n, std::shared_ptr<MemoryMapper> mapper)
    : DenseMatrix(n, n, std::move(mapper)) {}

DenseMatrix<bool>::DenseMatrix(uint64_t rows, uint64_t cols, std::shared_ptr<MemoryMapper> mapper)
    : DenseMatrixBase(rows, cols, std::move(mapper), pycauset::Structure::DENSE, pycauset::DataType::BIT) {
    calculate_stride();
}

void DenseMatrix<bool>::fill(bool value) {
    ensure_unique();
    set_conjugated(false);  // conj of real values is a no-op flag; the fill replaces them

    const uint64_t words_per_row = stride_bytes_ / 8;
    const uint64_t total_words = base_rows() * words_per_row;
    uint64_t* dst = data();

    const uint64_t fill_word = value ? ~0ULL : 0ULL;
    for (uint64_t idx = 0; idx < total_words; ++idx) {
        dst[idx] = fill_word;
    }

    const uint64_t tail_bits = base_cols() % 64;
    if (value && tail_bits != 0) {
        const uint64_t mask = (1ULL << tail_bits) - 1ULL;
        for (uint64_t i = 0; i < base_rows(); ++i) {
            dst[i * words_per_row + (words_per_row - 1)] &= mask;
        }
    }

}

void DenseMatrix<bool>::set(uint64_t i, uint64_t j, bool value) {
    ensure_unique();
    if (i >= rows() || j >= cols()) throw std::out_of_range("Index out of bounds");

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
    if (i >= rows() || j >= cols()) throw std::out_of_range("Index out of bounds");

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

double DenseMatrix<bool>::element_as_double(uint64_t i, uint64_t j) const {
    return pycauset::view_value(get(i, j), scalar_, false).real();  // False * -1.0 is -0.0
}

std::unique_ptr<MatrixBase> DenseMatrix<bool>::multiply_scalar(double factor, const std::string& result_file) const {
    (void)result_file;
    return scaled_clone(factor);
}

std::unique_ptr<MatrixBase> DenseMatrix<bool>::transpose(const std::string& result_file) const {
    (void)result_file;
    return transposed_clone();
}

std::unique_ptr<DenseMatrix<bool>> DenseMatrix<bool>::bitwise_not(const std::string& result_file) const {
    auto result = std::make_unique<DenseMatrix<bool>>(base_rows(), base_cols(), result_file);
    
    const uint64_t* src = data();
    uint64_t* dst = result->data();
    uint64_t total_words = (base_rows() * stride_bytes_) / 8;
    
    for (uint64_t i = 0; i < total_words; ++i) {
        dst[i] = ~src[i];
    }

    // Mask out padding bits in the last word of each row.
    const uint64_t tail_bits = base_cols() % 64;
    if (tail_bits != 0) {
        const uint64_t mask = (1ULL << tail_bits) - 1ULL;
        const uint64_t words_per_row = stride_bytes_ / 8;
        for (uint64_t i = 0; i < base_rows(); ++i) {
            dst[i * words_per_row + (words_per_row - 1)] &= mask;
        }
    }
    
    result->set_scalar(scalar_);
    return result;
}

std::unique_ptr<DenseMatrix<bool>> DenseMatrix<bool>::random(uint64_t n, double density,
                                            const std::string& backing_file,
                                            std::optional<uint64_t> seed) {
    return random(n, n, density, backing_file, seed);
}

std::unique_ptr<DenseMatrix<bool>> DenseMatrix<bool>::random(uint64_t rows, uint64_t cols, double density,
                                            const std::string& backing_file,
                                            std::optional<uint64_t> seed) {
    auto mat = std::make_unique<DenseMatrix<bool>>(rows, cols, backing_file);
    mat->fill_random(density, seed);
    return mat;
}

void DenseMatrix<bool>::fill_random(double density, std::optional<uint64_t> seed) {
    if (!(density >= 0.0 && density <= 1.0)) throw std::invalid_argument("p must be in [0, 1]");  // NaN too
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
    const uint64_t r = rows();
    const uint64_t c = cols();
    for (uint64_t i = 0; i < r; ++i) {
        for (uint64_t j = 0; j < c; ++j) {
            if (d(gen)) {
                set(i, j, true);
            }
        }
    }
}




} // namespace pycauset
