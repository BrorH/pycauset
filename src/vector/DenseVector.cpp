#include "pycauset/vector/DenseVector.hpp"
#include <cstring>
#include <memory>
#include <filesystem>
#include <algorithm>

namespace pycauset {

DenseVector<bool>::DenseVector(uint64_t n, const std::string& backing_file)
    : VectorBase(n) {
    // 1 bit per element, packed into 64-bit words. No row padding.
    uint64_t words = (n + 63) / 64;
    uint64_t size_in_bytes = words * 8;
    
    initialize_storage(size_in_bytes, backing_file, "vector_bit", 8, 
                      pycauset::MatrixType::VECTOR, pycauset::DataType::BIT,
                      n, 1);
}

DenseVector<bool>::DenseVector(uint64_t n, 
                               const std::string& backing_file,
                               size_t offset,
                               uint64_t seed,
                               std::complex<double> scalar,
                               bool is_transposed)
    : VectorBase(n) {
    uint64_t words = (n + 63) / 64;
    uint64_t size_in_bytes = words * 8;
    
    initialize_storage(size_in_bytes, backing_file, "", 8, 
                      pycauset::MatrixType::VECTOR, pycauset::DataType::BIT,
                      n, 1,
                      offset,
                      false);
                      
    set_seed(seed);
    set_scalar(scalar);
    set_transposed(is_transposed);
}

DenseVector<bool>::DenseVector(uint64_t n, std::shared_ptr<MemoryMapper> mapper)
    : VectorBase(n, std::move(mapper), pycauset::MatrixType::VECTOR, pycauset::DataType::BIT) {}

void DenseVector<bool>::set(uint64_t i, bool value) {
    if (i >= n_) throw std::out_of_range("Index out of bounds");

    uint64_t word_index = i / 64;
    uint64_t bit_index = i % 64;

    uint64_t* ptr = data();
    if (value) {
        ptr[word_index] |= (1ULL << bit_index);
    } else {
        ptr[word_index] &= ~(1ULL << bit_index);
    }
}

bool DenseVector<bool>::get(uint64_t i) const {
    if (i >= n_) throw std::out_of_range("Index out of bounds");

    uint64_t word_index = i / 64;
    uint64_t bit_index = i % 64;

    const uint64_t* ptr = data();
    return (ptr[word_index] >> bit_index) & 1ULL;
}

double DenseVector<bool>::get_element_as_double(uint64_t i) const {
    return get(i) ? scalar_.real() : 0.0;
}

std::unique_ptr<VectorBase> DenseVector<bool>::transpose(const std::string& saveas) const {
    (void)saveas;
    auto out = std::make_unique<DenseVector<bool>>(this->size(), this->mapper_);
    out->set_scalar(this->scalar_);
    out->set_seed(this->seed_);
    out->set_transposed(!this->is_transposed());
    out->set_conjugated(this->is_conjugated());
    out->set_temporary(this->is_temporary());
    return out;
}

std::unique_ptr<VectorBase> DenseVector<bool>::multiply_scalar(double factor, const std::string& result_file) const {
    // Copy storage
    std::string target = result_file;
    if (target.empty()) {
        target = pycauset::make_unique_storage_file("vector_bit_mul");
    }
    std::string new_path = this->copy_storage(target);
    
    // Create new vector wrapper
    uint64_t file_size = std::filesystem::file_size(new_path);
    auto mapper = std::make_unique<MemoryMapper>(new_path, file_size, 0, false);
    auto new_vec = std::make_unique<DenseVector<bool>>(n_, std::move(mapper));
    
    new_vec->set_scalar(scalar_ * factor);
    new_vec->set_seed(seed_);
    new_vec->set_transposed(is_transposed());
    
    if (result_file.empty()) {
        new_vec->set_temporary(true);
    }
    return new_vec;
}

std::unique_ptr<VectorBase> DenseVector<bool>::add_scalar(double scalar, const std::string& result_file) const {
    auto result = std::make_unique<DenseVector<double>>(n_, result_file);
    double* dst = result->data();
    
    double val_0 = scalar;
    double val_1 = (scalar_ + scalar).real();
    
    for (uint64_t i = 0; i < n_; ++i) {
        dst[i] = get(i) ? val_1 : val_0;
    }
    
    if (is_transposed()) {
        result->set_transposed(true);
    }
    return result;
}

} // namespace pycauset
