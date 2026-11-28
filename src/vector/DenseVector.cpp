#include "pycauset/vector/DenseVector.hpp"
#include <cstring>
#include <memory>
#include <filesystem>
#include <algorithm>

namespace pycauset {

DenseVector<bool>::DenseVector(uint64_t n, const std::string& backing_file)
    : DenseVectorBase(n) {
    // Padded to 64 bytes like a bit-matrix row, so the vector is also a 1 x n bit matrix
    // over the same storage (math/VectorOps.hpp).
    uint64_t size_in_bytes = (((n + 63) / 64) * 8 + 63) & ~63ULL;
    
    initialize_storage(size_in_bytes, backing_file, "vector_bit", 8, 
                      pycauset::Structure::VECTOR, pycauset::DataType::BIT,
                      n, 1);
}

DenseVector<bool>::DenseVector(uint64_t n, 
                               const std::string& backing_file,
                               size_t offset,
                               uint64_t seed,
                               std::complex<double> scalar,
                               bool is_transposed)
    : DenseVectorBase(n) {
    // Padded to 64 bytes like a bit-matrix row, so the vector is also a 1 x n bit matrix
    // over the same storage (math/VectorOps.hpp).
    uint64_t size_in_bytes = (((n + 63) / 64) * 8 + 63) & ~63ULL;
    
    initialize_storage(size_in_bytes, backing_file, "", 8, 
                      pycauset::Structure::VECTOR, pycauset::DataType::BIT,
                      n, 1,
                      offset,
                      false);
                      
    set_seed(seed);
    set_scalar(scalar);
    set_transposed(is_transposed);
}

DenseVector<bool>::DenseVector(uint64_t n, std::shared_ptr<MemoryMapper> mapper)
    : DenseVectorBase(n, std::move(mapper), pycauset::Structure::VECTOR, pycauset::DataType::BIT) {}

void DenseVector<bool>::set(uint64_t i, bool value) {
    if (i >= n_) throw std::out_of_range("Index out of bounds");
    ensure_unique();  // copy-on-write (and the derived-value cache's invalidation gate)

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

double DenseVector<bool>::element_as_double(uint64_t i) const {
    return get(i) ? scalar_.real() : 0.0;
}

std::unique_ptr<VectorBase> DenseVector<bool>::transpose(const std::string& saveas) const {
    (void)saveas;
    auto out = std::make_unique<DenseVector<bool>>(this->size(), this->mapper_);
    out->set_scalar(this->scalar_);
    out->set_seed(this->seed_);
    out->set_transposed(!this->is_transposed());
    out->set_conjugated(this->is_conjugated());
    return out;
}



} // namespace pycauset
