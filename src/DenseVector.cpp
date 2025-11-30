#include "DenseVector.hpp"
#include <cstring>

DenseVector<bool>::DenseVector(uint64_t n, const std::string& backing_file)
    : VectorBase(n) {
    // 1 bit per element, packed into 64-bit words. No row padding.
    uint64_t words = (n + 63) / 64;
    uint64_t size_in_bytes = words * 8;
    
    initialize_storage(size_in_bytes, backing_file, "vector_bit", 8, 
                      pycauset::MatrixType::VECTOR, pycauset::DataType::BIT,
                      n, 1);
}

DenseVector<bool>::DenseVector(uint64_t n, std::unique_ptr<MemoryMapper> mapper)
    : VectorBase(n, std::move(mapper)) {}

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
    return get(i) ? scalar_ : 0.0;
}
