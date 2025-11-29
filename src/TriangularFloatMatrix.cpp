#include "TriangularFloatMatrix.hpp"
#include <stdexcept>

TriangularFloatMatrix::TriangularFloatMatrix(uint64_t n, const std::string& backing_file)
    : TriangularMatrix(n) {
    // Calculate offsets for doubles (64 bits per element), aligned to 64 bits (natural alignment)
    uint64_t size_in_bytes = calculate_triangular_offsets(64, 64);
    initialize_storage(size_in_bytes, backing_file, "triangular_float_matrix", 8,
                      pycauset::MatrixType::TRIANGULAR_FLOAT, pycauset::DataType::FLOAT64);
}

TriangularFloatMatrix::TriangularFloatMatrix(uint64_t n, std::unique_ptr<MemoryMapper> mapper)
    : TriangularMatrix(n, std::move(mapper)) {
    calculate_triangular_offsets(64, 64);
}

void TriangularFloatMatrix::set(uint64_t i, uint64_t j, double value) {
    if (i >= j) throw std::invalid_argument("Strictly upper triangular");
    if (j >= n_) throw std::out_of_range("Index out of bounds");

    // Row i starts at row_offsets_[i] (in bytes)
    // It represents columns i+1 to N-1
    // The element for column j is at index (j - (i + 1)) in this row
    
    uint64_t row_offset_bytes = get_row_offset(i);
    uint64_t col_index = j - (i + 1);
    
    // Pointer arithmetic: base + row_offset + col_index * sizeof(double)
    char* base_ptr = static_cast<char*>(require_mapper()->get_data());
    double* row_ptr = reinterpret_cast<double*>(base_ptr + row_offset_bytes);
    
    row_ptr[col_index] = value;
}

double TriangularFloatMatrix::get(uint64_t i, uint64_t j) const {
    if (i >= j) return 0.0; // Implicit zero
    if (j >= n_) throw std::out_of_range("Index out of bounds");

    uint64_t row_offset_bytes = get_row_offset(i);
    uint64_t col_index = j - (i + 1);
    
    const char* base_ptr = static_cast<const char*>(require_mapper()->get_data());
    const double* row_ptr = reinterpret_cast<const double*>(base_ptr + row_offset_bytes);
    
    return row_ptr[col_index];
}
