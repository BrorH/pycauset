#include "TriangularMatrix.hpp"
#include <cmath>

uint64_t TriangularMatrixBase::calculate_triangular_offsets(uint64_t element_bits, uint64_t alignment_bits) {
    row_offsets_.resize(n_);
    uint64_t current_offset_bits = 0;

    for (uint64_t i = 0; i < n_; ++i) {
        // Store current offset (converted to bytes)
        row_offsets_[i] = current_offset_bits / 8;

        // Number of elements in this row
        // If diagonal: columns i to N-1 => Count = N - i
        // If strict: columns i+1 to N-1 => Count = N - 1 - i
        uint64_t num_elements;
        if (has_diagonal_) {
            num_elements = (n_ > i) ? (n_ - i) : 0;
        } else {
            num_elements = (n_ > i) ? (n_ - 1 - i) : 0;
        }
        
        if (num_elements > 0) {
            uint64_t row_bits = num_elements * element_bits;
            
            // Align row_bits to alignment_bits
            if (alignment_bits > 1) {
                uint64_t remainder = row_bits % alignment_bits;
                if (remainder != 0) {
                    row_bits += (alignment_bits - remainder);
                }
            }
            current_offset_bits += row_bits;
        }
    }
    
    // Return total size in bytes
    return (current_offset_bits + 7) / 8;
}
