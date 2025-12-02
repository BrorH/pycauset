#include "Sprinkler.hpp"
#include "TriangularBitMatrix.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>

namespace pycauset {

std::unique_ptr<MatrixBase> pycauset::Sprinkler::sprinkle(
    const pycauset::CausalSpacetime& spacetime, 
    uint64_t n, 
    uint64_t seed,
    const std::string& saveas
) {
    auto matrix = std::make_unique<TriangularMatrix<bool>>(n, saveas);
    
    // Block size for coordinate generation.
    // 10,000 points * 2 doubles * 8 bytes = 160KB.
    // This is small enough to fit in L2 cache.
    const uint64_t BLOCK_SIZE = 10000; 
    uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Deterministic seed generator for blocks
    auto get_block_seed = [seed](uint64_t block_idx) {
        uint64_t z = seed + block_idx * 0x9e3779b97f4a7c15;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        return z ^ (z >> 31);
    };

    for (uint64_t i_block = 0; i_block < num_blocks; ++i_block) {
        uint64_t i_start = i_block * BLOCK_SIZE;
        uint64_t i_end = std::min(i_start + BLOCK_SIZE, n);
        
        // Generate coordinates for block I
        std::mt19937_64 rng_i(get_block_seed(i_block));
        std::vector<std::vector<double>> coords_i(i_end - i_start);
        for(auto& p : coords_i) p = spacetime.generate_point(rng_i);
        
        // Compare with block J (where J >= I)
        for (uint64_t j_block = i_block; j_block < num_blocks; ++j_block) {
            uint64_t j_start = j_block * BLOCK_SIZE;
            uint64_t j_end = std::min(j_start + BLOCK_SIZE, n);
            
            std::vector<std::vector<double>> coords_j;
            
            if (i_block == j_block) {
                // Same block, use coords_i
            } else {
                // Generate coords for block J
                std::mt19937_64 rng_j(get_block_seed(j_block));
                coords_j.resize(j_end - j_start);
                for(auto& p : coords_j) p = spacetime.generate_point(rng_j);
            }
            
            const auto& target_coords_j = (i_block == j_block) ? coords_i : coords_j;

            for (uint64_t i = i_start; i < i_end; ++i) {
                // If same block, start j from i+1
                uint64_t j_loop_start = (i_block == j_block) ? (i + 1) : j_start;
                
                for (uint64_t j = j_loop_start; j < j_end; ++j) {
                    const auto& p_i = coords_i[i - i_start];
                    const auto& p_j = target_coords_j[j - j_start];
                    
                    if (spacetime.causality(p_i, p_j)) {
                        matrix->set(i, j, true);
                    }
                }
            }
        }
    }
    
    return matrix;
}

}
