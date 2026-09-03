#include "pycauset/causet/Sprinkler.hpp"
#include "pycauset/matrix/TriangularBitMatrix.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace pycauset {

namespace {

// Distinct message used to signal an interrupt request up to the Python binding
// (which then translates it into KeyboardInterrupt).
constexpr const char* kInterrupted = "pycauset: interrupted";

// Deterministic per-block seed (identical to the original scheme, so the RNG
// stream for a given (spacetime, seed) is unchanged).
uint64_t block_seed(uint64_t seed, uint64_t block_idx) {
    uint64_t z = seed + block_idx * 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

void check_abort(const Sprinkler::AbortCheck& should_abort) {
    if (should_abort && should_abort()) {
        throw std::runtime_error(kInterrupted);
    }
}

// Generate all n points and return their time-sorted order (ascending
// coordinate 0). `order[k]` is the generation index of the k-th point in time
// order, so the causal matrix (which stores the strictly-upper triangle) is
// indexed by time, matching the causal-set labelling convention.
//
// This is essential: a causal relation u < v requires u[0] < v[0], so sorting
// by coordinate 0 guarantees every relation lands in the upper triangle.
// Without the sort, relations whose past endpoint happened to be generated
// later are silently dropped (they would be in the lower triangle).
std::vector<uint64_t> generate_time_sorted(
    const CausalSpacetime& spacetime,
    uint64_t n,
    uint64_t seed,
    std::vector<std::vector<double>>& coords,
    const Sprinkler::AbortCheck& should_abort
) {
    const uint64_t BLOCK_SIZE = 10000;
    const uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    coords.resize(n);
    for (uint64_t b = 0; b < num_blocks; ++b) {
        check_abort(should_abort);
        const uint64_t start = b * BLOCK_SIZE;
        const uint64_t end = std::min(start + BLOCK_SIZE, n);
        std::mt19937_64 rng(block_seed(seed, b));
        for (uint64_t i = start; i < end; ++i) {
            coords[i] = spacetime.generate_point(rng);
        }
    }

    std::vector<uint64_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](uint64_t a, uint64_t b) {
        return coords[a][0] < coords[b][0];
    });
    return order;
}

} // namespace

std::unique_ptr<MatrixBase> Sprinkler::sprinkle(
    const CausalSpacetime& spacetime,
    uint64_t n,
    uint64_t seed,
    const std::string& saveas,
    const AbortCheck& should_abort
) {
    std::vector<std::vector<double>> coords;
    const std::vector<uint64_t> order = generate_time_sorted(spacetime, n, seed, coords, should_abort);

    auto matrix = std::make_unique<TriangularMatrix<bool>>(n, saveas);
    for (uint64_t i = 0; i < n; ++i) {
        // Poll the abort hook frequently (every 64 outer rows) so Ctrl+C stays
        // responsive even while the O(n^2) causality loop is running.
        if ((i & 0x3F) == 0) {
            check_abort(should_abort);
        }
        for (uint64_t j = i + 1; j < n; ++j) {
            if (spacetime.causality(coords[order[i]], coords[order[j]])) {
                matrix->set(i, j, true);
            }
        }
    }
    return matrix;
}

std::vector<std::vector<double>> Sprinkler::make_coordinates(
    const CausalSpacetime& spacetime,
    uint64_t n,
    uint64_t seed,
    std::vector<uint64_t> indices,
    const AbortCheck& should_abort
) {
    std::vector<std::vector<double>> coords;
    const std::vector<uint64_t> order = generate_time_sorted(spacetime, n, seed, coords, should_abort);

    std::vector<std::vector<double>> results;
    results.reserve(indices.size());
    for (const uint64_t idx : indices) {
        if (idx < n) {
            results.push_back(coords[order[idx]]);
        }
    }
    return results;
}

} // namespace pycauset
