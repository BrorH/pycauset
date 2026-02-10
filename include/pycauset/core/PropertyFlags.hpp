#pragma once

#include <cstdint>

namespace pycauset {
namespace properties {

// Bitmask flags for fast property mirroring in C++.
// These correspond to boolean properties exposed in python/pycauset/_internal/properties.py.
enum class PropertyFlag : uint64_t {
    IsZero = 1ULL << 0,
    IsIdentity = 1ULL << 1,
    IsPermutation = 1ULL << 2,
    IsDiagonal = 1ULL << 3,
    IsUpperTriangular = 1ULL << 4,
    IsLowerTriangular = 1ULL << 5,
    HasUnitDiagonal = 1ULL << 6,
    HasZeroDiagonal = 1ULL << 7,
    IsSymmetric = 1ULL << 8,
    IsAntiSymmetric = 1ULL << 9,
    IsHermitian = 1ULL << 10,
    IsSkewHermitian = 1ULL << 11,
    IsUnitary = 1ULL << 12,
    IsAtomic = 1ULL << 13,
    IsSorted = 1ULL << 14,
    IsStrictlySorted = 1ULL << 15,
    IsUnitNorm = 1ULL << 16,
};

inline constexpr uint64_t to_mask(PropertyFlag flag) {
    return static_cast<uint64_t>(flag);
}

} // namespace properties
} // namespace pycauset
