#pragma once

#include <cstdint>

#include "pycauset/core/PropertyFlags.hpp"
#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/vector/VectorBase.hpp"

namespace pycauset {

struct MatrixPropertyTraits {
    bool is_zero = false;
    bool is_identity = false;
    bool is_permutation = false;
    bool is_diagonal = false;
    bool is_upper_triangular = false;
    bool is_lower_triangular = false;
    bool has_unit_diagonal = false;
    bool has_zero_diagonal = false;
    bool is_symmetric = false;
    bool is_anti_symmetric = false;
    bool is_hermitian = false;
    bool is_skew_hermitian = false;
    bool is_unitary = false;
    bool is_atomic = false;
    bool is_sorted = false;
    bool is_strictly_sorted = false;
    bool is_unit_norm = false;

    static MatrixPropertyTraits from_flags(uint64_t flags) {
        MatrixPropertyTraits t;
        t.is_zero = (flags & properties::to_mask(properties::PropertyFlag::IsZero)) != 0;
        t.is_identity = (flags & properties::to_mask(properties::PropertyFlag::IsIdentity)) != 0;
        t.is_permutation = (flags & properties::to_mask(properties::PropertyFlag::IsPermutation)) != 0;
        t.is_diagonal = (flags & properties::to_mask(properties::PropertyFlag::IsDiagonal)) != 0;
        t.is_upper_triangular = (flags & properties::to_mask(properties::PropertyFlag::IsUpperTriangular)) != 0;
        t.is_lower_triangular = (flags & properties::to_mask(properties::PropertyFlag::IsLowerTriangular)) != 0;
        t.has_unit_diagonal = (flags & properties::to_mask(properties::PropertyFlag::HasUnitDiagonal)) != 0;
        t.has_zero_diagonal = (flags & properties::to_mask(properties::PropertyFlag::HasZeroDiagonal)) != 0;
        t.is_symmetric = (flags & properties::to_mask(properties::PropertyFlag::IsSymmetric)) != 0;
        t.is_anti_symmetric = (flags & properties::to_mask(properties::PropertyFlag::IsAntiSymmetric)) != 0;
        t.is_hermitian = (flags & properties::to_mask(properties::PropertyFlag::IsHermitian)) != 0;
        t.is_skew_hermitian = (flags & properties::to_mask(properties::PropertyFlag::IsSkewHermitian)) != 0;
        t.is_unitary = (flags & properties::to_mask(properties::PropertyFlag::IsUnitary)) != 0;
        t.is_atomic = (flags & properties::to_mask(properties::PropertyFlag::IsAtomic)) != 0;
        t.is_sorted = (flags & properties::to_mask(properties::PropertyFlag::IsSorted)) != 0;
        t.is_strictly_sorted = (flags & properties::to_mask(properties::PropertyFlag::IsStrictlySorted)) != 0;
        t.is_unit_norm = (flags & properties::to_mask(properties::PropertyFlag::IsUnitNorm)) != 0;
        return t;
    }
};

inline MatrixPropertyTraits traits_for(const MatrixBase& m) {
    return MatrixPropertyTraits::from_flags(m.get_properties_flags());
}

inline MatrixPropertyTraits traits_for(const VectorBase& v) {
    return MatrixPropertyTraits::from_flags(v.get_properties_flags());
}

} // namespace pycauset
