#pragma once

// Derived-value cache (review/features/DERIVED_CACHE.md, steps D.1-D.2): scalar results of
// whole-matrix computations (determinant, slogdet, trace, sum, norms, rank) are kept on the
// matrix's storage and dropped the moment the storage can change (MemoryMapper::bump_version,
// called by PersistentObject::ensure_unique before every write).
//
// Exactness: an entry is keyed on the exact view (class, structure, factor, conjugation,
// transposition) and on the parameters, so a cached answer is bit-for-bit what recomputing
// would return now. (The spec's view rules, deriving A * s from A's entry, are not used: they
// round differently from recomputation.) Windows (`m[1:4, 2:6]`) and objects without storage
// are not cached.

#include <any>
#include <atomic>
#include <cstdio>
#include <string>

#include "pycauset/core/DTypeDispatch.hpp"
#include "pycauset/matrix/BitClosureMatrix.hpp"
#include "pycauset/matrix/MatrixBase.hpp"
#include "pycauset/matrix/TriangularMatrix.hpp"
#include "pycauset/vector/VectorBase.hpp"

namespace pycauset {

namespace derived_cache {
// Counters for tests and the atlas (hits and stores since start); the switch is for timing
// a cold computation.
inline std::atomic<uint64_t>& hits() { static std::atomic<uint64_t> h{0}; return h; }
inline std::atomic<uint64_t>& stores() { static std::atomic<uint64_t> s{0}; return s; }
inline std::atomic<bool>& enabled() { static std::atomic<bool> e{true}; return e; }

// The key names the matrix by what makes its values differ for the same storage: dtype,
// structure (a C + C^T closure is int32 symmetric over C's bool triangular bits), a triangle's
// diagonal, and the view. Portable (no compiler type names): keys are saved with the file.
inline std::string view_key(const MatrixBase& m, const char* quantity, const std::string& params) {
    const std::complex<double> s = m.get_scalar();
    const auto* tri = dynamic_cast<const TriangularMatrixBase*>(&m);
    char buf[160];
    // %a: the factor exactly (hex float), so 0.1 and 0.1000000000001 are different keys.
    std::snprintf(buf, sizeof buf, "|%s|%d|%d|%a|%a|%d|%d", dtype_name(m.get_data_type()),
                  static_cast<int>(m.get_structure()), tri && tri->has_diagonal() ? 1 : 0, s.real(), s.imag(),
                  m.is_conjugated() ? 1 : 0, m.is_transposed() ? 1 : 0);
    return std::string(quantity) + "|" + params + buf;
}
}  // namespace derived_cache

// compute() through the cache of m's storage. V must be copyable.
template <typename V, typename F>
V cached(const MatrixBase& m, const char* quantity, const std::string& params, F&& compute) {
    if (!derived_cache::enabled().load(std::memory_order_relaxed) || m.is_subview()) return compute();
    // A written lazy closure holds its own values (materialized) but keeps C's bit storage as
    // its mapper, shared with C's other closures: no entry, or a write to one closure would
    // answer for every other (review 3.2 C-1).
    if (const auto* cl = dynamic_cast<const BitClosureMatrix*>(&m); cl && cl->is_materialized()) return compute();
    const auto mapper = m.shared_mapper();
    if (!mapper) return compute();
    const std::string key = derived_cache::view_key(m, quantity, params);
    std::any hit;
    if (mapper->cache_get(key, hit)) {
        if (const V* v = std::any_cast<V>(&hit)) {
            derived_cache::hits().fetch_add(1, std::memory_order_relaxed);
            return *v;
        }
    }
    const uint64_t version = mapper->version();
    V value = compute();
    mapper->cache_put(key, version, std::any(value));
    derived_cache::stores().fetch_add(1, std::memory_order_relaxed);
    return value;
}

// Vector-valued results (eigenvalues): the cache holds one copy-on-write clone of the result
// and every call returns a new clone of it (O(1): the clones share the storage, and a write to
// one copies it first, so the cached vector never changes).
template <typename F>
std::unique_ptr<VectorBase> cached_vector(const MatrixBase& m, const char* quantity, F&& compute) {
    auto clone_of = [](const VectorBase& v) {
        return std::unique_ptr<VectorBase>(static_cast<VectorBase*>(v.clone().release()));
    };
    std::shared_ptr<const VectorBase> held;
    std::unique_ptr<VectorBase> fresh;
    held = cached<std::shared_ptr<const VectorBase>>(m, quantity, "", [&] {
        fresh = compute();
        return std::shared_ptr<const VectorBase>(clone_of(*fresh));
    });
    if (fresh) return fresh;  // computed now: the caller gets the original
    return clone_of(*held);
}

}  // namespace pycauset
