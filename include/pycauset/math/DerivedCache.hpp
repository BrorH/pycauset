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
#include <typeinfo>

#include "pycauset/matrix/MatrixBase.hpp"

namespace pycauset {

namespace derived_cache {
// Counters for tests and the atlas (hits and stores since start); the switch is for timing
// a cold computation.
inline std::atomic<uint64_t>& hits() { static std::atomic<uint64_t> h{0}; return h; }
inline std::atomic<uint64_t>& stores() { static std::atomic<uint64_t> s{0}; return s; }
inline std::atomic<bool>& enabled() { static std::atomic<bool> e{true}; return e; }

inline std::string view_key(const MatrixBase& m, const char* quantity, const std::string& params) {
    const std::complex<double> s = m.get_scalar();
    char buf[96];
    // %a: the factor exactly (hex float), so 0.1 and 0.1000000000001 are different keys.
    std::snprintf(buf, sizeof buf, "|%a|%a|%d|%d|%d", s.real(), s.imag(), m.is_conjugated() ? 1 : 0,
                  m.is_transposed() ? 1 : 0, static_cast<int>(m.get_structure()));
    return std::string(quantity) + "|" + params + "|" + typeid(m).name() + buf;
}
}  // namespace derived_cache

// compute() through the cache of m's storage. V must be copyable.
template <typename V, typename F>
V cached(const MatrixBase& m, const char* quantity, const std::string& params, F&& compute) {
    if (!derived_cache::enabled().load(std::memory_order_relaxed) || m.is_subview()) return compute();
    const auto& mapper = m.shared_mapper();  // a reference: a copy would raise use_count (shared)
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

}  // namespace pycauset
