#include "pycauset/math/CausalMatrix.hpp"

#include <bit>
#include <vector>

#include "pycauset/core/ParallelUtils.hpp"

namespace pycauset {

// Link matrix (transitive reduction) of a TRANSITIVELY CLOSED causal matrix C, on the bits
// (F-053; LAZY.md L7.4). Row i: walk future(i) in increasing order; j is a link iff it is not
// yet covered by the futures of the links found so far (transitivity makes that equivalent to
// "no k with i < k < j, C_ik and C_kj"). Cost per row ~ (#links) * n/64 word ORs instead of
// an n^3 product; no n^2 integer counts. Rows start on 64-bit boundaries, so rows are
// independent and processed in parallel.
std::unique_ptr<MatrixBase> transitive_reduction(const TriangularMatrix<bool>& c) {
    const uint64_t n = c.rows();
    auto out = std::make_unique<TriangularMatrix<bool>>(n, "");
    if (n < 2) return out;
    const auto* base = static_cast<const char*>(c.require_mapper()->get_data());
    auto* obase = static_cast<char*>(out->require_mapper()->get_data());
    const uint64_t full_words = (n + 63) / 64 + 1;
    pycauset::ParallelFor(0, n - 1, [&](size_t i) {
        const auto* row = reinterpret_cast<const uint64_t*>(base + c.get_row_offset(i));
        auto* orow = reinterpret_cast<uint64_t*>(obase + out->get_row_offset(i));
        const uint64_t nbits = n - 1 - i;
        std::vector<uint64_t> covered(full_words, 0);  // absolute column bitset
        for (uint64_t q = 0; q * 64 < nbits; ++q) {
            uint64_t bits = row[q];
            const uint64_t rem = nbits - q * 64;
            if (rem < 64) bits &= (uint64_t{1} << rem) - 1;
            while (bits) {
                const uint64_t t = static_cast<uint64_t>(std::countr_zero(bits));
                bits &= bits - 1;
                const uint64_t j = i + 1 + q * 64 + t;
                if ((covered[j >> 6] >> (j & 63)) & 1ULL) continue;
                orow[q] |= uint64_t{1} << t;  // (i, j) is a link
                // covered |= future(j): row j bit b is column j + 1 + b.
                const uint64_t jb = n - 1 - j;
                if (jb == 0) continue;
                const auto* rj = reinterpret_cast<const uint64_t*>(base + c.get_row_offset(j));
                const uint64_t s = j + 1, sw = s >> 6, sh = s & 63;
                for (uint64_t p = 0; p * 64 < jb; ++p) {
                    uint64_t w = rj[p];
                    const uint64_t r2 = jb - p * 64;
                    if (r2 < 64) w &= (uint64_t{1} << r2) - 1;
                    covered[sw + p] |= w << sh;
                    if (sh != 0) covered[sw + p + 1] |= w >> (64 - sh);
                }
            }
        }
    });
    return out;
}

}  // namespace pycauset
