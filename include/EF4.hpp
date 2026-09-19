/*
ELIAS-FANO BUCKET IMPLEMENTATION 4:
single flat array composed by S[]P[]F[]
where:
* s = bucket starts 
* p = last predecessor of the bucket
* f = suffixes

The initial header now contains also the last predecessor of each bucket.
It's retrieved in O(1) time and no cache misses during prefix fetching, 
completely avoinding the need of exponential search.
space: log_n * #buckets bits = m/logm * logn = o(mlog(n/m)). still succint.
*/

#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <span>
#include <vector>
#include <bit>
#include <optional>

#include "concept_type.hpp"
#include "EF_utils.h"

using u32 = uint32_t;
using u64 = uint64_t;

class EF4 final {
public:

    /// @brief Constructor for EF4
    /// @param sorted_vals an ascending sorted vector of unique integers in [0, universe)
    /// @param universe a positive integer representing the universe size
    EF4 (std::span<const u64> sorted_vals) : m(sorted_vals.size()) {
        checkInput(sorted_vals);
        n = sorted_vals.back() + 1;
        
        m_bitsForM = std::bit_width(m); 
        m_bitsForN = ceilLog2(n); 
        
        m_prefixBits = prefixBitsFor(m, m_bitsForN);
        m_suffixBits = (m_bitsForN - m_prefixBits);
        m_suffixMask = lowMask(m_suffixBits);
        
        m_numBuckets = 1ULL << m_prefixBits; 
        
        /* total length of the flat array is given by:
        1. bucket_starts S[]: (num_buckets - 1) * (bits_for_m ) + padding up to 64 bits for alignment
            which is std::ceil((num_buckets - 1) * (bits_for_m ) / 64) words
        2. last predecessors P[]: (num_buckets - 1) * (bits_for_n ) + padding up to 64 bits for alignment
            which is std::ceil((num_buckets - 1) * (bits_for_n ) / 64) words
        3. suffixes F[]: m * bits_for_suffix + padding up to 64 bits for alignment
            which is std::ceil(m * bits_for_suffix / 64) words
		One word of padding at the end is added in order to read 2 words without 
		risking segmentation fault, during binary search on suffixes.
        */
        m_predOffset = ((m_numBuckets - 1) * m_bitsForM + 63) / 64;
        m_suffixOffset = m_predOffset + ((m_numBuckets - 1) * m_bitsForN + 63) / 64;
        u64 total_words = (m_suffixOffset + ((u64)m * m_suffixBits + 63) / 64) + 1;
        m_data.resize(total_words);
        

        // population of S[] now has the interleaved last predecessor
        std::vector<u64> accumulator(m_numBuckets, 0);
        for (auto v : sorted_vals)
            accumulator[getPrefixBits(v)]++;
        u64 position = 0;
        for (size_t i = 1; i < m_numBuckets; i++) {
            position += accumulator[i - 1];
            setS(i, position);
            if (position > 0) setP(i, sorted_vals[position - 1]);
        }

        // population of F[] stays the same
        size_t k = 0;
        for (auto v : sorted_vals) {
            setF(k++, getSuffixBits(v));
        }
    }

    u64 size() const { return m; }
    u64 universe() const { return n; }
    size_t bytes() const { return m_data.size() * 8; }

    std::optional<u64> access(u64 k) const {
        if (k >= m) [[unlikely]] return std::nullopt;

        u64 lo = 0, hi = m_numBuckets;
        while (lo < hi) {
            u64 mid = lo + (hi - lo) / 2;
            if (getS(mid + 1) <= k)
                lo = mid + 1;
            else
                hi = mid;
        }
        return std::make_optional(reconstruct(lo, k));
    }

    std::optional<u64> predecessor(u64 x) const {
        if (m == 0) return std::nullopt;
        if (x >= n) x = n - 1;

        u64 prefix = getPrefixBits(x);
        u64 suffix = getSuffixBits(x);
        
        u64 bucket = prefix; 
        u64 lo_s = getS(bucket);
        u64 hi_s = getS(bucket + 1);

        // as before, check if it has elements
        if (lo_s < hi_s) {

            // binary search on suffixes of the bucket
            u64 pos = suffix_upper_bound(lo_s, hi_s, suffix);
            if (pos > lo_s) return reconstruct(bucket, pos - 1); // founnd
        }
        // else (both in case not found and empty bucket): check backwards

        if (lo_s == 0) return std::nullopt; // no previous bucket
        return std::make_optional(getP(bucket));
    }

    std::optional<u64> successor(u64 x) const {
        if (m == 0 || x >= n) return std::nullopt;

        u64 prefix = getPrefixBits(x);
        u64 suffix = getSuffixBits(x);
        
        u64 bucket = prefix; 
        u64 lo_s = getS(bucket);
        u64 hi_s = getS(bucket + 1);

        // check if it has elements
        if (lo_s < hi_s) {

            // binary search on suffixes of the bucket
            u64 pos = suffix_lower_bound(lo_s, hi_s, suffix);
            if (pos < hi_s) return reconstruct(bucket, pos); // found
        }
        // else (both in case not found and empty bucket): check forwards

        u64 k = hi_s;
        if (k == m) return std::nullopt; // no next bucket

        // forward Exponential search 
        u64 lo = bucket + 1;
        u64 hi = m_numBuckets;
        u64 step = 1;
        
        while (lo + step <= m_numBuckets && getS(lo + step) <= k) {
            lo += step;
            step <<= 1;
        }

        // find correct range
        hi = (lo + step <= m_numBuckets) ? lo + step : m_numBuckets;

        // binary search on the range
        while (lo < hi - 1) {
            u64 mid = lo + (hi - lo) / 2;
            if (getS(mid) <= k)
                lo = mid;
            else
                hi = mid;
        }

        return std::make_optional(reconstruct(lo, k));
    }

    bool contains(u64 x) const {
        if (m == 0 || x >= n) return false;

        u64 prefix = getPrefixBits(x);
        u64 suffix = getSuffixBits(x);
        
        u64 lo_s = getS(prefix);
        u64 hi_s = getS(prefix + 1);

        if (lo_s == hi_s) return false;

        u64 pos = suffix_lower_bound(lo_s, hi_s, suffix);

        return (pos < hi_s && getF(pos) == suffix);
    }

private:
    u64 m, n, m_numBuckets;
    u32 m_prefixBits, m_predOffset, m_suffixBits, m_bitsForM, m_bitsForN, m_suffixOffset;
    u64 m_suffixMask;
    std::vector<u64> m_data;

    // Offset helpers: separati
    u64 s_offset(u64 i) const { return (i - 1) * m_bitsForM; }
    u64 p_offset(u64 i) const { return m_predOffset * 64 + (i - 1) * m_bitsForN; }
    u64 f_offset(u64 i) const { return m_suffixOffset * 64 + i * m_suffixBits; }

    u64 reconstruct(u64 bucket, u64 k) const { return (bucket << m_suffixBits) | getF(k); }
    u64 getPrefixBits(u64 v) const { return v >> m_suffixBits; }
    u64 getSuffixBits(u64 v) const { return v & m_suffixMask; }

    u64 getS(u64 i) const {
        if (i == 0) return 0;
        if (i == m_numBuckets) return m;
        return readBits(m_data, s_offset(i), m_bitsForM); 
    };
    u64 getP(u64 i) const {
        if (i==0) return 0;
        return readBits(m_data, p_offset(i), m_bitsForN); 
    }
    void setS(u64 i, u64 val) { writeBits(m_data, s_offset(i), m_bitsForM, val); }
    void setP(u64 i, u64 val) { writeBits(m_data, p_offset(i), m_bitsForN, val); }
    u64 getF(u64 i) const { return readBits(m_data, f_offset(i), m_suffixBits); }
    void setF(u64 i, u64 val) { writeBits(m_data, f_offset(i), m_suffixBits, val); }

    u64 suffix_upper_bound(u64 lo, u64 hi, u64 suf) const {
        while (lo < hi) {
            u64 mid = lo + (hi - lo) / 2;
            if (getF(mid) <= suf) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    u64 suffix_lower_bound(u64 lo, u64 hi, u64 suf) const {
        while (lo < hi) {
            u64 mid = lo + (hi - lo) / 2;
            if (getF(mid) < suf) lo = mid + 1; else hi = mid;
        }
        return lo;
    }
};

static_assert(EF<EF4>, "EF4 does not satisfy EF concept");
