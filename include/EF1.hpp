/*
ELIAS-FANO BUCKET IMPLEMENTATION 1:
single flat array composed by S[]B[]F[]
where:
* s = bucket starts,
* b = bitflags
* f = suffixes
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

class EF1 {
public:

    /// @brief Constructor for EF1
    /// @param sorted_vals an ascending sorted vector of unique integers in [0, universe)
    EF1(std::span<const u64> sorted_vals)
        : m(sorted_vals.size()) {

        checkInput(sorted_vals);
        n = sorted_vals.back() + 1;
        
        m_bitsForM = std::bit_width(m); // number of bits to represent size
        m_bitsForN = ceilLog2(n); // number of bits to represent universe
        
        // split prefix/suffix
        m_prefixBits = prefixBitsFor(m, m_bitsForN);
        m_suffixBits = (m_bitsForN - m_prefixBits);
        m_suffixMask = lowMask(m_suffixBits);
        
        // number of buckets = 2^(prefix bits)
        m_numBuckets = 1ULL << m_prefixBits; 
        
        /* total length of the flat array is given by:
        1. bucket_starts S[]: (num_buckets - 1) * bits_for_m + padding up to 64 bits for alignment
            which is std::ceil((num_buckets - 1) * bits_for_m / 64) words
        2. bit-flags B[] for occupied buckets: num_buckets bits, 
            which is std::ceil(num_buckets / 64) words
        3. suffixes F[]: m * bits_for_suffix + padding up to 64 bits for alignment
            which is std::ceil(m * bits_for_suffix / 64) words
		One word of padding at the end is added in order to read 2 words without 
		risking segmentation fault, during binary search on suffixes.
        */
        m_bitflagOffset = ((m_numBuckets - 1) * m_bitsForM + 63) / 64;
        m_suffixOffset  = m_bitflagOffset + (m_numBuckets + 63) / 64;
        u64 total_words = (m_suffixOffset + ((u64)m * m_suffixBits + 63) / 64) + 1;
        m_data.resize(total_words);

        /* Populate S[]:
        1. count elements per bucket on a temporary array acc
        2. store the offsets of each bucket start in S[]
            note that S[0] = 0 is implicit and not stored,
            so S[i] is actually stored at index i-1
        */
        std::vector<u64> acc(m_numBuckets, 0);
        for (auto v : sorted_vals)
            acc[getPrefixBits(v)]++;
        u64 position = 0;
        for (size_t i =1; i < m_numBuckets; i++) {
            position += acc[i - 1];
            setS(i, position);
        }

        /*Populate B[]: empty = 0
        straightforward from acc
        */
        for (u64 i = 0; i < m_numBuckets; i++)
            if (acc[i]) setOccupiedBit(i);
        
        /*Populate F[]: suffixes 
        for each element in sorted_vals, compute its suffix and store it compactly
        */
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

        // Binary search on bucket starts read from S[]
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
        if (x >= n) x = n - 1; // clamp to universe max

		u64 prefix = getPrefixBits(x);
		u64 suffix = getSuffixBits(x);
		u64 bucket = prefix; 

		if (isOccupied(bucket)) {
			u64 lo = getS(bucket), hi = getS(bucket + 1);
			u64 pos = suffix_upper_bound(lo, hi, suffix);
			if (pos > lo) return reconstruct(bucket, pos - 1);
			// else: just continue with the flow
		}

        // check previous occupied bucket via bitvector
        auto prev = prevOccupied(bucket);
        if (!prev) return std::nullopt;
        return std::make_optional(reconstruct(*prev, getS(*prev + 1) - 1));
    }

    std::optional<u64> successor(u64 x) const {
        if (m == 0 || x >= n) return std::nullopt;

        u64 prefix = getPrefixBits(x);
		u64 suffix = getSuffixBits(x);
		u64 bucket = prefix;

		if (isOccupied(bucket)) {
			u64 lo = getS(bucket), hi = getS(bucket + 1);
			u64 pos = suffix_lower_bound(lo, hi, suffix);
			if (pos < hi) return reconstruct(bucket, pos);
		}

        // Skip forwards via occupied bitvector
        auto next = nextOccupied(bucket + 1);
        if (!next) return std::nullopt;
        u64 nb = *next;
        return std::make_optional(reconstruct(nb, getS(nb)));
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
    u32 m_prefixBits, m_suffixBits, m_bitsForM, m_bitsForN, m_suffixOffset, m_bitflagOffset;
    u64 m_suffixMask;
    std::vector<u64> m_data; // bit-packed S[] + bitflag + suffixes
    
    // ------------------------------------------------------------
    // data layout: helpers to get offsets of the various components 
    // ------------------------------------------------------------
    
    // offset (in bits) of S[i]: bucket_start
    u64 s_offset(u64 i) const { return (i - 1) * m_bitsForM; }

    // offset (in bits) of F[k]: suffixes
    u64 f_offset(u64 i) const { return m_suffixOffset * 64 + i * m_suffixBits; }

    // ------------------------------------------------------------
    // split and reconstruct a number into/from prefix & suffix 
    // ------------------------------------------------------------

    // reconstruct the original value from bucket and suffix index
    u64 reconstruct(u64 bucket, u64 k) const {
        return (bucket << m_suffixBits) | getF(k);
    }

    // get prefix bits of a value v in the set
    u64 getPrefixBits(u64 v) const { return v >> m_suffixBits; }

    // get suffix bits of a value v in the set
    u64 getSuffixBits(u64 v) const { return v & m_suffixMask; }

    // ------------------------------------------------------------
    // read and write components from the data layout on-the-fly 
    // ------------------------------------------------------------

    // read S[i]. returns the starting position of the i-th bucket on the suffix array F[]
    u64 getS(u64 i) const {
        if (i == 0) return 0;
        if (i == m_numBuckets) return m;
        return readBits(m_data, s_offset(i), m_bitsForM); 
	};

    // write S[i]. store the starting position of the i-th bucket on the suffix array F[]
    void setS(u64 i, u64 val) { writeBits(m_data, s_offset(i), m_bitsForM, val); }

    // read F[k]: suffix of k-th element
    u64 getF(u64 i) const { return readBits(m_data, f_offset(i), m_suffixBits); }

    // write F[k]: suffix of k-th element
    void setF(u64 i, u64 val) { writeBits(m_data, f_offset(i), m_suffixBits, val); }

    // set B[b] = 1: mark bucket b as occupied
    void setOccupiedBit(u64 b) { m_data[m_bitflagOffset + b / 64] |= (u64(1) << (b % 64)); }

    // get B[b]: check if bucket b is occupied
    u64 getOccupiedWord(u64 w) const { return m_data[m_bitflagOffset + w]; }


    // ------------------------------------------------------------
    // binary search on suffixes of a bucket: [lo, hi) is the range
    // ------------------------------------------------------------

    // binary search for the largest index in [lo, hi) such that getF(index) <= suf
    u64 suffix_upper_bound(u64 lo, u64 hi, u64 suf) const {
        while (lo < hi) {
            u64 mid = lo + (hi - lo) / 2;
            if (getF(mid) <= suf) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    // binary search for the smallest index in [lo, hi) such that getF(index) >= suf
    u64 suffix_lower_bound(u64 lo, u64 hi, u64 suf) const {
        while (lo < hi) {
            u64 mid = lo + (hi - lo) / 2;
            if (getF(mid) < suf) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    // -------------------------------------------------------------
    // function concerning B[] bitvector of occupied buckets
    // -------------------------------------------------------------

    bool isOccupied(u64 b) const {
        return (getOccupiedWord(b / 64) >> (b % 64)) & 1;
    }

    // Next occupied bucket >= b, or nullopt
    std::optional<u64> nextOccupied(u64 b) const {
        if (b >= m_numBuckets) return std::nullopt;
        u64 w = b / 64;
        u32 bit = b % 64;
        u64 masked = getOccupiedWord(w) & (~u64(0) << bit);
        if (masked) {
            u64 r = w * 64 + std::countr_zero(masked);
            return r < m_numBuckets ? std::optional(r) : std::nullopt;
        }

        u64 nwords = (m_numBuckets + 63) / 64;
        for (++w; w < nwords; w++) {
            if (getOccupiedWord(w)) {
                u64 r = w * 64 + std::countr_zero(getOccupiedWord(w));
                return r < m_numBuckets ? std::optional(r) : std::nullopt;
            }
        }
        return std::nullopt;
    }

    // Previous occupied bucket < b (strict), or nullopt
    std::optional<u64> prevOccupied(u64 b) const {
        if (b == 0) return std::nullopt;
        --b; 
        u64 w = b / 64;
        u32 bit = b % 64;
        u64 mask = (bit == 63) ? ~u64(0) : (u64(2) << bit) - 1;
        u64 masked = getOccupiedWord(w) & mask;
        if (masked) return w * 64 + 63 - std::countl_zero(masked);
        for (int64_t ww = static_cast<int64_t>(w) - 1; ww >= 0; ww--) {
            u64 word = getOccupiedWord(ww);
            if (word) {
                return static_cast<u64>(ww) * 64 + 63 - std::countl_zero(word);
            }
        }
        return std::nullopt;
    }

};

static_assert(EF<EF1>, "EF1 does not satisfy EF concept");
