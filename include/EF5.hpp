/*
ELIAS-FANO FIXED-SIZE BUCKET IMPLEMENTATION
Layout: SB[] | BUCKET_DATA[] | OVERFLOW (flat array with packed suffixes)

Each bucket at arithmetic offset:
	[size: log(m) bits][max_prev: log(n) bits][slot[0]..slot[C-1]: b bits each][offset in the ovf array: log(ovf) bits]

	size     = total # elements with this prefix (including overflow)
	max_prev = full value of max element among all preceding non-empty buckets
	C        = lambda + c * sqrt(lambda)   (bucket capacity) where lambda = m/numBuckets
	b        = ceil(log n) - floor(log(m/log m))   (suffix bits)

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
#include <tuple>

#include "concept_type.hpp"
#include "EF_utils.h"

using u32 = uint32_t;
using u64 = uint64_t;

/// @brief Elias-Fano variant with fixed-size buckets and a flat overflow array (layout described above)
/// @tparam c_param extra slots per bucket, in standard deviations: capacity = lambda + c_param * sqrt(lambda), with lambda = m / numBuckets
template <float c_param = 0.0f>
class EF5 {
public:

	/// @brief Constructor for EF5
	/// @param sorted_vals an ascending sorted vector of unique integers in [0, universe)
	EF5(std::span<const u64> sorted_vals)
		: m(sorted_vals.size()) {

		checkInput(sorted_vals);
		n = sorted_vals.back() + 1;

		m_bitsForM = std::bit_width(m); // number of bits to represent size
		m_bitsForN = ceilLog2(n); // number of bits to represent universe
		
		m_prefixBits = prefixBitsFor(m, m_bitsForN);
		m_suffixBits = (m_bitsForN - m_prefixBits);
		m_suffixMask = lowMask(m_suffixBits);

		// Number of buckets.
		// 
		// I cant just use 1 << m_prefix_bits.
		// for example, with n=700 and prefixBits = 2:
		// bitsForN =10.
		// with 1<<prefixBits, I'd have 4 buckets of 256 elements each, but the last bucket would be empty (768 - 1023).
		//
		// instead, i check what is the bucket the last element belongs to. If it's in the bucket number 2, it'll be the third (thus +1) bucket.
		// 700 has '10' as prefix -> bucket number 2. so I need 3 buckets (0 - 255)(256 - 511)(512 - 767).
		m_numBuckets = (sorted_vals.back() >> m_suffixBits) + 1;

		u64 occupancyPerBucket = std::round(m / double(m_numBuckets));
		u64 nAdditionalSlots = c_param * std::sqrt(occupancyPerBucket);
		m_slotsPerBucket = occupancyPerBucket + nAdditionalSlots;

		m_sampleRate = std::max<u64>(1, u64( std::sqrt( double(m_numBuckets) )));
		m_numSamples = (m_numBuckets + m_sampleRate - 1) / m_sampleRate;

		// count per bucket
		std::vector<u64> acc(m_numBuckets, 0);
		for (auto v : sorted_vals)
			acc[getPrefixBits(v)]++;

		// count the number of overflow elements
		u64 ovfCount = 0;
		for (auto b : acc)
			if (b > m_slotsPerBucket) ovfCount += b - m_slotsPerBucket;

		m_bitsForOvf = std::bit_width(ovfCount);
		m_ovfElements = ovfCount;
	
		/*
		total length of the flat array is given by:
		1. bucket sampling SB[]: m_numSamples * m_bitsForM.
		2. suffixes F[]: m_numBuckets of
			i. size (m_bitsForM bits)
			ii. max_prev (m_bitsForN bits)
			iii. m_slotsPerBucket slots (m_suffixBits bits each)
			iiii. pointer to overflow (m_bitsForOvf bits)
		3. overflow elements (m_suffixBits bits each)
		*/
		u64 wordsForSB = (m_numSamples * m_bitsForM + 63) / 64;
		m_bitsPerBucket = m_bitsForM + m_bitsForN + m_slotsPerBucket * m_suffixBits + m_bitsForOvf;
		u64 wordsForBuckets = (m_numBuckets * m_bitsPerBucket + 63) / 64;
		u64 wordsForOvf = (ovfCount * m_suffixBits + 63) / 64;

		m_bucketsOffset = wordsForSB;
		m_ovfOffset = (m_bucketsOffset + wordsForBuckets);

		u64 totalWords = m_ovfOffset + wordsForOvf +1; // +1 for padding to avoid illegal reads 
		m_data.resize(totalWords);

		m_isFirstElementZero = (sorted_vals[0] == 0);

		u64 lastPrev = 0; // last valid predecessor for each bucket
		u64 ovfOffsetPartial = 0;
		u64 p = 0;		// absolute position in sorted_vals
		u64 SBaccumulator = 0;

		for (u64 b = 0; b < m_numBuckets; b++) {

			// populate SB[]. SB[0] is never written: it stays 0 from the zero-initialized m_data
			if (b % m_sampleRate == 0 && b != 0) {
				setSB(b / m_sampleRate, SBaccumulator);
			}

			u64 bucketSize = acc[b]; SBaccumulator += bucketSize;
			u64 inBucket = 0;
			u64 nOvfElements = 0;
			
			if (bucketSize > m_slotsPerBucket) {
				inBucket = m_slotsPerBucket;
				nOvfElements = bucketSize - inBucket;
			}
			else {
				inBucket = bucketSize;
			}
			
			// bucket header
			setBucketSize(b, bucketSize);
			setBucketMaxPrev(b, lastPrev);
			setOvfPointer(b, ovfOffsetPartial); 
			

			// fulfill bucket
			for (u64 i = 0; i < inBucket; i++) {
				setBucketSlotSuffix(b, i, getSuffixBits(sorted_vals[p++]));
			}

			// consume remaining overflow elements
			for (u64 i = 0; i < nOvfElements; i++) {
				setOvfSlotSuffix(ovfOffsetPartial, i, getSuffixBits(sorted_vals[p++]));
			}

			if (bucketSize > 0) lastPrev = sorted_vals[p - 1];
			ovfOffsetPartial += nOvfElements;
		}
		
		assert(p == m && ovfOffsetPartial == ovfCount); // deleted in release
	}

	u64 size() const { return m; }

	u64 universe() const { return n; }

	size_t bytes() const { return m_data.size() * 8; }

	std::optional<u64> access(u64 k) const {
		if (k >= m) [[unlikely]]
			return std::nullopt;

		// binary search on SB[]: last sample with SB <= k (SB[0] = 0 always qualifies)
		u64 lo = 1 /*zero is known*/, hi = m_numSamples;
		u64 cumulativeSum = 0; // SB[0]
		while (lo < hi) {
			u64 mid = lo + (hi - lo) / 2;
			u64 elementsBefore = getSB(mid);
			if (elementsBefore <= k) {
				cumulativeSum = elementsBefore; // lo changes only here, so at the end this is SB[lo - 1]
				lo = mid + 1;
			}
			else
				hi = mid;
		}

		u64 sampleIndex = lo - 1; // SB[lo] > k, thus -1
		u64 startBucket = sampleIndex * m_sampleRate;

		u64 prefix = 0;
		u64 suffix = 0;

		for (u64 b = startBucket; b < m_numBuckets; b++) {
			u64 bucketSize = getBucketSize(b);
			if (cumulativeSum + bucketSize > k) {
				u64 localK = k - cumulativeSum; // index inside the bucket
				prefix = b; // the bucket index is the prefix

				if (localK < m_slotsPerBucket)
					suffix = getBucketSlotSuffix(b, localK);
				else
					suffix = getOvfSlotSuffix(getOvfPointer(b), localK - m_slotsPerBucket);
				return reconstructNumber(prefix, suffix);
			}
			cumulativeSum += bucketSize;
		}
		return std::nullopt;
	}

	std::optional<u64> predecessor(u64 x) const {
		if (m == 0) [[unlikely]]
			return std::nullopt;
		if (x >= n)
			x = n - 1;

		u64 queryPrefix = getPrefixBits(x); // which is also the bucket index
		u64 querySuffix = getSuffixBits(x); 
		
		u64 bucketSize = getBucketSize(queryPrefix);

		// cross bucket predecessor
		if (bucketSize == 0 || getBucketSlotSuffix(queryPrefix, 0) > querySuffix) {
			u64 maxPrev = getBucketMaxPrev(queryPrefix);
			if (maxPrev != 0)
				return std::optional<u64>{maxPrev};
			if (m_isFirstElementZero)
				return std::optional<u64>{0};
			else return std::nullopt;
		}

		// in-bucket predecessor
		if (bucketSize > m_slotsPerBucket) {
			u64 lastSlotSuffix = getBucketSlotSuffix(queryPrefix, m_slotsPerBucket - 1);
			if (lastSlotSuffix < querySuffix) {
				// binary search in the overflow section
				int64_t candidate = findPredecessorIndex(true, queryPrefix, bucketSize - m_slotsPerBucket, querySuffix);
				return (candidate == -1) ?
					std::optional<u64>{reconstructNumber(queryPrefix, lastSlotSuffix)} :
					std::optional<u64>{reconstructNumber(queryPrefix, getOvfSlotSuffix(getOvfPointer(queryPrefix), candidate))};
			}
			// binary search in the bucket section, in [0, m_slotsPerBucket)
			int64_t candidate = findPredecessorIndex(false, queryPrefix, m_slotsPerBucket, querySuffix);
			assert(candidate != -1);
			return std::optional<u64>{reconstructNumber(queryPrefix, getBucketSlotSuffix(queryPrefix, candidate))};
			
		}
		// binary search in the bucket section, in [0, bucketSize)
		int64_t candidate = findPredecessorIndex(false, queryPrefix, bucketSize, querySuffix);
		assert(candidate != -1);
		return std::optional<u64>{reconstructNumber(queryPrefix, getBucketSlotSuffix(queryPrefix, candidate))};
	}

	std::optional<u64> successor(u64 x) const {
		if (x >= n || m == 0)
			return std::nullopt;

		u64 queryPrefix = getPrefixBits(x); // which is also the bucket index
		u64 querySuffix = getSuffixBits(x);
		u64 bucketSize = getBucketSize(queryPrefix);

		if (bucketSize == 0)
			return findSuccessorCrossBucket(queryPrefix);

		u64 inBucket = std::min(bucketSize, m_slotsPerBucket);
		u64 lastSlotSuffix = getBucketSlotSuffix(queryPrefix, inBucket - 1);

		if (lastSlotSuffix >= querySuffix) {
			int64_t candidate = findSuccessorIndex(false, queryPrefix, inBucket, querySuffix);
			assert(candidate != -1); // the last slot is >= querySuffix
			return reconstructNumber(queryPrefix, getBucketSlotSuffix(queryPrefix, candidate));
		}
		assert(lastSlotSuffix < querySuffix);

		// x is past the last slot: the successor is in the overflow run, if any, or in a following bucket
		if (bucketSize > m_slotsPerBucket) {
			int64_t candidate = findSuccessorIndex(true, queryPrefix, bucketSize - m_slotsPerBucket, querySuffix);
			if (candidate != -1)
				return reconstructNumber(queryPrefix, getOvfSlotSuffix(getOvfPointer(queryPrefix), candidate));
		}
		return findSuccessorCrossBucket(queryPrefix);
	}

	bool contains(u64 x) const {
		if (m == 0 || x >= n)
			return false;

		u64 queryPrefix = getPrefixBits(x);
		u64 querySuffix = getSuffixBits(x);

		u64 bucketSize = getBucketSize(queryPrefix);
		if (bucketSize == 0)
			return false;

		u64 inBucket = std::min(bucketSize, m_slotsPerBucket);
		u64 lastSlotSuffix = getBucketSlotSuffix(queryPrefix, inBucket - 1);
		int64_t candidate = -1;
		if (lastSlotSuffix == querySuffix) {
			return true;
		}
		else if (lastSlotSuffix < querySuffix) {
			if (bucketSize <= m_slotsPerBucket)
				return false;
			candidate = findSuccessorIndex(true, queryPrefix, bucketSize - inBucket, querySuffix);
			return candidate == -1 ? false : (getOvfSlotSuffix(getOvfPointer(queryPrefix), candidate) == querySuffix);
		}
		else {
			candidate = findSuccessorIndex(false, queryPrefix, inBucket - 1, querySuffix);
			return candidate == -1 ? false : (getBucketSlotSuffix(queryPrefix, candidate) == querySuffix);
		}
	}

	std::tuple<u64, u64> ovfInfo() const { return { m - m_ovfElements, m_ovfElements}; }

private:
	u64 m, n, m_numBuckets, m_ovfElements; //ovfElements is just for statistics, useless otherwise
	u32 m_prefixBits, m_suffixBits, m_bitsForM, m_bitsForN, m_bitsForOvf;
	u64 m_slotsPerBucket, m_bitsPerBucket, m_sampleRate, m_numSamples;
	u64 m_bucketsOffset, m_ovfOffset; // word offsets into m_data
	u64 m_suffixMask;
	bool m_isFirstElementZero;

	std::vector<u64> m_data; // bit-packed SB[] + F[] + OVF[] + padding

	/// @brief prefix of a value (which is the index of the bucket it belongs to)
	/// @param v a value in [0, universe)
	u64 getPrefixBits(u64 v) const {
		assert(v < n);
		return v >> m_suffixBits;
	}

	/// @brief suffix of a value (which is the part stored)
	/// @param v a value in [0, universe)
	u64 getSuffixBits(u64 v) const {
		assert(v < n);
		return v & m_suffixMask;
	}

	/// @brief write a sample of SB[]
	/// @param index sample index, in [1, m_numSamples). SB[0] is stored but never written:
	///        it reads 0 because m_data is zero-initialized
	/// @param val number of elements in the buckets before bucket index * m_sampleRate
	void setSB(u64 index, u64 val) {
		assert(index >= 1 && index < m_numSamples);
		u64 position = index * m_bitsForM;
		writeBits(m_data, position, m_bitsForM, val);
	}

	/// @brief write the size field of a bucket
	/// @param bucket bucket index
	/// @param val number of elements with this prefix, overflow included
	void setBucketSize(u64 bucket, u64 val) {
		assert(bucket < m_numBuckets);
		u64 position = m_bucketsOffset * 64 + bucket * m_bitsPerBucket;
		writeBits(m_data, position, m_bitsForM, val);
	}

	/// @brief write the max_prev field of a bucket
	/// @param bucket bucket index
	/// @param val largest element in the buckets before this one; 0 if there is none
	void setBucketMaxPrev(u64 bucket, u64 val) {
		assert(bucket < m_numBuckets);
		u64 position = m_bucketsOffset * 64 + bucket * m_bitsPerBucket + m_bitsForM /*size*/;
		writeBits(m_data, position, m_bitsForN, val);
	}

	/// @brief write the overflow pointer of a bucket
	/// @param bucket bucket index
	/// @param val position in the overflow array where the run of this bucket starts
	void setOvfPointer(u64 bucket, u64 val) {
		assert(bucket < m_numBuckets);
		u64 position = m_bucketsOffset * 64 + bucket * m_bitsPerBucket + m_bitsForM + m_bitsForN + m_slotsPerBucket * m_suffixBits;
		writeBits(m_data, position, m_bitsForOvf, val);
	}

	/// @brief write a suffix in a slot of a bucket
	/// @param bucket bucket index
	/// @param slotIndex slot index
	/// @param val the suffix to store
	void setBucketSlotSuffix(u64 bucket, u64 slotIndex, u64 val) {
		assert(bucket < m_numBuckets);
		assert(slotIndex < m_slotsPerBucket);
		u64 position = m_bucketsOffset * 64 + 
			bucket * m_bitsPerBucket + 
			m_bitsForM + //size
			m_bitsForN + //max_prev
			slotIndex * m_suffixBits;
		writeBits(m_data, position, m_suffixBits, val);
	}

	/// @brief write a suffix in the overflow array
	/// @param base position where the run of the bucket starts
	/// @param slotIndex position inside the run
	/// @param val the suffix to store, written at position base + slotIndex
	void setOvfSlotSuffix(u64 base, u64 slotIndex, u64 val) {
		u64 position = m_ovfOffset * 64 + (base + slotIndex) * m_suffixBits;
		writeBits(m_data, position, m_suffixBits, val);
	}

	/// @brief read the size field of a bucket
	/// @param bucket bucket index
	u64 getBucketSize(u64 bucket) const {
		assert(bucket < m_numBuckets);
		u64 position = m_bucketsOffset * 64 + bucket * m_bitsPerBucket;
		return readBits(m_data, position, m_bitsForM);
	}

	/// @brief read the suffix stored in a slot of a bucket
	/// @param bucket bucket index
	/// @param slotIndex slot index
	u64 getBucketSlotSuffix(u64 bucket, u64 slotIndex) const {
		assert(bucket < m_numBuckets);
		assert(slotIndex < m_slotsPerBucket);
		u64 position = m_bucketsOffset * 64 +
			bucket * m_bitsPerBucket +
			m_bitsForM + //size
			m_bitsForN + //max_prev
			slotIndex * m_suffixBits;
		return readBits(m_data, position, m_suffixBits);
	}

	/// @brief read a suffix from the overflow array
	/// @param base position where the run of the bucket starts
	/// @param slotIndex position inside the run
	u64 getOvfSlotSuffix(u64 base, u64 slotIndex) const {
		u64 position = m_ovfOffset * 64 + (base + slotIndex) * m_suffixBits;
		return readBits(m_data, position, m_suffixBits);
	}

	/// @brief read the overflow pointer of a bucket
	/// @param bucket bucket index
	u64 getOvfPointer(u64 bucket) const {
		assert(bucket < m_numBuckets);
		u64 position = m_bucketsOffset * 64 + bucket * m_bitsPerBucket + m_bitsForM + m_bitsForN + m_slotsPerBucket * m_suffixBits;
		return readBits(m_data, position, m_bitsForOvf);
	}

	/// @brief binary search for the first suffix >= querySuffix among the elements of a bucket
	/// @param inOvf false to search the slots of the bucket, true to search its run in the overflow array
	/// @param bucket bucket index
	/// @param qty number of elements to search, starting from the first slot of the bucket / overflow run
	/// @param querySuffix the suffix to search for
	/// @return the index (relative to the first slot / to the start of the run) or -1 if all suffixes are smaller
	int64_t findSuccessorIndex(bool inOvf, u64 bucket, u64 qty, u64 querySuffix) const {
		
		u64 ovfBase = inOvf ? 
			getOvfPointer(bucket) : 
			0; // read once, not at every probe
		
		auto suffixAt = [&](u64 i) {
			return inOvf ? 
				getOvfSlotSuffix(ovfBase, i) : 
				getBucketSlotSuffix(bucket, i);
		};

		// lower bound in [0, qty): first index with suffix >= querySuffix
		u64 lo = 0, hi = qty;
		while (lo < hi) {
			u64 mid = lo + (hi - lo) / 2;
			if (suffixAt(mid) < querySuffix)
				lo = mid + 1;
			else
				hi = mid;
		}
		return (lo == qty) ? -1 : int64_t(lo);
	}

	/// @brief binary search for the last suffix <= querySuffix among the elements of a bucket
	/// @param inOvf false to search the slots of the bucket, true to search its run in the overflow array
	/// @param bucket bucket index
	/// @param qty number of elements to search, starting from the first slot of the bucket / overflow run
	/// @param querySuffix the suffix to search for
	/// @return the index (relative to the first slot / to the start of the run) or -1 if all suffixes are greater
	int64_t findPredecessorIndex(bool inOvf, u64 bucket, u64 qty, u64 querySuffix) const {

		u64 ovfBase = inOvf ?
			getOvfPointer(bucket) :
			0;

		auto suffixAt = [&](u64 i) {
			return inOvf ?
				getOvfSlotSuffix(ovfBase, i) :
				getBucketSlotSuffix(bucket, i);
			};

		// upper bound in [0, qty): first index with suffix > querySuffix
		u64 lo = 0, hi = qty;
		while (lo < hi) {
			u64 mid = lo + (hi - lo) / 2;
			if (suffixAt(mid) <= querySuffix)
				lo = mid + 1;
			else
				hi = mid;
		}
		return int64_t(lo) - 1; // the one before is the last suffix <= querySuffix; -1 if lo == 0
	}

	/// @brief linear search for the first non-empty bucket and retrieve the successor
	std::optional<u64> findSuccessorCrossBucket(u64 bucket) const {
		assert(bucket < m_numBuckets);
		for (u64 b = bucket + 1; b < m_numBuckets; b++) {
			if (getBucketSize(b) > 0)
				return reconstructNumber(b, getBucketSlotSuffix(b, 0)); // slot 0 holds the smallest element
		}
		return std::nullopt;
	}

	/// @brief read the max_prev field of a bucket
	/// @param bucket bucket index
	u64 getBucketMaxPrev(u64 bucket) const {
		assert(bucket < m_numBuckets);
		u64 position = m_bucketsOffset * 64 + bucket * m_bitsPerBucket + m_bitsForM /*size*/;
		return readBits(m_data, position, m_bitsForN);
	}

	/// @brief read the value in SB[index]
	u64 getSB(u64 index) const {
		assert(index >= 1 && index < m_numSamples);
		u64 position = index * m_bitsForM;
		return readBits(m_data, position, m_bitsForM);
	}

	/// @brief recontructs the number given its prefix and suffix
	u64 reconstructNumber(u64 prefix, u64 suffix) const {
		return u64{ prefix << m_suffixBits | suffix };
	}
	
};

static_assert(EF<EF5<>>, "EF5 does not satisfy EF concept");