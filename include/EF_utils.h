#pragma once

#include <cstdint>
#include <bit>
#include <span>
#include <cassert>
#include <algorithm>
#include <stdexcept>


/// @brief floor of the base-2 logarithm, i.e. the index of the most significant set bit
/// @param x the value to take the logarithm of
/// @return floor(log2(x)), e.g. floorLog2(10) = 3; 0 if x <= 1 (log2(0) is undefined, 0 by convention)
inline uint32_t floorLog2(uint64_t x) {
	return x <= 1 ? 
		0 : 
		63 - std::countl_zero(x);
}

/// @brief ceiling of the base-2 logarithm, i.e. the number of bits needed to represent every integer in [0, x)
/// @param x the value to take the logarithm of (typically a universe or a size)
/// @return ceil(log2(x)), e.g. ceilLog2(10) = 4 and ceilLog2(16) = 4; 0 if x <= 1
inline uint32_t ceilLog2(uint64_t x) {
	return x <= 1 ?
		0 :
		64 - std::countl_zero(x - 1);
}

/// @brief mask with the nBits least significant bits set
/// @param nBits number of bits, in [0, 64]
/// @return (1 << nBits) - 1, e.g. lowMask(4) = 0b1111; all ones if nBits == 64 (shifting by 64 is UB)
inline uint64_t lowMask(uint32_t nBits) {
	return nBits == 64 ?
		~uint64_t(0) :
		(uint64_t(1) << nBits) - 1;
}

/// @brief number of prefix bits used to split the values: about m / log2(m) buckets, rounded down to a power of 2.
///        Because of the rounding, a bucket holds between L and 2L elements on average, with L = floor(log2(m)).
///        Never less than bitsForN - 63, so the suffix has at most 63 bits and shifting by it is always defined
///        (it matters only for m = 1 and a value >= 2^63, where the formula alone gives 0 prefix bits)
/// @param m number of elements
/// @param bitsForN number of bits of the universe, i.e. ceilLog2(n)
/// @return max(floor(log2(m / max(1, floor(log2(m))))), bitsForN - 63), e.g. prefixBitsFor(1'000'000, 40) = 15
inline uint32_t prefixBitsFor(uint64_t m, uint32_t bitsForN) {
	const uint64_t log2m = std::max<uint64_t>(1, floorLog2(m));
	const uint32_t prefixBits = floorLog2(m / log2m);
	const uint32_t minPrefixBits = bitsForN > 63 ? bitsForN - 63 : 0;
	return std::max(prefixBits, minPrefixBits);
}

/// @brief write the nBits low bits of val starting at bit position pos of a packed bit array
/// 
/// @param buffer the packed bit array as 64-bit words: bit i of the array is bit i % 64 of buffer[i / 64]
/// @param pos position of the first bit to write, counted from bit 0 of buffer[0]
/// @param nBits number of bits to write, in [0, 64]
/// @param val the value to write; must fit in nBits bits (higher bits are not masked and would overwrite the next field)
/// 
/// @pre bits are OR-ed into the buffer, so the target bits must be zero beforehand (e.g. a zero-initialized buffer):
///       writing twice on the same field corrupts it
/// @pre buffer needs one padding word after the last field
inline void writeBits(std::span<uint64_t> buffer, uint64_t pos, uint32_t nBits, uint64_t val) {
	if (nBits == 0) return;
	uint64_t word = pos / 64;
	uint32_t off = pos % 64;

	assert(word + 1 < buffer.size());

	/* shifting by 64 is Undefined Behavior in C++. (off = 0).
	We can still botain branchless code by:
	splitting it into two shifts
	unconditional write on the next word, allowed by the padding added
	*/

	buffer[word] |= (val << off); // write on the first word
	buffer[word + 1] |= (val >> 1) >> (63 - off); // and the second one
}

/// @brief read nBits bits starting at bit position pos of a packed bit array
/// @param buffer the packed bit array as 64-bit words: bit i of the array is bit i % 64 of buffer[i / 64]
/// @param pos position of the first bit to read, counted from bit 0 of buffer[0]
/// @param nBits number of bits to read, in [0, 64]
/// @return the nBits bits as an unsigned integer, bit pos being the least significant; 0 if nBits == 0
/// @pre always reads buffer[pos / 64 + 1], even when the field fits in one word:
///       buffer needs one padding word after the last field
inline uint64_t readBits(std::span<const uint64_t> buffer, uint64_t pos, uint32_t nBits) {
	if (nBits == 0) return 0;
	uint64_t mask = lowMask(nBits);
	uint64_t word = pos / 64;
	uint32_t off = pos % 64;

	assert(word + 1 < buffer.size()); 

	uint64_t val = buffer[word] >> off;
	val |= (buffer[word + 1] << 1 << (63 - off));

	return val & mask;
}

/// @brief input validation for the constructors. Must run before sorted_vals.back() is read,
///        which is undefined on an empty span. Does not check uniqueness
/// @param sorted_vals the values the set is built from
/// @throw std::invalid_argument if sorted_vals is empty or not sorted
/// @throw std::out_of_range if the largest value is 2^64 - 1: the universe (largest value + 1) would not fit in 64 bits
inline void checkInput(std::span<const uint64_t> sorted_vals) {
	if (sorted_vals.empty()) [[unlikely]]
		throw std::invalid_argument("Empty vector");
	if (!std::is_sorted(sorted_vals.begin(), sorted_vals.end())) [[unlikely]]
		throw std::invalid_argument("Input vector must be sorted");
	if (sorted_vals.back() == UINT64_MAX) [[unlikely]]
		throw std::out_of_range("element == 2^64 - 1: the universe would not fit in 64 bits");
}