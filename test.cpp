#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <set>
#include <span>
#include <concepts>
#include <chrono>
#include <vector>
#include <ranges>
#include <fstream>
#include <iomanip>

#include "concept_type.hpp"
#include "EF1.hpp"
#include "EF2.hpp"
#include "EF3.hpp"
#include "EF4.hpp"
#include "EF5.hpp"
#include "EF_vigna.hpp"

using u64 = uint64_t;

constexpr int RANDOM_SEED = 42;
std::mt19937_64 rng(RANDOM_SEED);

// needed to adapt the number of queries to the average time per query
// in order to set a maximum time limit for the benchmark
constexpr u64 max_nanosec = 1e9; // 1 sec
constexpr u64 max_cold_q = 100;
constexpr u64 max_timed_q = 100000;

constexpr auto max_correctness_time = std::chrono::seconds(60);

std::ofstream csv; // results
u64 run_id = 0; // unique id for each run

/**
* @brief generate a vector of random queries in the given range
* @param num_queries the number of queries to generate
* @param query_range the range of queries to generate, inclusive
*/
std::vector<u64> generate_queries(size_t num_queries, std::pair<u64, u64> query_range) {
    std::vector<u64> queries{};
	queries.resize(num_queries);
    for (size_t i = 0; i < num_queries; i++) queries[i] = rng() % (query_range.second - query_range.first) + query_range.first;
    return queries;
}

/**
* @brief Check the correctness of an Elias-Fano implementation against a sorted vector of values.
*/
template<typename T>
requires EF<T>
void checkCorrectness(std::span<const u64> sorted_vals, const T& ef_impl, std::string_view label, std::string_view impl_name) {
    std::cout << "Checking correctness of " << label << " on " << impl_name << "...\n";
    auto deadline = std::chrono::steady_clock::now() + max_correctness_time;

    assert(ef_impl.size() == sorted_vals.size());

    // access test
    size_t k = 0;
    for (; k < sorted_vals.size(); k++) {
        assert(ef_impl.access(k) == sorted_vals[k]);
        if ((k & 0xFFF) == 0 && std::chrono::steady_clock::now() > deadline)  break;
		// k & 0xFFF because: access costs ~10ns while now() costs ~25 sec. one test every 4096 iteratons is enough to check the deadline without slowing down the test too much
    }

    // predecessor successor and contains tests
    u64 universe = sorted_vals.empty() ? 0 : sorted_vals.back() + 1;
	
	constexpr u64 EXHAUSTIVE_LIMIT = 100'000; // if universe is small, exhaustive search. else 100k spot-checks
    u64 target = (universe <= EXHAUSTIVE_LIMIT) ? universe : EXHAUSTIVE_LIMIT;
    
    auto check = [&](u64 x) {
        size_t i_up = std::ranges::upper_bound(sorted_vals, x) - sorted_vals.begin();
        size_t i_lo = std::ranges::lower_bound(sorted_vals, x) - sorted_vals.begin();

        // predecessor  
        if (i_up > 0)
            assert(ef_impl.predecessor(x) == std::optional(sorted_vals[i_up - 1]));
        else
            assert(ef_impl.predecessor(x) == std::nullopt);

        // successor
        if (i_lo < sorted_vals.size())
            assert(ef_impl.successor(x) == std::optional(sorted_vals[i_lo]));
        else
            assert(ef_impl.successor(x) == std::nullopt);

        // contains
        assert(ef_impl.contains(x) == (i_lo < sorted_vals.size() && sorted_vals[i_lo] == x));
    };

    u64 done = 0;
    if (universe <= EXHAUSTIVE_LIMIT) 
        for (u64 x = 0; x < universe; x++) {
            check(x);
            done++;
            if ((x & 15) == 0 && std::chrono::steady_clock::now() > deadline) break;
        }
    else
        for (u64 i = 0; i < EXHAUSTIVE_LIMIT; i++) {
			check(rng() % universe);
            done++;
            if ((i & 15) == 0 && std::chrono::steady_clock::now() > deadline) break;
        }
    
    std::cout << (k < sorted_vals.size() || done < target ? "PARTIAL: " : "OK: ")
        << label << " (m=" << sorted_vals.size()
        << ", n=" << universe << ", access=" << k 
        << "/" << sorted_vals.size()
        << ", checks=" << done << "/" << target
        << ", " << ef_impl.bytes() << " bytes)\n";
}

enum class TestType {
    ACCESS, 
    PREDECESSOR, 
    SUCCESSOR,
    CONTAINS
};

/**
* @brief for the given Elias-Fano implementation, run the specified queries and return a dummy result to prevent compiler optimizations
* @param ef_impl the Elias-Fano implementation
* @param type the type of query to run
* @param queries the queries to run
*/
template<typename T>
requires EF<T>
u64 run_queries(const T& ef_impl, TestType type, std::span<const u64> queries) {
	u64 result = 0; // to prevent compiler optimizations
    switch (type) {
        case TestType::ACCESS:
            for (u64 x : queries) result ^= ef_impl.access(x).value_or(0);
            break;
        case TestType::PREDECESSOR:
            for (u64 x : queries) result ^= ef_impl.predecessor(x).value_or(0);
            break;
        case TestType::SUCCESSOR:
            for (u64 x : queries) result ^= ef_impl.successor(x).value_or(0);
            break;
        case TestType::CONTAINS:
            for (u64 x : queries) result ^= ef_impl.contains(x);
            break;
    }
    return result;
}

/**
* @brief benchmark the given Elias-Fano implementation for the specified query type. 
* @details it estimates the number of queries to run based on the time taken for a small number of queries, and then runs the benchmark for that many queries. It also runs a cold start benchmark for a small number of queries to warm up the cache.
* @param ef_impl the Elias-Fano implementation
* @param type the type of query to benchmark
* @param test_label a label for the benchmark, used for printing results
* @param impl_name the name of the Elias-Fano implementation, used for printing results
* @param query_range an optional range of queries to run, if not provided, the range will be [0, ef_impl.universe())
*/
template<typename T>
requires EF<T>
void benchmark(const T& ef_impl, TestType type, std::string_view test_label, std::string_view impl_name,
               const std::vector<u64>& cold_q, const std::vector<u64>& timed_q, bool bounded) {
    
    
	// estimate num_queries to run based on the time taken for a small number of queries
	
    auto start = std::chrono::steady_clock::now();
    u64 dummy = run_queries(ef_impl, type, std::span<const u64>(cold_q.data(), 10));
    auto end = std::chrono::steady_clock::now();
    double time_per_query = std::max(0.1, (std::chrono::duration<double, std::nano>(end - start)).count() / 10);

	size_t num_cold_q = std::min<size_t>(max_cold_q, max_nanosec/time_per_query);
	size_t num_timed_q = std::min<size_t>(max_timed_q, max_nanosec/time_per_query);

    // cold start
    dummy ^= run_queries(ef_impl, type, std::span<const u64>(cold_q.data(), num_cold_q));

    // timed queries
    start = std::chrono::steady_clock::now();
    dummy ^= run_queries(ef_impl, type, std::span<const u64>(timed_q.data(), num_timed_q));
    end = std::chrono::steady_clock::now();

    double elapsed = (std::chrono::duration<double, std::nano>(end - start)).count();
    double ns_per_query = elapsed / num_timed_q;

    const char* type_str = "";
    if (type == TestType::ACCESS) type_str = "Access";
    if (type == TestType::PREDECESSOR) type_str = "Predecessor";
    if (type == TestType::SUCCESSOR) type_str = "Successor";
    if (type == TestType::CONTAINS) type_str = "Contains";

    std::cout << "Benchmark " << test_label 
        << " on " << impl_name << "\n"
        << "\ttype:" << type_str
        << "\telapsed time: " << elapsed
        << "\tns per query: " << ns_per_query
        << " (dummy=" << dummy << ")" << std::endl;

    csv << run_id << ';'
        << test_label << ';' << impl_name << ';'
        << ef_impl.size() << ';' << ef_impl.universe() << ';' << ef_impl.bytes() << ';'
        << type_str << ';' << num_timed_q << ';'
        << elapsed << ';' << ns_per_query << ';'
        << (bounded ? 1 : 0) << '\n';
}

/**
* @brief run the full test suite for a given Elias-Fano implementation, including correctness checks and benchmarks for all query types
* @param sorted_vals the sorted vector of values to use for testing, aka the dataset to build the Elias-Fano structure from
* @param test_label a label for the test suite, used for printing results
* @param impl_name the name of the Elias-Fano implementation, used for printing results
* @param query_range an optional range of queries to run, if not provided, the range will be [0, ef_impl.universe())
*/
template<typename T>
requires EF<T>
void run_test_suite(const std::vector<u64>& sorted_vals, std::string_view label, std::string_view impl_name,
                    const std::vector<u64>& cold_q, const std::vector<u64>& timed_q, 
                    const std::vector<u64>& access_cold_q, const std::vector<u64>& access_timed_q,
                    bool bounded) {
    T ef(sorted_vals);
    checkCorrectness(sorted_vals, ef, label, impl_name);
    std::cout << "Running benchmarks for " << label << " on " << impl_name <<"...\n";
    benchmark(ef, TestType::ACCESS, label, impl_name, access_cold_q, access_timed_q, false);
    benchmark(ef, TestType::PREDECESSOR, label, impl_name, cold_q, timed_q, bounded);
    benchmark(ef, TestType::SUCCESSOR, label, impl_name, cold_q, timed_q, bounded);
    benchmark(ef, TestType::CONTAINS, label, impl_name, cold_q, timed_q, bounded);
    std::cout << std::string(60, '-') << "\n";
}

/**
* @brief run the full test suite for every Elias-Fano implementations, including correctness checks and benchmarks for all query types
* @param sorted_vals the sorted vector of values to use for testing, aka the dataset to build the Elias-Fano structure from
* @param test_label a label for the test suite, used for printing results
* @param query_range an optional range of queries to run, if not provided, the range will be [0, ef_impl.universe())
*/
void test_every_method(const std::vector<u64>& sorted_vals, std::string_view label,
                    std::optional<std::pair<u64,u64>> query_range = std::nullopt) {

	auto cold_q = generate_queries(max_cold_q, query_range ? *query_range : std::pair<u64, u64>{ 0, sorted_vals.back() + 1 });
	auto timed_q = generate_queries(max_timed_q, query_range ? *query_range : std::pair<u64, u64>{ 0, sorted_vals.back() + 1 });
	
	auto access_cold_q = generate_queries(max_cold_q, { 0, sorted_vals.size() });
    auto access_timed_q = generate_queries(max_timed_q, { 0, sorted_vals.size() });

    const bool bounded = query_range.has_value();

    run_test_suite<EF1>(sorted_vals, label, "EF1", cold_q, timed_q, access_cold_q, access_timed_q, bounded);
    run_test_suite<EF2>(sorted_vals, label, "EF2", cold_q, timed_q, access_cold_q, access_timed_q, bounded);
    run_test_suite<EF3>(sorted_vals, label, "EF3", cold_q, timed_q, access_cold_q, access_timed_q, bounded);
    run_test_suite<EF4>(sorted_vals, label, "EF4", cold_q, timed_q, access_cold_q, access_timed_q, bounded);
    run_test_suite<EF5<0.0f>>(sorted_vals, label, "EF5 0", cold_q, timed_q, access_cold_q, access_timed_q, bounded);
    run_test_suite<EF5<0.5f>>(sorted_vals, label, "EF5 0.5", cold_q, timed_q, access_cold_q, access_timed_q, bounded);
    run_test_suite<EF5<3.0f>>(sorted_vals, label, "EF5 3", cold_q, timed_q, access_cold_q, access_timed_q, bounded);
    run_test_suite<VignaEFSet>(sorted_vals, label, "Vigna", cold_q, timed_q, access_cold_q, access_timed_q, bounded);
}


int main() {

    run_id = std::chrono::system_clock::now().time_since_epoch().count();
    csv.open("bench_" + std::to_string(run_id) + ".csv"); 
	csv << std::unitbuf; // flush after each write
    csv << std::setprecision(10);
    csv << "run_id;test;impl;m;n;bytes;query_type;num_queries;elapsed_ns;ns_per_query;bounded\n";



    // 1) Random uniform
    {
        uint64_t n = 1'000'000, m = 5000;
        std::set<uint64_t> dedup;
        while (dedup.size() < m) dedup.insert(rng() % n);
        std::vector<uint64_t> sorted(dedup.begin(), dedup.end());
        test_every_method(sorted, "1. Random uniform");
        
        dedup.insert(n-1); 
        sorted.assign(dedup.begin(), dedup.end());
        test_every_method(sorted, "1b. Random uniform with n-1");
    }
    // 2) Dense cluster — most buckets empty
    {
        uint64_t n = 1'000'000, m = 1000;
        std::set<uint64_t> dedup;
        uint64_t base = 500'000;
        while (dedup.size() < m) dedup.insert(base + (rng() % 2000));
        std::vector<uint64_t> sorted(dedup.begin(), dedup.end());
        test_every_method(sorted, "2. Dense cluster");

        dedup.insert(n-1);
        sorted.assign(dedup.begin(), dedup.end());
        test_every_method(sorted, "2b. Dense cluster with n-1");
    }
    // 3) Extreme sparsity: few elements, huge universe
    {
        uint64_t n = 1ULL << 40, m = 10;
        std::set<uint64_t> dedup;
        while (dedup.size() < m) dedup.insert(rng() % n);
        std::vector<uint64_t> sorted(dedup.begin(), dedup.end());
        test_every_method(sorted, "3. Extreme sparsity (m=10, n=2^40), tight universe");

        dedup.insert(n-1);
        sorted.assign(dedup.begin(), dedup.end());
        test_every_method(sorted, "3b. Extreme sparsity (m=10, n=2^40) with n-1");
    }
    // 4) Small edge cases
    {
        test_every_method({0}, "4a. Single {0} n=1 ");
        test_every_method({99}, "4b. Single {99} n=100");
        test_every_method({0, 99}, "4c. Two {0,99} n=100");
    }
    // 5) All elements at start of universe
    {
        std::vector<uint64_t> sorted;
        for (uint64_t i = 0; i < 500; i++) sorted.push_back(i);
        test_every_method(sorted, "5. n=m, elements at start: first 500");
    }
    // 6) All elements at end of universe
    {
        std::vector<uint64_t> sorted;
        for (uint64_t i = 0; i < 500; i++) sorted.push_back(999'500 + i);
        test_every_method(sorted, "6. n=1000000, elements at end: last 500");
    }
    // 7) Large random
    {
        uint64_t n = 1ULL << 32, m = 100'000;
        std::set<uint64_t> dedup;
        while (dedup.size() < m) dedup.insert(rng() % n);
        std::vector<uint64_t> sorted(dedup.begin(), dedup.end());
        test_every_method(sorted, "7. large random (n=2^32) tight universe");

        dedup.insert(n-1);
        sorted.assign(dedup.begin(), dedup.end());
        test_every_method(sorted, "7b. large random (n=2^32) with n-1");
    }
    // 8) Evenly spaced elements (stride)
    {
        uint64_t n = 1'000'000;
        std::vector<uint64_t> sorted;
        for (uint64_t i = 0; i < n; i += 100) sorted.push_back(i);
        test_every_method(sorted, "8. Evenly spaced (stride=100)");
    }
    // 9) Two disjoint dense clusters separated by huge gap
    {
        uint64_t n = 1'000'000;
        std::set<uint64_t> dedup;
        for(int i=0; i<500; ++i) dedup.insert(rng() % 1000); 
        for(int i=0; i<500; ++i) dedup.insert(999'000 + (rng() % 1000));
        std::vector<uint64_t> sorted(dedup.begin(), dedup.end());
        test_every_method(sorted, "9. Two disjoint clusters, tight universe");

        dedup.insert(n-1);
        sorted.assign(dedup.begin(), dedup.end());
        test_every_method(sorted, "9b. Two disjoint clusters, with n-1");
    }
    // 10) Exponential gaps
    {
        uint64_t n = 1ULL << 60;
        std::vector<uint64_t> sorted;
        for (uint64_t i = 1; i < 60; i++) sorted.push_back(1ULL << i);
        test_every_method(sorted, "10. Exponential gaps (powers of 2), tight universe");
        if (sorted.back() != n - 1) {
            sorted.push_back(n - 1);
            test_every_method(sorted, "10b. Exponential gaps (powers of 2) with n-1");
        }
    }
    // 11) m = 10M elements
    {
        uint64_t n = 1ULL << 63;
        uint64_t m = 10000000;
        std::set<uint64_t> dedup;
        while (dedup.size() < m) dedup.insert(rng() % n);
        std::vector<uint64_t> sorted(dedup.begin(), dedup.end());
        test_every_method(sorted, "11. m = 10M elements tight universe");

        dedup.insert(n-1);
        sorted.assign(dedup.begin(), dedup.end());
        test_every_method(sorted, "11b. m = 10M elements with n-1");
    }
    // 12) 50M Random Walk (90% small gaps, 10% massive gaps)
    {
        uint64_t n = 1ULL << 63;
        uint64_t m = 50'000'000;
        std::vector<uint64_t> sorted;
        sorted.reserve(m);
        
        uint64_t current = 0;
        for (uint64_t i = 0; i < m; i++) {
            sorted.push_back(current);
            if (rng() % 100 < 90) {
                current += (rng() % 100) + 1;           // 90% chance: gap of 1-100
            } else {
                current += (rng() % (1ULL << 30)) + 1;  // 10% chance: massive gap
            }
            if (current >= n) break; // Prevent overflow
        }
        
        // Ensure unique and strictly increasing
        sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
        
        test_every_method(sorted, "12. 50M Random Walk tight universe");

        if (sorted.back() != n - 1) {
            sorted.push_back(n - 1);
            test_every_method(sorted, "12b. 50M Random Walk with n-1");
        }
    }
    // 13) 100M Contiguous Bursts
    {
        uint64_t n = 1ULL << 63;
        uint64_t num_clusters = 100'000;
        uint64_t burst_size = 1000;
        
        std::vector<uint64_t> sorted;
        sorted.reserve(num_clusters * burst_size);
        
        for (uint64_t i = 0; i < num_clusters; i++) {
            uint64_t start_val = rng() % (n - burst_size);
            for (uint64_t j = 0; j < burst_size; j++) {
                sorted.push_back(start_val + j);
            }
        }
        
        std::sort(sorted.begin(), sorted.end());
        sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
        
        test_every_method(sorted, "13. 100M Contiguous Bursts, tight universe");

        if (sorted.back() != n - 1) {
            sorted.push_back(n - 1);
            test_every_method(sorted, "13b. 100M Contiguous Bursts with n-1");
        }
    }
    // 14) 50M Geometric (Power-Law) Gaps
    {
        uint64_t n = 1ULL << 63;
        uint64_t m = 50'000'000;
        std::vector<uint64_t> sorted;
        sorted.reserve(m);
        
        uint64_t current = 0;
        for (uint64_t i = 0; i < m; i++) {
            sorted.push_back(current);
            // Gap magnitude is 2^X, where X is random between 0 and 35
            uint64_t bit_width = rng() % 36; 
            current += (1ULL << bit_width) + (rng() % 10); // Base jump + noise
            if (current >= n) break;
        }
        
        sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
        
        test_every_method(sorted, "14. 50M Geometric Gaps tight universe");

        if (sorted.back() != n - 1) {
            sorted.push_back(n - 1);
            test_every_method(sorted, "14b. 50M Geometric Gaps with n-1");
        }
    }
    // 15) 10M Adversarial Single-Prefix Cluster
    {
        uint64_t n = 1ULL << 63;
        uint64_t m = 10'000'000;
        std::vector<uint64_t> sorted;
        sorted.reserve(m);
        
        // Fix the top 40 bits to a random static value
        uint64_t fixed_prefix = (rng() % (1ULL << 40)) << 23; 
        
        for (uint64_t i = 0; i < m; i++) {
            // Only the bottom 23 bits vary
            sorted.push_back(fixed_prefix | (rng() % (1ULL << 23)));
        }
        
        std::sort(sorted.begin(), sorted.end());
        sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
        
        test_every_method(sorted, "15. 10M Adversarial Single-Prefix tight universe");

        if (sorted.back() != n - 1) {
            sorted.push_back(n - 1);
            test_every_method(sorted, "15b. 10M Adversarial Single-Prefix with n-1");
        }
    }
    // 16) 100M True Uniform Random
    {
        uint64_t n = 1ULL << 63;
        uint64_t m = 100'000'000;
        std::vector<uint64_t> sorted(m);
        
        for (auto& val : sorted) {
            val = rng() % n;
        }
        
        std::sort(sorted.begin(), sorted.end());
        sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
        
        test_every_method(sorted, "16. 100M Uniform Random tight universe");

        if (sorted.back() != n - 1) {
            sorted.push_back(n - 1);
            test_every_method(sorted, "16b. 100M Uniform Random with n-1");
        }
    }
    // 17) Uniform scan to map EF2 vs Vigna crossover
    {
        for (uint64_t m : {10000ULL, 50000ULL, 200000ULL, 500000ULL, 2000000ULL}) {
            uint64_t n = 1000 * m;  // density kept roughly constant
            std::set<uint64_t> dedup;
            while (dedup.size() < m) dedup.insert(rng() % n);
            std::vector<uint64_t> sorted(dedup.begin(), dedup.end());
            std::string label = "17. Uniform crossover m=" + std::to_string(m) + " tight universe";
            test_every_method(sorted, label);

            std::string label_b = "17b. Uniform crossover m=" + std::to_string(m) + " with n-1";
            if (sorted.back() != n - 1) {   
                sorted.push_back(n - 1);
                test_every_method(sorted, label_b);
            }
        }
    }

    // 18) Uniform sparse large m - stresses EF2 exp search in uniform regime
    {
        uint64_t m = 1'000'000, n = 1ULL << 50;  // n/m = 2^30, many empty buckets
        std::set<uint64_t> dedup;
        while (dedup.size() < m) dedup.insert(rng() % n);
        std::vector<uint64_t> sorted(dedup.begin(), dedup.end());
        test_every_method(sorted, "18. Uniform sparse m=1M tight universe");

        if (sorted.back() != n - 1) {
            sorted.push_back(n - 1);
            test_every_method(sorted, "18b. Uniform sparse m=1M with n-1");
        }
    }
    // 19) 50M Random Walk with queries restricted to [min, max] of data
    //     removes the "query in empty universe tail" artifact that inflated test 12
    {
        std::mt19937_64 local_rng(12345);
        uint64_t m = 50'000'000, n = 1ULL << 63;
        std::vector<uint64_t> sorted;
        sorted.reserve(m);
        uint64_t cur = 0;
        for (uint64_t i = 0; i < m; ++i) {
            cur += 1 + (local_rng() % 1000);
            sorted.push_back(cur);
        }
        auto range = std::make_pair(sorted.front(), sorted.back() + 1);
        test_every_method(sorted, "19. 50M Random Walk [bounded queries] tight universe", range);

        if (sorted.back() != n - 1) {
            sorted.push_back(n - 1);
            test_every_method(sorted, "19b. 50M Random Walk [bounded queries] with n-1", range);
        }
    }

    std::cout << "\nAll tests passed.\n";
    return 0;
}
