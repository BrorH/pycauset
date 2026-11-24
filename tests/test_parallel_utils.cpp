#include <gtest/gtest.h>
#include "pycauset/core/ParallelUtils.hpp"
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>
#include <numeric>

using namespace pycauset;

TEST(ParallelUtilsTest, BasicRange) {
    const size_t N = 1000;
    std::vector<int> data(N, 0);
    
    ParallelFor(0, N, [&](size_t i) {
        data[i] = 1;
    });
    
    for (size_t i = 0; i < N; ++i) {
        EXPECT_EQ(data[i], 1) << "Index " << i << " was not processed";
    }
}

TEST(ParallelUtilsTest, UnevenWorkload) {
    const size_t N = 100;
    std::atomic<int> completed{0};
    
    ParallelFor(0, N, [&](size_t i) {
        if (i % 10 == 0) {
            // Simulate heavy work
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        completed++;
    });
    
    EXPECT_EQ(completed.load(), N);
}

TEST(ParallelUtilsTest, ExceptionPropagation) {
    const size_t N = 100;
    
    EXPECT_THROW({
        ParallelFor(0, N, [&](size_t i) {
            if (i == 50) {
                throw std::runtime_error("Test Exception");
            }
        });
    }, std::runtime_error);
}

TEST(ParallelUtilsTest, NestedParallelism) {
    // ParallelFor should handle nested calls (though it might serialize them or use the pool)
    const size_t N = 10;
    std::atomic<int> count{0};
    
    ParallelFor(0, N, [&](size_t i) {
        ParallelFor(0, N, [&](size_t j) {
            count++;
        });
    });
    
    EXPECT_EQ(count.load(), N * N);
}

TEST(ParallelUtilsTest, ExceptionWaitsForAllWorkers) {
    // F-005: when a chunk throws, ParallelFor must not return (rethrow) while other
    // workers are still running the body, which references this stack frame.
    const size_t N = 4000;
    std::atomic<int> running{0};
    std::atomic<int> max_after_return{0};
    try {
        ParallelFor(0, N, [&](size_t i) {
            running++;
            if (i == 0) {
                running--;
                throw std::runtime_error("boom");
            }
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            running--;
        });
        FAIL() << "expected exception";
    } catch (const std::runtime_error&) {
        max_after_return = running.load();
    }
    EXPECT_EQ(max_after_return.load(), 0) << "workers still running after ParallelFor rethrew";
}

TEST(ParallelUtilsTest, NestedParallelismManyLevels) {
    // F-004: nesting must never deadlock, even three levels deep.
    std::atomic<int> count{0};
    ParallelFor(0, 8, [&](size_t) {
        ParallelFor(0, 8, [&](size_t) {
            ParallelFor(0, 8, [&](size_t) { count++; });
        });
    });
    EXPECT_EQ(count.load(), 8 * 8 * 8);
}
