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
