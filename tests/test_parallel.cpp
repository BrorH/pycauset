#include <gtest/gtest.h>
#include "ParallelUtils.hpp"
#include <vector>
#include <atomic>
#include <numeric>

using namespace pycauset;

TEST(ParallelTest, ThreadPoolSingleton) {
    ThreadPool& pool1 = ThreadPool::instance();
    ThreadPool& pool2 = ThreadPool::instance();
    EXPECT_EQ(&pool1, &pool2);
}

TEST(ParallelTest, ParallelForSum) {
    size_t N = 10000;
    std::vector<int> data(N, 1);
    std::atomic<int> sum(0);

    ParallelFor(0, N, [&](size_t i) {
        sum += data[i];
    });

    EXPECT_EQ(sum, N);
}

TEST(ParallelTest, ParallelForSmallRange) {
    // Should run sequentially but produce correct result
    size_t N = 5;
    std::vector<int> data(N, 1);
    std::atomic<int> sum(0);

    ParallelFor(0, N, [&](size_t i) {
        sum += data[i];
    });

    EXPECT_EQ(sum, N);
}

TEST(ParallelTest, ExceptionPropagation) {
    // This test verifies that exceptions in threads are caught and rethrown
    EXPECT_THROW({
        ParallelFor(0, 10, [](size_t i) {
            if (i == 5) throw std::runtime_error("Test Error");
        });
    }, std::runtime_error);
}
