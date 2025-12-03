#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <algorithm>

namespace pycauset {

class ThreadPool {
public:
    // Singleton access
    static ThreadPool& instance();

    // Delete copy/move
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Enqueue a task
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    // Destructor
    ~ThreadPool();

    // Global concurrency control
    static void set_num_threads(size_t n);
    static size_t get_num_threads();

private:
    // Private constructor for Singleton
    ThreadPool(size_t threads = std::thread::hardware_concurrency());

    // Worker threads
    std::vector<std::thread> workers;
    
    // Task queue
    std::queue<std::function<void()>> tasks;
    
    // Synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
    
    // Concurrency setting
    static size_t global_num_threads;
};

// Template implementation must be in header
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::invoke_result<F, Args...>::type>
{
    using return_type = typename std::invoke_result<F, Args...>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
        
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // Don't allow enqueueing after stopping
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}

// ParallelFor Helper
// Executes func(i) for i in [start, end) in parallel
template<typename Func>
void ParallelFor(size_t start, size_t end, Func func) {
    size_t range = end - start;
    if (range == 0) return;

    size_t num_threads = ThreadPool::get_num_threads();
    if (num_threads == 0) num_threads = 1;

    // If range is small or single thread requested, run sequentially
    if (range < num_threads || num_threads == 1) {
        for (size_t i = start; i < end; ++i) {
            func(i);
        }
        return;
    }

    size_t chunk_size = (range + num_threads - 1) / num_threads;
    
    std::vector<std::future<void>> futures;
    
    for (size_t i = 0; i < num_threads; ++i) {
        size_t chunk_start = start + i * chunk_size;
        size_t chunk_end = std::min(end, chunk_start + chunk_size);
        
        if (chunk_start >= end) break;

        futures.emplace_back(ThreadPool::instance().enqueue([=]() {
            for (size_t j = chunk_start; j < chunk_end; ++j) {
                func(j);
            }
        }));
    }

    // Wait for all chunks to complete and propagate exceptions
    for (auto& fut : futures) {
        fut.get(); 
    }
}

// ParallelBlockMap Helper
// Iterates over a 2D range [0, rows) x [0, cols) in blocks
// Kernel signature: void(size_t i_start, size_t i_end, size_t j_start, size_t j_end)
template<typename Kernel>
void ParallelBlockMap(size_t rows, size_t cols, size_t block_size, Kernel kernel) {
    size_t num_block_rows = (rows + block_size - 1) / block_size;
    
    ParallelFor(0, num_block_rows, [&](size_t bi) {
        size_t i_start = bi * block_size;
        size_t i_end = std::min(i_start + block_size, rows);
        
        for (size_t j_start = 0; j_start < cols; j_start += block_size) {
            size_t j_end = std::min(j_start + block_size, cols);
            
            // Execute kernel on the block [i_start, i_end) x [j_start, j_end)
            kernel(i_start, i_end, j_start, j_end);
        }
    });
}

} // namespace pycauset
