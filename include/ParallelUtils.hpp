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

    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 2; // Fallback

    // If range is small, don't spawn tasks (heuristic)
    // For now, we use a simple heuristic: if range < num_threads, just run sequentially
    if (range < num_threads) {
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

} // namespace pycauset
