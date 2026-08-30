#include "pycauset/core/ParallelUtils.hpp"

namespace pycauset {

size_t ThreadPool::global_num_threads = std::thread::hardware_concurrency();

void ThreadPool::set_num_threads(size_t n) {
    if (n == 0) n = 1;
    global_num_threads = n;
}

size_t ThreadPool::get_num_threads() {
    return global_num_threads;
}

ThreadPool& ThreadPool::instance() {
    // Keep this a normal (non-leaked) Meyers singleton. Its destructor notifies
    // workers to stop and joins them, which is safe and keeps worker threads from
    // leaking into forked children (macOS). The leaked singletons that matter for
    // teardown ordering are MemoryGovernor/ComputeContext, whose destructors are
    // reached from PersistentObject destructors during finalization.
    static ThreadPool pool;
    return pool;
}

ThreadPool::ThreadPool(size_t threads) : stop(false) {
    // Ensure at least one thread
    if (threads == 0) threads = 1;

    for(size_t i = 0; i < threads; ++i)
        workers.emplace_back(
            [this] {
                for(;;) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                            [this]{ return this->stop || !this->tasks.empty(); });
                        
                        if(this->stop && this->tasks.empty())
                            return;
                        
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task();
                }
            }
        );
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers)
        worker.join();
}

} // namespace pycauset
