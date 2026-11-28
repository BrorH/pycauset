#include "pycauset/core/ParallelUtils.hpp"

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace pycauset {

std::atomic<size_t> ThreadPool::global_num_threads{std::thread::hardware_concurrency()};

namespace {
thread_local bool t_is_pool_worker = false;
}

void ThreadPool::set_num_threads(size_t n) {
    if (n == 0) n = 1;
    global_num_threads.store(n);
}

size_t ThreadPool::get_num_threads() {
    return global_num_threads.load();
}

bool ThreadPool::on_worker_thread() {
    return t_is_pool_worker;
}

namespace {
long current_pid() {
#ifdef _WIN32
    return static_cast<long>(_getpid());
#else
    return static_cast<long>(getpid());
#endif
}
std::atomic<ThreadPool*> g_pool{nullptr};
std::atomic<long> g_pool_pid{0};
std::mutex g_pool_mu;
}  // namespace

ThreadPool& ThreadPool::instance() {
    // One pool per process. A forked child (multiprocessing's fork start method) inherits the
    // pool object but not its worker threads: tasks queued there never ran and the first
    // parallel kernel waited forever (review 3.4 C-2). The pool records the pid that built it;
    // in a child a new pool is built, and the parent's is abandoned (its std::thread objects
    // cannot be joined or destroyed there). The pool is never destroyed: at exit its idle
    // workers end with the process.
    const long pid = current_pid();
    ThreadPool* pool = g_pool.load(std::memory_order_acquire);
    if (pool != nullptr && g_pool_pid.load(std::memory_order_acquire) == pid) return *pool;
    std::lock_guard<std::mutex> lock(g_pool_mu);
    pool = g_pool.load(std::memory_order_acquire);
    if (pool == nullptr || g_pool_pid.load(std::memory_order_acquire) != pid) {
        pool = new ThreadPool();
        g_pool_pid.store(pid, std::memory_order_release);
        g_pool.store(pool, std::memory_order_release);
    }
    return *pool;
}

ThreadPool::ThreadPool(size_t threads) : stop(false) {
    // Ensure at least one thread
    if (threads == 0) threads = 1;

    for(size_t i = 0; i < threads; ++i)
        workers.emplace_back(
            [this] {
                t_is_pool_worker = true;
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
