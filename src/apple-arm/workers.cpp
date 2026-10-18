#include "apple-arm/execution.hpp"

#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <sys/sysctl.h>
#include <thread>

namespace tsac::apple_arm
{
unsigned physical_threads()
{
    unsigned count = 1;
    std::size_t size = sizeof(count);
    return sysctlbyname("hw.physicalcpu", &count, &size, nullptr, 0) == 0 && count ? count : 1;
}

namespace
{
thread_local bool inside_parallel = false;
} // namespace

// Reuse workers across convolution tiles, Snake jobs and independent Transformer
// lanes instead of creating threads for every phase. The submitting thread also
// consumes jobs, so a budget of N needs only N-1 persistent workers. Serialize
// submissions because task_, counters and exception state describe one job batch;
// nested parallel_for calls run locally to avoid worker-on-worker deadlock.
class Execution::Impl
{
public:
    ~Impl()
    {
        stop();
    }

    void run(std::size_t count, unsigned threads, const std::function<void(std::size_t)>& task)
    {
        std::unique_lock submission(submit_mutex_);
        if (workers_.size() < threads - 1)
        {
            stop();
            stopping_ = false;
            try
            {
                // Once a thread starts, inserting its state must not throw:
                // stop() needs every live worker in the collection to wake it.
                workers_.reserve(threads - 1);
                for (unsigned i = 1; i < threads; ++i)
                {
                    auto worker = std::make_unique<Worker>();
                    auto* state = worker.get();
                    worker->thread = std::jthread(
                        [this, state, i]
                        {
                            work(*state, i - 1);
                        });
                    workers_.push_back(std::move(worker));
                }
            }
            catch (...)
            {
                stop();
                throw;
            }
        }
        {
            std::lock_guard lock(mutex_);
            task_ = &task;
            count_ = count;
            next_.store(0);
            error_ = nullptr;
            active_ = std::min<std::size_t>(threads, count) - 1;
            remaining_ = active_;
            ++generation_;
        }
        // Inactive workers retain their threads and sleep through this batch.
        // A condition variable per worker avoids waking the full pool for a
        // small number of independent Transformer lanes.
        for (std::size_t i = 0; i < active_; ++i)
        {
            workers_[i]->ready.notify_one();
        }
        consume();
        std::unique_lock lock(mutex_);
        // task_ borrows the caller's function. Join the whole batch before
        // releasing that reference or propagating a worker exception.
        done_.wait(lock,
                   [&]
                   {
                       return remaining_ == 0;
                   });
        task_ = nullptr;
        if (error_)
        {
            std::rethrow_exception(error_);
        }
    }

private:
    struct Worker
    {
        std::condition_variable ready;
        std::jthread thread;
    };

    void stop()
    {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        for (auto& worker : workers_)
        {
            worker->ready.notify_one();
        }
        workers_.clear();
    }

    void consume()
    {
        inside_parallel = true;
        try
        {
            for (auto i = next_.fetch_add(1); i < count_; i = next_.fetch_add(1))
            {
                (*task_)(i);
            }
        }
        catch (...)
        {
            // Stop handing out jobs after failure; already running jobs must
            // still finish so their captured buffers remain alive until run returns.
            next_.store(count_);
            std::lock_guard lock(mutex_);
            if (!error_)
            {
                error_ = std::current_exception();
            }
        }
        inside_parallel = false;
    }

    void work(Worker& worker, std::size_t index)
    {
        std::unique_lock lock(mutex_);
        auto seen = generation_;
        // Newly created workers must participate if submission began before
        // they acquired the lock.
        if (task_ && index < active_)
        {
            --seen;
        }
        for (;;)
        {
            worker.ready.wait(lock,
                              [&]
                              {
                                  return stopping_ || (generation_ != seen && index < active_);
                              });
            if (stopping_)
            {
                return;
            }
            seen = generation_;
            lock.unlock();
            consume();
            lock.lock();
            if (--remaining_ == 0)
            {
                done_.notify_one();
            }
        }
    }

    std::mutex submit_mutex_, mutex_;
    std::condition_variable done_;
    std::vector<std::unique_ptr<Worker>> workers_;
    const std::function<void(std::size_t)>* task_ = nullptr;
    std::atomic<std::size_t> next_{0};
    std::size_t count_ = 0, generation_ = 0, remaining_ = 0, active_ = 0;
    bool stopping_ = false;
    std::exception_ptr error_;
};

Execution::Execution(const unsigned& threads) : threads_(threads), impl_(std::make_unique<Impl>())
{
}

Execution::~Execution() = default;

bool Execution::parallel() const
{
    return threads_ > 1;
}

bool Execution::single_threaded_blas() const
{
    return threads_ != 0;
}

void Execution::parallel_for(std::size_t count, const std::function<void(std::size_t)>& task) const
{
    const auto threads = std::min<std::size_t>(std::max(1u, threads_), count);
    if (threads <= 1 || inside_parallel)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            task(i);
        }
        return;
    }
    impl_->run(count, threads_, task);
}
} // namespace tsac::apple_arm
