#include "apple-arm/kernels.hpp"
#include "scalar/dsp.hpp"

#include <array>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <set>
#include <thread>

namespace
{
void check(bool condition, const char* message)
{
    if (!condition)
    {
        throw tsac::Error(message);
    }
}

void worker_reuse()
{
    const unsigned budget = 8;
    tsac::apple_arm::Execution execution(budget);
    const auto batch = [&](std::size_t count)
    {
        std::mutex mutex;
        std::condition_variable ready;
        std::set<std::thread::id> threads;
        execution.parallel_for(count,
                               [&](std::size_t)
                               {
                                   std::unique_lock lock(mutex);
                                   threads.insert(std::this_thread::get_id());
                                   ready.notify_all();
                                   check(ready.wait_for(lock, std::chrono::seconds(5),
                                                        [&]
                                                        {
                                                            return threads.size() == count;
                                                        }),
                                         "inactive worker did not resume");
                               });
        return threads;
    };
    const auto original = batch(8);
    for (unsigned repeat = 0; repeat < 16; ++repeat)
    {
        batch(2);
        batch(3);
        batch(1);
        execution.parallel_for(0,
                               [](std::size_t)
                               {
                                   throw tsac::Error("empty batch ran");
                               });
        bool threw = false;
        try
        {
            execution.parallel_for(3,
                                   [](std::size_t)
                                   {
                                       throw tsac::Error("expected failure");
                                   });
        }
        catch (const tsac::Error&)
        {
            threw = true;
        }
        check(threw, "worker exception was lost");
        check(batch(8) == original, "thread pool recreated inactive workers");
    }
}

void convolutions()
{
    using namespace tsac;
    const unsigned budget = 4;
    apple_arm::Execution execution(budget);
    for (auto kind : {ConvolutionKind::ordinary, ConvolutionKind::transposed})
    {
        for (std::size_t stride : {1u, 2u, 3u, 4u})
        {
            for (std::size_t dilation : {1u, 2u, 3u, 4u})
            {
                for (std::size_t padding : {0u, 1u, 4u})
                {
                    for (std::size_t frames : {3u, 17u, 513u})
                    {
                        const Conv1dConfig config{3, 5, 3, stride, padding, dilation, kind};
                        const auto span = (config.kernel - 1) * dilation + 1;
                        if ((kind == ConvolutionKind::ordinary && frames + 2 * padding < span) ||
                            (kind == ConvolutionKind::transposed &&
                             (frames - 1) * stride + span <= 2 * padding))
                        {
                            continue;
                        }
                        std::vector<float> weights(45), input(frames * 3 * 2);
                        const std::array<float, 5> bias{1, -2, 3, -4, 5};
                        for (std::size_t i = 0; i < weights.size(); ++i)
                        {
                            weights[i] = static_cast<float>(static_cast<int>(i % 11) - 5) / 16;
                        }
                        for (std::size_t i = 0; i < input.size(); ++i)
                        {
                            input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 8;
                        }
                        scalar::Conv1d reference(config, weights, bias);
                        apple_arm::Conv1d candidate(config, weights, bias);
                        const SignalShape shape{3, frames, 2};
                        const auto expected = reference.apply(input, shape).values;
                        check(candidate.apply(input, shape).values == expected,
                              "serial convolution lowering differs");
                        candidate.set_execution(&execution);
                        check(candidate.apply(input, shape).values == expected,
                              "parallel convolution phases differ");
                    }
                }
            }
        }
    }
}

} // namespace

int main()
{
    try
    {
        worker_reuse();
        convolutions();
        std::cout << "Apple worker and convolution checks passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
