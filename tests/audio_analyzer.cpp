#include "tsac/apple_arm_analyzer.hpp"
#include "tsac/metal_analyzer.hpp"

#include "apple-arm/execution.hpp"
#include "format/entropy.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>

using namespace tsac;

namespace
{
void check(bool condition, const char* message)
{
    if (!condition)
    {
        throw Error(message);
    }
}

template <class F> void rejects(F&& action, const char* message)
{
    bool rejected = false;
    try
    {
        action();
    }
    catch (const Error&)
    {
        rejected = true;
    }
    check(rejected, message);
}

void execution_budgets()
{
    const unsigned one = 1, two = 2, automatic = 0;
    apple_arm::Execution serial(one), parallel(two), defaults(automatic);
    check(!serial.parallel() && parallel.parallel() && !defaults.parallel(),
          "execution budgets are not independent");
    check(serial.single_threaded_blas() && !defaults.single_threaded_blas(),
          "automatic BLAS policy changed");
    const auto caller = std::this_thread::get_id();
    serial.parallel_for(8,
                        [&](std::size_t)
                        {
                            check(std::this_thread::get_id() == caller,
                                  "serial budget used a worker");
                        });
    std::mutex mutex;
    std::condition_variable ready;
    unsigned entered = 0;
    std::array<std::thread::id, 2> workers;
    parallel.parallel_for(2,
                          [&](std::size_t i)
                          {
                              workers[i] = std::this_thread::get_id();
                              std::unique_lock lock(mutex);
                              ++entered;
                              ready.notify_all();
                              check(ready.wait_for(lock, std::chrono::seconds(5),
                                                   [&]
                                                   {
                                                       return entered == 2;
                                                   }),
                                    "parallel budget did not run two jobs concurrently");
                          });
    check(workers[0] != workers[1], "parallel budget reused one thread");
    std::atomic<unsigned> completed{0};
    parallel.parallel_for(16,
                          [&](std::size_t)
                          {
                              parallel.parallel_for(2,
                                                    [&](std::size_t)
                                                    {
                                                        ++completed;
                                                    });
                          });
    check(completed == 32, "nested work was lost");
    rejects(
        [&]
        {
            parallel.parallel_for(16,
                                  [](std::size_t)
                                  {
                                      throw Error("worker failure");
                                  });
        },
        "lost worker exception");
    parallel.parallel_for(8,
                          [&](std::size_t)
                          {
                              ++completed;
                          });
    check(completed == 40, "worker failure prevented execution reuse");
}

void thread_budgets(const ModelDirectory& directory)
{
    rejects(
        [&]
        {
            AppleArmAnalyzer invalid(directory, {}, 257);
        },
        "accepted excessive thread budget");
    AppleArmAnalyzer serial(directory, {}, 1), parallel(directory, {}, 2);
    std::vector<float> pcm(1025, 0.1f);
    EncodeOptions options;
    options.method = CodingMethod::fast;
    options.codebooks = 1;
    const auto input = interleaved_pcm(pcm, 1);
    const auto compare = [&]
    {
        auto other = std::async(std::launch::async,
                                [&]
                                {
                                    return parallel.encode(input, options);
                                });
        const auto expected = serial.encode(input, options);
        check(other.get() == expected, "thread budget changed encoded recording");
        auto decoded = std::async(std::launch::async,
                                  [&]
                                  {
                                      return parallel.decode(expected);
                                  });
        check(serial.decode(expected).samples == decoded.get().samples,
              "thread budget changed decoded recording");
    };
    compare();
#if TSAC_TEST_TRANSFORMER
    options.method = CodingMethod::normal;
    options.entropy_batch = 2;
    compare();
#endif
}

void session_reuse(const ModelDirectory& directory)
{
    AppleArmAnalyzer analyzer(directory, {}, 2);
    const std::vector<float> pcm(1025, 0.125f);
    const auto input = interleaved_pcm(pcm, 1);
    EncodeOptions options;
    options.method = CodingMethod::fast;
    options.codebooks = 1;
    const auto expected = analyzer.encode(input, options);
    check(analyzer.encode(input, options) == expected, "cached encode changed stream");
    auto concurrent = std::async(std::launch::async,
                                 [&]
                                 {
                                     return analyzer.encode(input, options);
                                 });
    check(analyzer.encode(input, options) == expected && concurrent.get() == expected,
          "concurrent session lease changed stream");
    const auto decoded = analyzer.decode(expected).samples;
    check(analyzer.decode(expected).samples == decoded, "cached decode changed PCM");
    auto other = std::async(std::launch::async,
                            [&]
                            {
                                return analyzer.decode(expected).samples;
                            });
    check(analyzer.decode(expected).samples == decoded && other.get() == decoded,
          "concurrent decode session changed PCM");
    check(analyzer.encode(input, options) == expected, "session role change lost model binding");
    options.codebooks = 2;
    const auto two = analyzer.encode(input, options);
    check(parse_header(two).codebooks == 2 && analyzer.encode(input, options) == two,
          "cached model codebook coverage changed");
#if TSAC_TEST_TRANSFORMER
    options.method = CodingMethod::normal;
    options.entropy_batch = 2;
    const auto normal = analyzer.encode(input, options);
    check(analyzer.encode(input, options) == normal, "cached normal encode retained token state");
    const auto samples = analyzer.decode(normal).samples;
    check(analyzer.decode(normal).samples == samples, "cached normal decode retained token state");
#endif
}

PcmView planar_view(const std::vector<float>& samples, std::size_t channels, std::size_t frames)
{
    return PcmView(samples.data(), PcmView::mapping_type(PcmView::extents_type(channels, frames),
                                                         std::array<std::size_t, 2>{frames, 1}));
}

void views()
{
    const std::array<float, 6> samples{0, 10, 1, 11, 2, 12};
    const auto audio = interleaved_pcm(samples, 2);
    check(audio.extent(0) == 2 && audio.extent(1) == 3, "wrong PCM extents");
    const auto right = channel_view(audio, 1);
    check(right.extent(0) == 3 && right[2] == 12 && right.stride(0) == 2, "wrong channel slice");
    const auto frames = frame_view(audio, 1, 2);
    check(frames[0, 0] == 1 && frames[1, 1] == 12, "wrong frame slice");
    const auto empty = frame_view(audio, 3, 0);
    check(empty.extent(1) == 0 && channel_view(empty, 1).extent(0) == 0, "empty slice failed");
    check(frame_view(interleaved_pcm({}, 2), 0, 0).extent(1) == 0, "null empty slice failed");
    rejects(
        [&]
        {
            channel_view(audio, 2);
        },
        "accepted invalid channel");
    rejects(
        [&]
        {
            frame_view(audio, 2, 2);
        },
        "accepted invalid frame slice");
    rejects(
        [&]
        {
            interleaved_pcm(samples, 4);
        },
        "accepted incomplete PCM frame");
    rejects(
        [&]
        {
            interleaved_pcm(samples, 0);
        },
        "accepted zero channels");
}

void layouts(AudioAnalyzer& analyzer)
{
    constexpr std::size_t frames = 529;
    std::vector<float> interleaved(frames * 2), planar(frames * 2);
    for (std::size_t t = 0; t < frames; ++t)
    {
        for (std::size_t c = 0; c < 2; ++c)
        {
            const float value = 0.2f * std::sin(float(t) * 0.031f + float(c));
            interleaved[t * 2 + c] = planar[c * frames + t] = value;
        }
    }
    EncodeOptions options;
    options.method = CodingMethod::fast;
    options.codebooks = 2;
    const auto input = interleaved_pcm(interleaved, 2);
    const auto stream = analyzer.encode(input, options);
    check(stream == analyzer.encode(planar_view(planar, 2, frames), options),
          "PCM layout changed tokens");
    const auto header = parse_header(stream);
    check(header.frames == 2 && header.model_channels == 2 && header.groups == 1,
          "wrong padded stereo framing");
    const auto decoded = analyzer.decode(stream);
    check(decoded.channels == 2 && decoded.samples.size() == 2048, "wrong decoded PCM size");
    check(std::all_of(decoded.samples.begin(), decoded.samples.end(),
                      [](float x)
                      {
                          return std::isfinite(x);
                      }),
          "nonfinite decoded PCM");
    check(decoded.view().extent(1) == 1024, "wrong owning PCM view");
    const auto slice = frame_view(input, 11, 513);
    const auto copied = std::span(interleaved).subspan(22, 1026);
    check(analyzer.encode(slice, options) == analyzer.encode(interleaved_pcm(copied, 2), options),
          "frame slice changed tokens");
    options.separate_channels = true;
    const auto separate = parse_header(analyzer.encode(input, options));
    check(separate.model_channels == 1 && separate.groups == 2, "separate channel encoding failed");
}

void groups_and_empty(AudioAnalyzer& analyzer)
{
    EncodeOptions options;
    options.method = CodingMethod::fast;
    options.codebooks = 1;
    // Distinct independent channels expose grouping/re-interleaving mistakes.
    std::vector<float> samples(std::size_t{513} * 3);
    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        samples[i] = float(int((i * 37 + i % 3 * 53) % 257) - 128) / 1024;
    }
    const auto stream = analyzer.encode(interleaved_pcm(samples, 3), options);
    const auto header = parse_header(stream);
    check(header.model_channels == 1 && header.groups == 3 && header.frames == 2,
          "wrong grouped framing");
    const auto decoded = analyzer.decode(stream);
    check(decoded.channels == 3 && decoded.samples.size() == 3072, "wrong grouped PCM");
    const auto tokens = unpack_fast(stream);
    for (std::size_t channel = 0; channel < 3; ++channel)
    {
        Header mono = header;
        mono.groups = 1;
        std::vector<std::uint16_t> selected{tokens[channel], tokens[3 + channel]};
        const auto independent = analyzer.decode(pack_fast(mono, selected));
        for (std::size_t frame = 0; frame < 1024; ++frame)
        {
            check(decoded.samples[frame * 3 + channel] == independent.samples[frame],
                  "grouped PCM channel order changed");
        }
    }
    const auto empty = analyzer.encode(interleaved_pcm({}, 2), options);
    const auto output = analyzer.decode(empty);
    check(output.channels == 2 && output.samples.empty(), "empty audio failed");
}

void failures(AudioAnalyzer& analyzer, const ModelDirectory& directory, bool metal)
{
    std::vector<float> pcm(512, 0.1f);
    EncodeOptions options;
    options.method = CodingMethod::fast;
    options.codebooks = 1;
    const auto input = interleaved_pcm(pcm, 1);
    const auto expected = analyzer.encode(input, options);
    pcm[0] = std::numeric_limits<float>::quiet_NaN();
    rejects(
        [&]
        {
            analyzer.encode(input, options);
        },
        "accepted nonfinite PCM");
    pcm[0] = 0.1f;
    auto invalid = options;
    invalid.codebooks = 10;
    rejects(
        [&]
        {
            analyzer.encode(input, invalid);
        },
        "accepted unsupported codebooks");
    invalid = options;
    invalid.entropy_batch = 1;
    rejects(
        [&]
        {
            analyzer.encode(input, invalid);
        },
        "accepted fast entropy batch");
    rejects(
        [&]
        {
            analyzer.decode({});
        },
        "accepted missing stream");
    check(analyzer.encode(input, options) == expected, "failure prevented analyzer reuse");
    Limits limits;
    limits.tensor_elements = 511;
    std::unique_ptr<AudioAnalyzer> bounded;
    if (metal)
    {
        bounded = std::make_unique<MetalAnalyzer>(directory, limits);
    }
    else
    {
        bounded = std::make_unique<AppleArmAnalyzer>(directory, limits, 1);
    }
    rejects(
        [&]
        {
            bounded->encode(input, options);
        },
        "ignored PCM allocation limit");
    rejects(
        [&]
        {
            bounded->decode(expected);
        },
        "ignored output allocation limit");
}

#if TSAC_TEST_TRANSFORMER
void normal_streams(AudioAnalyzer& analyzer)
{
    std::vector<float> pcm(std::size_t{513} * 2, 0.1f);
    EncodeOptions options;
    options.codebooks = 2;
    options.entropy_batch = 7; // More lanes than frames.
    std::vector<std::size_t> encoded, decoded;
    auto bytes = analyzer.encode(interleaved_pcm(pcm, 2), options,
                                 {{},
                                  [&](const EntropyPosition& p, std::size_t)
                                  {
                                      encoded.push_back(p.token_offset);
                                  }});
    auto result = analyzer.decode(bytes, {{},
                                          [&](const EntropyPosition& p, std::size_t)
                                          {
                                              decoded.push_back(p.token_offset);
                                          }});
    std::vector<std::size_t> expected;
    visit_entropy_positions(parse_header(bytes),
                            [&](const EntropyPosition& p)
                            {
                                expected.push_back(p.token_offset);
                            });
    check(encoded == expected && decoded == expected, "entropy observer order changed");
    check(result.channels == 2 && result.samples.size() == 2048, "normal decode shape changed");
    auto corrupted = bytes;
    corrupted.back() ^= 1;
    rejects(
        [&]
        {
            analyzer.decode(corrupted);
        },
        "accepted invalid normal CRC");
    rejects(
        [&]
        {
            analyzer.encode(interleaved_pcm(pcm, 2), options,
                            {{},
                             [](const EntropyPosition&, std::size_t)
                             {
                                 throw Error("observer failure");
                             }});
        },
        "lost observer exception");
    check(analyzer.decode(bytes).samples == result.samples, "normal failure prevented reuse");
    options.entropy_batch = 0;
    rejects(
        [&]
        {
            analyzer.encode(interleaved_pcm(pcm, 2), options);
        },
        "accepted explicit zero batch");
}
#endif
} // namespace

int main(int argc, char** argv)
{
    try
    {
        check(argc == 2 || argc == 3, "expected model directory and optional --metal");
        const bool metal = argc == 3 && std::string_view(argv[2]) == "--metal";
        execution_budgets();
        views();
        const ModelDirectory directory(argv[1]);
        std::unique_ptr<AudioAnalyzer> analyzer;
        if (metal)
        {
            analyzer = std::make_unique<MetalAnalyzer>(directory);
        }
        else
        {
            analyzer = std::make_unique<AppleArmAnalyzer>(directory, Limits{}, 2);
        }
        if (!metal)
        {
            thread_budgets(directory);
            session_reuse(directory);
        }
        layouts(*analyzer);
        groups_and_empty(*analyzer);
        failures(*analyzer, directory, metal);
#if TSAC_TEST_TRANSFORMER
        normal_streams(*analyzer);
#endif
        std::cout << "Audio analyzer contract passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
