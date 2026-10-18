// This consumer target receives only include/, never the private src/ tree.
#include "tsac/apple_arm_analyzer.hpp"
#include "tsac/audio_analyzer.hpp"
#include "tsac/metal_analyzer.hpp"
#include "tsac/model_directory.hpp"
#include "tsac/pcm.hpp"
#include "tsac/scalar_analyzer.hpp"
#include "tsac/types.hpp"

#include <array>
#include <iostream>
#include <memory>
#include <type_traits>

static_assert(std::is_base_of_v<tsac::AudioAnalyzer, tsac::AppleArmAnalyzer>);
static_assert(std::is_base_of_v<tsac::AudioAnalyzer, tsac::MetalAnalyzer>);
static_assert(std::is_base_of_v<tsac::AudioAnalyzer, tsac::ScalarAnalyzer>);
static_assert(std::is_copy_constructible_v<tsac::ModelDirectory>);

int main(int argc, char** argv)
{
    try
    {
        if (argc != 2)
        {
            throw tsac::Error("expected model directory");
        }
        tsac::ModelDirectory directory(argv[1]);
        std::unique_ptr<tsac::AudioAnalyzer> analyzer =
#if TSAC_TEST_SCALAR
            std::make_unique<tsac::ScalarAnalyzer>(directory);
#else
            std::make_unique<tsac::AppleArmAnalyzer>(directory, tsac::Limits{}, 1);
#endif
        const std::array<float, 513> samples{};
        const auto pcm = tsac::interleaved_pcm(samples, 1);
        tsac::EncodeOptions options;
        options.method = tsac::CodingMethod::fast;
        options.codebooks = 1;
        const auto bytes = analyzer->encode(pcm, options);
        const auto decoded = analyzer->decode(bytes);
        const auto channel = tsac::channel_view(decoded.view(), 0);
        if (decoded.channels != 1 || channel.extent(0) != 1024 ||
            tsac::frame_view(decoded.view(), 512, 512).extent(1) != 512)
        {
            throw tsac::Error("public API recording contract failed");
        }
#if TSAC_TEST_SCALAR && TSAC_TEST_TRANSFORMER
        options.method = tsac::CodingMethod::normal;
        options.entropy_batch = 1;
        std::size_t encoded = 0, recovered = 0;
        const auto normal = analyzer->encode(pcm, options,
                                             {{},
                                              [&](const auto&, std::size_t)
                                              {
                                                  ++encoded;
                                              }});
        const auto restored = analyzer->decode(normal, {{},
                                                        [&](const auto&, std::size_t)
                                                        {
                                                            ++recovered;
                                                        }});
        if (restored.samples != decoded.samples || encoded != 2 || recovered != encoded)
        {
            throw tsac::Error("scalar normal entropy recording contract failed");
        }
#endif
        std::cout << "Public-only consumer passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
