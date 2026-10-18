#include "codec.hpp"

#include "tsac/apple_arm_analyzer.hpp"
#include "tsac/metal_analyzer.hpp"

#include "apple-arm/execution.hpp"
#include "apple-arm/pipeline.hpp"
#include "audio.hpp"
#include "common/audio_pipeline.hpp"
#include "common/internal.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mach-o/dyld.h>
#include <sys/resource.h>

namespace tsac
{
std::filesystem::path executable_directory()
{
    std::uint32_t length = 0;
    _NSGetExecutablePath(nullptr, &length);
    detail::require(length > 0, "cannot locate executable directory");
    std::vector<char> executable(length);
    detail::require(_NSGetExecutablePath(executable.data(), &length) == 0,
                    "cannot locate executable directory");
    return std::filesystem::canonical(executable.data()).parent_path();
}

namespace
{
ModelVariant model_variant(const Header& header)
{
    detail::require(header.model_channels == 1 || header.model_channels == 2,
                    "unsupported model channel count");
    return header.model_channels == 1 ? ModelVariant::mono : ModelVariant::stereo;
}

class EntropyProgress
{
public:
    EntropyProgress(const Header& header, unsigned verbosity)
        : header_(header), verbosity_(verbosity), bits_(header.codebooks)
    {
        if (verbosity_)
        {
            std::cout << "batch_size=" << header.batch << " bs=" << unsigned(header.groups)
                      << " n_cb=" << unsigned(header.codebooks) << " block_len="
                      << (header.frames / header.batch + (header.frames % header.batch != 0))
                      << '\n';
            if (header.symbol_count() == 0)
            {
                std::cout << '\n';
            }
        }
    }

    EntropyObserver observer()
    {
        if (!verbosity_)
        {
            return {};
        }
        return {[this](const EntropyPosition& position)
                {
                    const auto steps =
                        header_.frames / header_.batch + (header_.frames % header_.batch != 0);
                    const auto percent = static_cast<int>((position.step + 1) * 100 / steps);
                    if (percent != percent_)
                    {
                        std::cout << std::setw(3) << percent << "%\r" << std::flush;
                        percent_ = percent;
                    }
                },
                [this](const EntropyPosition& position, std::size_t bits)
                {
                    bits_[position.codebook] = checked_add(bits_[position.codebook], bits);
                    if (++committed_ == header_.symbol_count())
                    {
                        std::cout << '\n';
                    }
                }};
    }

    void averages() const
    {
        if (verbosity_ < 2)
        {
            return;
        }
        std::cout << "CB AVG_BITS\n";
        const auto count = header_.frames * header_.groups;
        for (std::size_t book = 0; book < bits_.size(); ++book)
        {
            std::cout << std::setw(2) << book << ' ' << std::fixed << std::setprecision(3)
                      << std::setw(7) << (count ? double(bits_[book]) / double(count) : 0.0)
                      << '\n';
        }
    }

private:
    Header header_;
    unsigned verbosity_;
    std::vector<std::size_t> bits_;
    std::size_t committed_ = 0;
    int percent_ = -1;
};

std::unique_ptr<AudioAnalyzer> make_analyzer(const CliOptions& options, const Limits& limits)
{
    ModelDirectory directory(executable_directory(), limits);
    if (options.metal)
    {
        return std::make_unique<MetalAnalyzer>(std::move(directory), limits);
    }
    return std::make_unique<AppleArmAnalyzer>(std::move(directory), limits,
                                              options.threads > 0
                                                  ? static_cast<unsigned>(options.threads)
                                                  : apple_arm::physical_threads());
}

EncodeOptions encode_options(const CliOptions& options)
{
    detail::require(options.quality >= 0, "invalid number of codebooks");
    EncodeOptions result;
    result.method = options.fast ? CodingMethod::fast : CodingMethod::normal;
    result.codebooks = static_cast<unsigned>(options.quality);
    result.separate_channels = options.separate;
    if (!options.fast && options.batch)
    {
        detail::require(options.batch > 0 && options.batch <= 65535, "invalid entropy batch size");
        result.entropy_batch = static_cast<std::uint16_t>(options.batch);
    }
    return result;
}

void stream_statistics(const Header& header, std::size_t bytes, CodecStatistics& statistics)
{
    statistics.frames = header.frames;
    statistics.channels = header.channels();
    statistics.stream_bytes = bytes;
}

Bytes compress_with_analyzer(AudioAnalyzer& analyzer, const CliOptions& options,
                             CodecStatistics& statistics, const Limits& limits)
{
    const auto input = read_audio(options.input, options.channels, limits);
    const auto pcm = interleaved_pcm(input.values, input.shape.channels);
    const auto encoding = encode_options(options);
    const auto header = detail::audio_header(pcm, encoding, limits);
    EntropyProgress progress(header, options.fast ? 0 : options.verbose);
    auto stream = analyzer.encode(pcm, encoding, progress.observer());
    progress.averages();
    stream_statistics(header, stream.size(), statistics);
    return stream;
}

Bytes decompress_with_analyzer(AudioAnalyzer& analyzer, ByteView stream, const CliOptions& options,
                               CodecStatistics& statistics, const Limits& limits)
{
    const auto header = parse_header(stream, limits);
    EntropyProgress progress(header, header.method == 1 ? options.verbose : 0);
    const auto audio = analyzer.decode(stream, progress.observer());
    stream_statistics(header, stream.size(), statistics);
    return pcm16_wave(audio.samples, {audio.channels, audio.samples.size() / audio.channels, 1},
                      limits);
}

double peak_memory_gb()
{
    rusage usage{};
    return getrusage(RUSAGE_SELF, &usage) == 0 ? double(usage.ru_maxrss) / 1e9 : 0;
}
} // namespace

std::vector<std::uint16_t> decode_tokens(ByteView stream, const Limits& limits)
{
    const auto header = parse_header(stream, limits);
    if (header.method == 0)
    {
        return unpack_fast(stream, limits);
    }
    detail::require_probability_assets(header);
    auto transformer = apple_arm::CpuTransformer::from_model(
        TransformerModel(ModelDirectory(executable_directory(), limits), model_variant(header),
                         header.codebooks),
        limits);
    TransformerPredictor predictor(transformer, limits);
    return unpack_normal(stream, predictor, limits);
}

double CodecStatistics::bitrate_kbps() const
{
    return frames ? double(stream_bytes) * 8.0 * 44100.0 / (double(frames) * 512.0 * 1000.0) : 0;
}

Bytes compress_audio(const CliOptions& options, CodecStatistics& statistics, const Limits& limits)
{
    auto analyzer = make_analyzer(options, limits);
    return compress_with_analyzer(*analyzer, options, statistics, limits);
}

Bytes decompress_audio(ByteView stream, const CliOptions& options, CodecStatistics& statistics,
                       const Limits& limits)
{
    auto analyzer = make_analyzer(options, limits);
    return decompress_with_analyzer(*analyzer, stream, options, statistics, limits);
}

CodecStatistics run_codec(const CliOptions& options, const Limits& limits)
{
    const auto start = std::chrono::steady_clock::now();
    auto analyzer = make_analyzer(options, limits);
    CodecStatistics statistics;
    Bytes output;
    if (options.command == 'c' || options.command == 't')
    {
        output = compress_with_analyzer(*analyzer, options, statistics, limits);
        if (options.verbose)
        {
            std::cout << std::fixed << std::setprecision(2)
                      << "bitrate=" << statistics.bitrate_kbps()
                      << " kb/s, max_memory=" << peak_memory_gb() << " GB\n";
        }
        if (options.command == 't')
        {
            output = decompress_with_analyzer(*analyzer, output, options, statistics, limits);
        }
    }
    else if (options.command == 'd')
    {
        output = decompress_with_analyzer(*analyzer, read_file(options.input, limits), options,
                                          statistics, limits);
    }
    else
    {
        throw Error("invalid codec command");
    }
    write_file(options.output, output);
    statistics.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    statistics.peak_memory_gb = peak_memory_gb();
    return statistics;
}
} // namespace tsac
