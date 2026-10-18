#include "audio.hpp"

#include "common/internal.hpp"
#include "common/numeric.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace tsac
{
Signal pcm16_input(ByteView bytes, unsigned channels, const Limits& limits)
{
    detail::require(channels > 0 && channels <= 127, "unsupported number of channels");
    const auto block = checked_mul(channels, 2);
    const auto frames = bytes.size() / block; // Original discards an incomplete interleaved frame.
    const auto padded = checked_mul(checked_add(frames, 511) / 512, 512);
    const auto count = checked_mul(padded, channels);
    detail::require(checked_mul(count, 2) <= limits.pcm_bytes && count <= limits.tensor_elements,
                    "input PCM exceeds limit");
    Signal signal{{channels, padded, 1}, std::vector<float>(count, 0)};
    for (std::size_t i = 0; i < frames * channels; ++i)
    {
        const unsigned value = unsigned(bytes[2 * i]) | (unsigned(bytes[2 * i + 1]) << 8);
        const int signed_value =
            value < 32768 ? static_cast<int>(value) : static_cast<int>(value) - 65536;
        signal.values[i] = static_cast<float>(signed_value) * (1.0f / 32768.0f);
    }
    return signal;
}

Signal wave_input(ByteView bytes, const Limits& limits)
{
    const auto tag = [&](std::size_t offset, const char* text)
    {
        const auto part = detail::slice(bytes, offset, 4);
        return std::equal(part.begin(), part.end(), text);
    };
    detail::require(tag(0, "RIFF") && tag(8, "WAVE"), "invalid wav file");
    unsigned channels = 0;
    for (std::size_t offset = 12; offset < bytes.size();)
    {
        const auto length = detail::le32(bytes, checked_add(offset, 4));
        const auto payload = checked_add(offset, 8);
        if (tag(offset, "fmt "))
        {
            const auto fmt = detail::slice(bytes, payload, length);
            detail::require(length >= 16 && fmt[0] == 1 && fmt[1] == 0 && fmt[14] == 16 &&
                                fmt[15] == 0,
                            "invalid wav file");
            channels = unsigned(fmt[2]) | (unsigned(fmt[3]) << 8);
            // The original native WAVE reader accepts 1..8, despite the raw-input limit of 127.
            detail::require(channels > 0 && channels <= 8, "unsupported number of channels");
            detail::require(detail::le32(fmt, 4) == 44100, "the sample rate must be 44100 Hz");
        }
        else if (tag(offset, "data"))
        {
            detail::require(channels > 0, "invalid wav file");
            // Retained probes confirm the original reads to EOF, ignoring the data/RIFF sizes.
            return pcm16_input(bytes.subspan(payload), channels, limits);
        }
        detail::slice(bytes, payload, length);
        offset = checked_add(payload, length);
    }
    throw Error("invalid wav file");
}

namespace
{
struct TemporaryAudio
{
    std::filesystem::path directory;

    TemporaryAudio()
    {
        auto pattern = (std::filesystem::temp_directory_path() / "tsac-audio-XXXXXX").string();
        if (!mkdtemp(pattern.data()))
        {
            throw Error("cannot create audio conversion directory");
        }
        directory = pattern;
    }

    ~TemporaryAudio()
    {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }
};
} // namespace

Signal read_audio(const std::filesystem::path& path, int requested_channels, const Limits& limits)
{
    detail::require(requested_channels >= 0 && requested_channels <= 127,
                    "unsupported number of channels");
    if (path.extension() == ".sw")
    {
        return pcm16_input(read_file(path, limits),
                           requested_channels ? static_cast<unsigned>(requested_channels) : 1,
                           limits);
    }
    if (path.extension() == ".wav" && requested_channels == 0)
    {
        const auto bytes = read_file(path, limits);
        try
        {
            return wave_input(bytes, limits);
        }
        catch (const Error& error)
        {
            throw Error(path.string() + ": " + error.what());
        }
    }
    TemporaryAudio temporary;
    const auto converted = temporary.directory / "converted.wav";
    // Absolute paths prevent a filename starting with '-' from becoming an FFmpeg option.
    std::vector<std::string> arguments{
        "ffmpeg",   "-i",        std::filesystem::absolute(path).string(),
        "-y",       "-loglevel", "error",
        "-ar",      "44100",     "-acodec",
        "pcm_s16le"};
    if (requested_channels)
    {
        arguments.insert(arguments.end(), {"-ac", std::to_string(requested_channels)});
    }
    arguments.push_back(converted.string());
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1);
    for (auto& argument : arguments)
    {
        argv.push_back(argument.data());
    }
    argv.push_back(nullptr);
    pid_t child;
    const int error = posix_spawnp(&child, "ffmpeg", nullptr, nullptr, argv.data(), environ);
    if (error)
    {
        throw Error("Error while trying to run FFmpeg: " + std::string(std::strerror(error)));
    }
    int status = 0;
    pid_t waited;
    do
    {
        waited = waitpid(child, &status, 0);
    } while (waited == -1 && errno == EINTR);
    if (waited == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
    {
        throw Error("Error while trying to run FFmpeg");
    }
    return wave_input(read_file(converted, limits), limits);
}

Signal group_audio(const Signal& input, bool separate, const Limits& limits)
{
    detail::require(input.shape.groups == 1 && input.shape.channels > 0 &&
                        input.shape.channels <= 127,
                    "invalid interleaved input shape");
    const auto count = checked_mul(input.shape.channels, input.shape.frames);
    detail::require(count == input.values.size() && count <= limits.tensor_elements,
                    "input audio shape mismatch");
    const auto channels = input.shape.channels == 2 && !separate ? std::size_t{2} : std::size_t{1};
    Signal output{{channels, input.shape.frames, input.shape.channels / channels},
                  std::vector<float>(count)};
    for (std::size_t group = 0; group < output.shape.groups; ++group)
    {
        for (std::size_t frame = 0; frame < input.shape.frames; ++frame)
        {
            for (std::size_t channel = 0; channel < channels; ++channel)
            {
                output.values[(group * input.shape.frames + frame) * channels + channel] =
                    input.values[frame * input.shape.channels + group * channels + channel];
            }
        }
    }
    return output;
}

Bytes pcm16_wave(std::span<const float> samples, SignalShape shape, const Limits& limits)
{
    const auto count = checked_mul(checked_mul(shape.channels, shape.frames), shape.groups);
    detail::require(shape.channels && shape.groups && count <= limits.tensor_elements &&
                        samples.size() == count,
                    "audio sample shape mismatch");
    const auto channels = checked_mul(shape.channels, shape.groups);
    const auto block = checked_mul(channels, 2);
    const auto rate = checked_mul(block, 44100);
    const auto bytes = checked_mul(samples.size(), 2);
    detail::require(bytes <= limits.pcm_bytes, "PCM exceeds byte limit");
    detail::require(block <= std::numeric_limits<std::uint16_t>::max() &&
                        rate <= std::numeric_limits<std::uint32_t>::max() &&
                        bytes <= std::numeric_limits<std::uint32_t>::max() - 36,
                    "audio exceeds RIFF/WAVE dimensions");
    detail::require_finite(samples, "nonfinite audio sample");
    Bytes output;
    output.reserve(checked_add(bytes, 44));
    const auto word = [&](std::size_t value, unsigned count)
    {
        for (unsigned i = 0; i < count; ++i)
        {
            output.push_back(static_cast<std::uint8_t>((value >> (8 * i)) & 255));
        }
    };
    const auto tag = [&](const char* value)
    {
        output.insert(output.end(), value, value + 4);
    };
    tag("RIFF");
    word(bytes + 36, 4);
    tag("WAVE");
    tag("fmt ");
    word(16, 4);
    word(1, 2);
    word(channels, 2);
    word(44100, 4);
    word(rate, 4);
    word(block, 2);
    word(16, 2);
    tag("data");
    word(bytes, 4);
    for (std::size_t frame = 0; frame < shape.frames; ++frame)
    {
        for (std::size_t group = 0; group < shape.groups; ++group)
        {
            for (std::size_t channel = 0; channel < shape.channels; ++channel)
            {
                const auto x = samples[channel + shape.channels * (frame + shape.frames * group)];
                // Clipping first avoids overflow for any finite float sample.
                const double scaled = std::clamp(double(x) * 32768.0, -32768.0, 32767.0);
                const double lower = std::floor(scaled);
                const auto integer = static_cast<std::int32_t>(lower);
                const auto fraction = scaled - lower;
                const auto rounded =
                    integer + (fraction > 0.5 || (fraction == 0.5 && integer % 2 != 0));
                word(static_cast<std::uint16_t>(rounded), 2);
            }
        }
    }
    return output;
}
} // namespace tsac
