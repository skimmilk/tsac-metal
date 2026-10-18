#include "tsac/audio_analyzer.hpp"

#include "common/audio_pipeline.hpp"

#include <array>
#include <cmath>

namespace tsac
{
PcmView interleaved_pcm(std::span<const float> samples, std::size_t channels)
{
    detail::require(channels > 0 && samples.size() % channels == 0,
                    "interleaved PCM channel/sample mismatch");
    const PcmView::extents_type shape(channels, samples.size() / channels);
    return PcmView(samples.data(),
                   PcmView::mapping_type(shape, std::array<std::size_t, 2>{1, channels}));
}

PcmView PcmAudio::view() const&
{
    return interleaved_pcm(samples, channels);
}

PcmChannelView channel_view(PcmView audio, std::size_t channel)
{
    detail::require(channel < audio.extent(0), "PCM channel out of range");
    const auto* data = audio.data_handle();
    if (audio.extent(1))
    {
        data += channel * audio.stride(0);
    }
    return PcmChannelView(
        data, PcmChannelView::mapping_type(PcmChannelView::extents_type(audio.extent(1)),
                                           std::array<std::size_t, 1>{audio.stride(1)}));
}

PcmView frame_view(PcmView audio, std::size_t first, std::size_t count)
{
    detail::require(first <= audio.extent(1) && count <= audio.extent(1) - first,
                    "PCM frame range out of bounds");
    const auto* data = audio.data_handle();
    if (count)
    {
        data += first * audio.stride(1);
    }
    return PcmView(
        data, PcmView::mapping_type(PcmView::extents_type(audio.extent(0), count),
                                    std::array<std::size_t, 2>{audio.stride(0), audio.stride(1)}));
}
} // namespace tsac

namespace tsac::detail
{
EntropyObserver completed_symbols(const EntropyObserver& observer)
{
    if (!observer.before_symbol && !observer.after_symbol)
    {
        return {};
    }
    return {{},
            [&observer](const EntropyPosition& position, std::size_t bits)
            {
                if (observer.before_symbol)
                {
                    observer.before_symbol(position);
                }
                if (observer.after_symbol)
                {
                    observer.after_symbol(position, bits);
                }
            }};
}

ModelVariant audio_variant(const Header& header)
{
    return header.model_channels == 1 ? ModelVariant::mono : ModelVariant::stereo;
}

Header audio_header(PcmView audio, const EncodeOptions& options, const Limits& limits)
{
    const auto channels = audio.extent(0), samples = audio.extent(1);
    require(channels > 0 && channels <= 127, "invalid PCM channel count");
    require(audio.stride(0) > 0 && audio.stride(1) > 0, "invalid PCM strides");
    require(samples == 0 || audio.data_handle() != nullptr, "null PCM data");
    require(checked_mul(samples, channels) <= limits.tensor_elements, "PCM exceeds element limit");
    require(options.method == CodingMethod::fast || options.method == CodingMethod::normal,
            "invalid coding method");
    Header header;
    header.method = options.method == CodingMethod::fast ? 0 : 1;
    header.model_channels = channels == 2 && !options.separate_channels ? 2 : 1;
    header.groups = static_cast<std::uint8_t>(channels / header.model_channels);
    const unsigned maximum = header.model_channels == 2 ? 12 : 9;
    require(options.codebooks <= maximum, "invalid number of codebooks for audio model");
    header.codebooks = static_cast<std::uint8_t>(options.codebooks ? options.codebooks : maximum);
    header.frames = checked_add(samples, 511) / 512;
    require(!options.entropy_batch || *options.entropy_batch > 0, "zero entropy batch size");
    require(header.method == 1 || !options.entropy_batch, "entropy batch requires normal coding");
    if (header.method == 1)
    {
        header.batch = select_entropy_batch(header.frames, options.entropy_batch.value_or(0));
    }
    validate_header(header, limits);
    require(checked_mul(header.samples_per_channel(), channels) <= limits.tensor_elements,
            "padded PCM exceeds element limit");
    require_probability_assets(header);
    return header;
}

void require_probability_assets(const Header& header)
{
#if defined(TSAC_ENABLE_TRANSFORMER) && !TSAC_ENABLE_TRANSFORMER
    require(header.method == 0, "Transformer assets were disabled; use fast mode (-f)");
#else
    (void)header;
#endif
}

Signal padded_audio(PcmView audio, const Header& header, const Limits& limits)
{
    Signal result{{header.model_channels, header.samples_per_channel(), header.groups}, {}};
    const auto count =
        checked_mul(checked_mul(result.shape.channels, result.shape.frames), result.shape.groups);
    require(count <= limits.tensor_elements, "padded PCM exceeds element limit");
    result.values.resize(count);
    for (std::size_t group = 0; group < header.groups; ++group)
    {
        for (std::size_t frame = 0; frame < audio.extent(1); ++frame)
        {
            for (std::size_t channel = 0; channel < header.model_channels; ++channel)
            {
                const auto value = audio[group * header.model_channels + channel, frame];
                require(std::isfinite(value), "nonfinite PCM sample");
                result.values[(group * result.shape.frames + frame) * header.model_channels +
                              channel] = value;
            }
        }
    }
    return result;
}

PcmAudio allocate_pcm(const Header& header, const Limits& limits)
{
    const auto count = checked_mul(header.channels(), header.samples_per_channel());
    require(count <= limits.tensor_elements, "output audio exceeds element limit");
    return {std::vector<float>(count), static_cast<unsigned>(header.channels())};
}

std::size_t audio_chunk_frames(std::size_t target, const Limits& limits)
{
    const auto available = limits.tensor_elements / (std::size_t{96} * 512);
    return std::min(target, available > 22 ? available - 22 : std::size_t{1});
}

AudioChunk audio_chunk(std::size_t begin, std::size_t frames, std::size_t size,
                       std::size_t left_context, std::size_t right_context)
{
    require(size > 0 && begin < frames, "invalid audio chunk");
    const auto count = std::min(size, frames - begin);
    const auto end = begin + count;
    return {begin, count, begin > left_context ? begin - left_context : 0,
            end + std::min(right_context, frames - end)};
}

Header AudioChunk::header(const Header& recording) const
{
    auto result = recording;
    result.method = 0; // Full normal-stream CRC is verified before chunk synthesis.
    result.groups = 1;
    result.frames = right - left;
    return result;
}

void store_tokens(std::span<std::uint16_t> destination, std::span<const std::uint16_t> source,
                  const Header& header, std::size_t group, const AudioChunk& chunk)
{
    for (std::size_t frame = 0; frame < chunk.count; ++frame)
    {
        std::copy_n(source.data() + frame * header.codebooks, header.codebooks,
                    destination.data() +
                        ((chunk.begin + frame) * header.groups + group) * header.codebooks);
    }
}

std::vector<std::uint16_t> chunk_tokens(std::span<const std::uint16_t> tokens, const Header& header,
                                        std::size_t group, const AudioChunk& chunk)
{
    std::vector<std::uint16_t> result((chunk.right - chunk.left) * header.codebooks);
    for (std::size_t frame = chunk.left; frame < chunk.right; ++frame)
    {
        std::copy_n(tokens.data() + (frame * header.groups + group) * header.codebooks,
                    header.codebooks, result.data() + (frame - chunk.left) * header.codebooks);
    }
    return result;
}

void store_pcm(PcmAudio& output, const Signal& evaluated, const Header& header, std::size_t group,
               const AudioChunk& chunk)
{
    for (std::size_t sample = 0; sample < chunk.count * 512; ++sample)
    {
        for (std::size_t channel = 0; channel < header.model_channels; ++channel)
        {
            output.samples[(chunk.begin * 512 + sample) * output.channels +
                           group * header.model_channels + channel] =
                evaluated
                    .values[((chunk.begin - chunk.left) * 512 + sample) * header.model_channels +
                            channel];
        }
    }
}
} // namespace tsac::detail
