#pragma once
#include "tsac/audio_analyzer.hpp"

#include "common/numeric.hpp"
#include "common/signal.hpp"
#include "format/entropy.hpp"
#include "models/types.hpp"

namespace tsac::detail
{
EntropyObserver completed_symbols(const EntropyObserver& observer);
ModelVariant audio_variant(const Header& header);
Header audio_header(PcmView audio, const EncodeOptions& options, const Limits& limits);
Signal padded_audio(PcmView audio, const Header& header, const Limits& limits);
PcmAudio allocate_pcm(const Header& header, const Limits& limits);
void require_probability_assets(const Header& header);
std::size_t audio_chunk_frames(std::size_t target, const Limits& limits);

struct AudioChunk
{
    std::size_t begin, count, left, right;
    Header header(const Header& recording) const;
};

AudioChunk audio_chunk(std::size_t begin, std::size_t frames, std::size_t size,
                       std::size_t left_context, std::size_t right_context);
void store_tokens(std::span<std::uint16_t> destination, std::span<const std::uint16_t> source,
                  const Header& header, std::size_t group, const AudioChunk& chunk);
std::vector<std::uint16_t> chunk_tokens(std::span<const std::uint16_t> tokens, const Header& header,
                                        std::size_t group, const AudioChunk& chunk);
void store_pcm(PcmAudio& output, const Signal& evaluated, const Header& header, std::size_t group,
               const AudioChunk& chunk);

// Shared format geometry only. The concrete operators own weights, kernels,
// numerical behavior and execution resources; no execution choice enters here.
template <class Analysis, class Quantizer>
std::vector<std::uint16_t> analyze_audio(const Signal& audio, const Header& header,
                                         const Analysis& analysis, const Quantizer& quantizer,
                                         std::size_t chunk_size, const Limits& limits)
{
    std::vector<std::uint16_t> tokens(header.symbol_count());
    for (std::size_t group = 0; group < header.groups; ++group)
    {
        for (std::size_t begin = 0; begin < header.frames;)
        {
            const auto chunk = audio_chunk(begin, header.frames, chunk_size, 8, 9);
            const SignalShape shape{header.model_channels, (chunk.right - chunk.left) * 512, 1};
            const auto offset = (group * audio.shape.frames + chunk.left * 512) * shape.channels;
            const auto latent = analysis.apply(
                std::span(audio.values).subspan(offset, shape.elements(limits)), shape, limits);
            auto central_header = chunk.header(header);
            central_header.frames = chunk.count;
            const auto central =
                std::span(latent.values).subspan((begin - chunk.left) * 1024, chunk.count * 1024);
            const auto selected = quantizer.encode(central_header, central, limits);
            store_tokens(tokens, selected, header, group, chunk);
            begin += chunk.count;
        }
    }
    return tokens;
}

template <class Quantizer, class Synthesis>
PcmAudio synthesize_audio(std::span<const std::uint16_t> tokens, const Header& header,
                          const Quantizer& quantizer, const Synthesis& synthesis,
                          std::size_t chunk_size, const Limits& limits)
{
    auto audio = allocate_pcm(header, limits);
    for (std::size_t group = 0; group < header.groups; ++group)
    {
        for (std::size_t begin = 0; begin < header.frames;)
        {
            const auto chunk = audio_chunk(begin, header.frames, chunk_size, 11, 11);
            const auto selected = chunk_tokens(tokens, header, group, chunk);
            const auto latent = quantizer.decode(chunk.header(header), selected, limits);
            const auto evaluated =
                synthesis.apply(latent, {1024, chunk.right - chunk.left, 1}, limits);
            store_pcm(audio, evaluated, header, group, chunk);
            begin += chunk.count;
        }
    }
    return audio;
}
} // namespace tsac::detail
