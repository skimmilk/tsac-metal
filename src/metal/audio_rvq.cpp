#include "audio_rvq.hpp"

#include "common/audio_pipeline.hpp"
#include "models/contract.hpp"

namespace tsac::metal
{
namespace
{
RvqBuffers output_buffers(const TensorView<float>& codebook,
                          const FloatProjectionWeights& projection, Buffer normalized,
                          const Limits& limits)
{
    detail::require(limits.tensor_elements >= 8192, "RVQ codebook exceeds element limit");
    return {Buffer::import_weights(weights::View(codebook)),
            Buffer::import_weights(weights::View(projection.values)),
            Buffer::import_weights(weights::View(projection.bias)), std::move(normalized)};
}

void validate_rvq(const Header& header, unsigned channels, std::size_t books, const Limits& limits)
{
    validate_header(header, limits);
    detail::require(header.model_channels == channels && header.codebooks == books,
                    "Metal RVQ stream/model mismatch");
    detail::require(checked_mul(checked_mul(header.frames, header.groups), 1024) <=
                        limits.tensor_elements,
                    "Metal RVQ latent exceeds element limit");
}
} // namespace

AudioRvqEncoder AudioRvqEncoder::from_model(RvqEncoderModel model, const Limits& limits)
{
    detail::require(model.input_layout() == RvqInputLayout::output_major,
                    "Metal RVQ requires Metal weights");
    AudioRvqEncoder result;
    result.channels_ = model_contract::channels(model.variant());
    for (unsigned book = 0; book < model.codebooks(); ++book)
    {
        const auto projection = model.input_projection(book);
        auto input =
            MetalConvolution::from_prepared({1024, 8, 1, 1, 0, 1}, weights::View(projection.values),
                                            weights::View(projection.bias));
        auto output = output_buffers(
            model.codebook(book), model.output_projection(book),
            Buffer::import_weights(weights::View(model.search_vectors(book))), limits);
        result.codebooks_.push_back({std::move(input), std::move(output)});
    }
    return result;
}

std::vector<Buffer> AudioRvqEncoder::record(DacGraph& graph, const Header& header,
                                            const DacTensor& latent, const Limits& limits) const
{
    validate_rvq(header, channels_, codebooks_.size(), limits);
    detail::require(latent.shape.channels == 1024 && latent.shape.frames == header.frames &&
                        latent.shape.groups == header.groups &&
                        latent.buffer.size() >= latent.shape.elements(limits) * sizeof(float),
                    "Metal RVQ input shape mismatch");
    auto residual = latent;
    std::vector<Buffer> selected;
    for (const auto& quantizer : codebooks_)
    {
        auto projected = graph.convolution(quantizer.input, residual, limits);
        auto tokens = graph.search(projected, quantizer.output, limits);
        graph.release(projected);
        auto next = graph.project(tokens, quantizer.output, residual, true, limits);
        graph.release(residual);
        residual = next;
        selected.push_back(tokens);
    }
    graph.release(residual);
    return selected;
}

std::vector<std::uint16_t> AudioRvqEncoder::encode(const Header& header,
                                                   std::span<const float> latent,
                                                   const Limits& limits) const
{
    validate_rvq(header, channels_, codebooks_.size(), limits);
    detail::require(latent.size() ==
                        SignalShape{1024, header.frames, header.groups}.elements(limits),
                    "Metal RVQ input shape mismatch");
    detail::require_finite(latent, "nonfinite Metal RVQ input");
    std::vector<std::uint16_t> result(header.symbol_count());
    DacGraph graph;
    for (std::size_t group = 0; group < header.groups; ++group)
    {
        for (std::size_t begin = 0; begin < header.frames;)
        {
            const auto chunk = detail::audio_chunk(begin, header.frames, 100, 0, 0);
            graph.begin();
            const auto input = graph.upload(
                latent.subspan((group * header.frames + begin) * 1024, chunk.count * 1024),
                {1024, chunk.count, 1}, limits);
            const auto selected = record(graph, chunk.header(header), input, limits);
            graph.finish();
            for (std::size_t book = 0; book < selected.size(); ++book)
            {
                const auto tokens = selected[book].mapped<std::uint16_t>();
                for (std::size_t frame = 0; frame < chunk.count; ++frame)
                {
                    result[((begin + frame) * header.groups + group) * header.codebooks + book] =
                        tokens[frame];
                }
            }
            begin += chunk.count;
        }
    }
    return result;
}

AudioRvqDecoder AudioRvqDecoder::from_model(RvqDecoderModel model, const Limits& limits)
{
    AudioRvqDecoder result;
    result.channels_ = model_contract::channels(model.variant());
    for (unsigned book = 0; book < model.codebooks(); ++book)
    {
        result.codebooks_.push_back(output_buffers(
            model.codebook(book), model.output_projection(book), Buffer::allocate(0), limits));
    }
    return result;
}

std::vector<float> AudioRvqDecoder::decode(const Header& header,
                                           std::span<const std::uint16_t> tokens,
                                           const Limits& limits) const
{
    validate_rvq(header, channels_, codebooks_.size(), limits);
    if (!header.frames)
    {
        detail::require(tokens.empty(), "Metal RVQ token count mismatch");
        if (header.method == 1)
        {
            verify_token_crc(header, tokens);
        }
        return {};
    }
    DacGraph graph;
    return graph.download(record(graph, header, tokens, limits), limits).values;
}

DacTensor AudioRvqDecoder::record(DacGraph& graph, const Header& header,
                                  std::span<const std::uint16_t> tokens, const Limits& limits) const
{
    validate_rvq(header, channels_, codebooks_.size(), limits);
    detail::require(tokens.size() == header.symbol_count(), "Metal RVQ token count mismatch");
    for (auto token : tokens)
    {
        detail::require(token < 1024, "Metal RVQ token exceeds alphabet");
    }
    if (header.method == 1)
    {
        verify_token_crc(header, tokens);
    }
    const SignalShape shape{1024, header.frames, header.groups};
    auto residual = graph.zeros(shape, limits);
    for (std::size_t book = 0; book < codebooks_.size(); ++book)
    {
        auto input = graph.shared_temporary(header.frames * header.groups * sizeof(std::uint16_t));
        auto* selected = reinterpret_cast<std::uint16_t*>(input.mapped_bytes().data());
        for (std::size_t group = 0; group < header.groups; ++group)
        {
            for (std::size_t frame = 0; frame < header.frames; ++frame)
            {
                selected[group * header.frames + frame] =
                    tokens[(frame * header.groups + group) * header.codebooks + book];
            }
        }
        auto next = graph.project(input, codebooks_[book], residual, false, limits);
        graph.release(residual);
        residual = next;
    }
    return residual;
}
} // namespace tsac::metal
