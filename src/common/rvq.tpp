#include "common/numeric.hpp"
#include "common/rvq.hpp"
#include "models/contract.hpp"

#include <array>

namespace tsac
{
std::uint16_t nearest_rvq_index(std::span<const float, 8>, std::span<const float>);

template <class Kernels>
RvqEncoder<Kernels> RvqEncoder<Kernels>::from_model(RvqEncoderModel model, const Limits& limits,
                                                    Kernels kernels)
{
    const auto codebooks = model.codebooks();
    RvqEncoder encoder;
    encoder.model_channels_ = model_contract::channels(model.variant());
    detail::require(model.input_layout() != RvqInputLayout::output_major,
                    "CPU RVQ requires ordinary projection weights");
    encoder.codebooks_.reserve(codebooks);
    for (unsigned book = 0; book < codebooks; ++book)
    {
        auto output =
            RvqCodebook::from_prepared(model.codebook(book), model.output_projection(book), limits);
        auto normalized = weights::View(model.search_vectors(book));
        const auto projection = model.input_projection(book);
        auto input =
            Kernels::Conv1d::from_prepared({1024, 8, 1, 1, 0, 1}, weights::View(projection.values),
                                           weights::View(projection.bias), limits);
        kernels.bind(input);
        encoder.codebooks_.push_back(
            {std::move(input), std::move(output), weights::Array<float>(normalized)});
    }
    return encoder;
}

template <class Kernels>
std::vector<std::uint16_t> RvqEncoder<Kernels>::Quantizer::apply(std::span<float> residual,
                                                                 const Limits& limits) const
{
    const auto frames = residual.size() / 1024;
    const auto projected = input_projection.apply(residual, {1024, frames, 1}, limits);
    std::vector<std::uint16_t> tokens(frames);
    std::array<float, 1024> contribution;
    for (std::size_t frame = 0; frame < frames; ++frame)
    {
        tokens[frame] = nearest_rvq_index(
            std::span<const float, 8>(projected.values.data() + frame * 8, 8), normalized_vectors);
        output_projection.project(tokens[frame], contribution);
        for (std::size_t i = 0; i < 1024; ++i)
        {
            residual[frame * 1024 + i] -= contribution[i];
        }
    }
    detail::require_finite(residual, "nonfinite RVQ residual");
    return tokens;
}

template <class Kernels>
std::vector<std::uint16_t> RvqEncoder<Kernels>::encode(const Header& header,
                                                       std::span<const float> latent,
                                                       const Limits& limits) const
{
    validate_header(header, limits);
    detail::require(header.model_channels == model_channels_,
                    "stream and DAC model channels do not match");
    detail::require(header.codebooks == codebooks_.size(),
                    "stream and prepared RVQ codebooks do not match");
    detail::require(latent.size() ==
                        SignalShape{1024, header.frames, header.groups}.elements(limits),
                    "RVQ input shape mismatch");
    detail::require_finite(latent, "nonfinite RVQ input");
    std::vector<std::uint16_t> indices(header.symbol_count());
    // Quantizers couple codebooks only within one frame. Tile groups to bound scratch.
    for (std::size_t group = 0; group < header.groups; ++group)
    {
        for (std::size_t begin = 0; begin < header.frames;)
        {
            const auto count = std::min<std::size_t>(100, header.frames - begin);
            const auto tile = latent.subspan((group * header.frames + begin) * 1024, count * 1024);
            std::vector<float> residual(tile.begin(), tile.end());
            for (std::size_t book = 0; book < codebooks_.size(); ++book)
            {
                const auto tokens = codebooks_[book].apply(residual, limits);
                for (std::size_t frame = 0; frame < count; ++frame)
                {
                    indices[((begin + frame) * header.groups + group) * header.codebooks + book] =
                        tokens[frame];
                }
            }
            begin += count;
        }
    }
    return indices;
}

} // namespace tsac
