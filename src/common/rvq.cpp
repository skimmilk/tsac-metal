#include "common/rvq.hpp"

#include "common/numeric.hpp"
#include "models/contract.hpp"
#include "weights/preparation.hpp"

#include <array>
#include <limits>
#include <utility>

namespace tsac
{
RvqCodebook::RvqCodebook(std::span<const float> codebook, std::span<const float> v,
                         std::span<const float> g, std::span<const float> bias,
                         const Limits& limits)
{
    detail::require(codebook.size() == vector_width * entries &&
                        v.size() == vector_width * latent_width && g.size() == latent_width &&
                        bias.size() == latent_width,
                    "RVQ codebook shape mismatch");
    detail::require(codebook.size() <= limits.tensor_elements, "codebook exceeds element limit");
    detail::require_finite(codebook, "nonfinite RVQ value");
    detail::require_finite(bias, "nonfinite RVQ value");
    weights_ = fold_weight_norm(v, g, vector_width, limits);
    codebook_.assign(codebook.begin(), codebook.end());
    bias_.assign(bias.begin(), bias.end());
}

void RvqCodebook::project(std::uint16_t index, std::span<float> output) const
{
    weights::project_codebook(index, codebook_, weights_, bias_, output, fused_);
}

RvqCodebook RvqCodebook::from_dac_weights(std::span<const float> codebook, std::span<const float> v,
                                          std::span<const float> g, std::span<const float> bias,
                                          const Limits& limits)
{
    auto result = RvqCodebook(codebook, v, g, bias, limits);
    result.weights_ = fold_dac_weight_norm(v, g, vector_width, limits);
    result.fused_ = true;
    return result;
}

RvqCodebook RvqCodebook::from_prepared(const TensorView<float>& codebook,
                                       const FloatProjectionWeights& projection,
                                       const Limits& limits)
{
    detail::require(vector_width * entries <= limits.tensor_elements,
                    "codebook exceeds element limit");
    RvqCodebook result;
    result.codebook_ = weights::Array<float>(codebook);
    result.weights_ = weights::Array<float>(projection.values);
    result.bias_ = weights::Array<float>(projection.bias);
    detail::require(result.codebook_.size() == 8192 && result.weights_.size() == 8192 &&
                        result.bias_.size() == 1024,
                    "invalid prepared codebook shape");
    result.fused_ = true;
    return result;
}

RvqDecoder RvqDecoder::from_model(RvqDecoderModel model, const Limits& limits)
{
    const auto codebooks = model.codebooks();
    RvqDecoder decoder;
    decoder.model_channels_ = model_contract::channels(model.variant());
    decoder.codebooks_.reserve(codebooks);
    for (unsigned book = 0; book < codebooks; ++book)
    {
        auto book_weights =
            RvqCodebook::from_prepared(model.codebook(book), model.output_projection(book), limits);
        decoder.codebooks_.push_back(std::move(book_weights));
    }
    return decoder;
}

std::vector<float> RvqDecoder::decode(const Header& header, std::span<const std::uint16_t> indices,
                                      const Limits& limits) const
{
    validate_header(header, limits);
    detail::require(header.model_channels == model_channels_,
                    "stream and DAC model channels do not match");
    detail::require(header.codebooks == codebooks_.size(),
                    "stream and prepared RVQ codebooks do not match");
    detail::require(indices.size() == header.symbol_count(), "symbol count does not match header");
    const auto count =
        checked_mul(checked_mul(header.frames, header.groups), RvqCodebook::latent_width);
    detail::require(count <= limits.tensor_elements, "RVQ latent exceeds element limit");
    for (auto index : indices)
    {
        detail::require(index < RvqCodebook::entries, "index outside 10-bit alphabet");
    }
    if (header.method == 1)
    {
        verify_token_crc(header, indices);
    }
    std::vector<float> latent(count, 0);
    if (!count)
    {
        return latent;
    }
    std::array<float, RvqCodebook::latent_width> contribution;
    for (std::size_t group = 0; group < header.groups; ++group)
    {
        for (std::size_t frame = 0; frame < header.frames; ++frame)
        {
            const auto token_offset = (frame * header.groups + group) * header.codebooks;
            const auto output_offset = (group * header.frames + frame) * RvqCodebook::latent_width;
            for (std::size_t book = 0; book < codebooks_.size(); ++book)
            {
                codebooks_[book].project(indices[token_offset + book], contribution);
                for (std::size_t j = 0; j < contribution.size(); ++j)
                {
                    latent[output_offset + j] += contribution[j];
                }
            }
        }
    }
    detail::require_finite(latent, "nonfinite RVQ value");
    return latent;
}
} // namespace tsac

namespace tsac
{
// Strict comparison preserves lowest-index ties; the eight FMAs stay ordered.
std::uint16_t nearest_rvq_index(std::span<const float, 8> input, std::span<const float> vectors)
{
    std::array<float, 8> query;
    weights::normalize_search(input, query);
    float best = -std::numeric_limits<float>::infinity();
    std::uint16_t selected = 0;
    for (std::size_t index = 0; index < RvqCodebook::entries; ++index)
    {
        float similarity = 0;
        for (std::size_t i = 0; i < 8; ++i)
        {
            similarity = std::fma(vectors[index * 8 + i], query[i], similarity);
        }
        if (similarity > best)
        {
            best = similarity;
            selected = static_cast<std::uint16_t>(index);
        }
    }
    return selected;
}
} // namespace tsac
