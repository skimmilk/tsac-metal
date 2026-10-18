#include "models/contract.hpp"

namespace tsac::model_contract
{
namespace
{
void require(bool valid, const char* message)
{
    if (!valid)
    {
        throw Error(message);
    }
}

std::size_t aligned(std::size_t n, std::size_t a)
{
    return checked_add(n, a - 1) / a * a;
}

void residual_indices(unsigned stage, unsigned unit)
{
    require(stage < stages && unit < residual_units, "invalid DAC stage or residual unit");
}
} // namespace

unsigned channels(ModelVariant variant)
{
    require(variant == ModelVariant::mono || variant == ModelVariant::stereo,
            "invalid model variant");
    return variant == ModelVariant::mono ? 1u : 2u;
}

unsigned codebooks(ModelVariant variant)
{
    return channels(variant) == 1 ? 9u : 12u;
}

unsigned layers(TransformerStack stack)
{
    require(stack == TransformerStack::temporal || stack == TransformerStack::codebook,
            "invalid Transformer stack");
    return stack == TransformerStack::temporal ? temporal_layers : codebook_layers;
}

void validate_rvq_input_layout(RvqInputLayout layout)
{
    require(layout == RvqInputLayout::input_major || layout == RvqInputLayout::output_major,
            "invalid RVQ input layout");
}

TensorShape dac_shape(bool synthesis, ModelVariant variant, unsigned convolution)
{
    const auto c = channels(variant);
    require(convolution < dac_convolutions, "invalid DAC convolution");
    if (!convolution)
    {
        return synthesis ? TensorShape{{1024, 7, 1536}} : TensorShape{{c, 7, 64}};
    }
    if (convolution == 29)
    {
        return synthesis ? TensorShape{{96, 7, c}} : TensorShape{{1024, 3, 1024}};
    }
    const auto stage = (convolution - 1) / 7, within = (convolution - 1) % 7;
    const std::size_t width = synthesis ? (1536u >> stage) : (64u << stage);
    const auto stride = synthesis ? std::array<unsigned, 4>{8, 8, 4, 2}[stage]
                                  : std::array<unsigned, 4>{2, 4, 8, 8}[stage];
    if (synthesis && within == 0)
    {
        return {{width / 2, std::size_t{2} * stride, width}};
    }
    if (!synthesis && within == 6)
    {
        return {{width, std::size_t{2} * stride, width * 2}};
    }
    const auto residual = synthesis ? within - 1 : within;
    const auto w = synthesis ? width / 2 : width;
    return {{w, residual % 2 == 0 ? 7u : 1u, w}};
}

TensorEncoding dac_encoding(bool synthesis, unsigned convolution)
{
    require(convolution < dac_convolutions, "invalid DAC convolution");
    return ((!synthesis && convolution == 0) || (synthesis && convolution >= 22))
               ? TensorEncoding::float32
               : TensorEncoding::bf8;
}

TensorShape projection_shape(unsigned projection)
{
    require(projection < 63, "invalid Transformer projection");
    if (projection == 60 || projection == 61)
    {
        return {{1024, 512}};
    }
    if (projection == 62)
    {
        return {{512, 1024}};
    }
    switch (projection % 4)
    {
    case 0:
        return {{512, 1536}};
    case 1:
        return {{512, 512}};
    case 2:
        return {{512, 2048}};
    default:
        return {{2048, 512}};
    }
}

TensorShape activation_shape(bool synthesis, unsigned index)
{
    require(index < 29, "invalid DAC activation");
    const auto width = index == 28 ? (synthesis ? 96u : 1024u)
                       : synthesis ? (1536u >> (index / 7)) / (index % 7 == 0 ? 1u : 2u)
                                   : (64u << (index / 7));
    return {{width, 1, 1}};
}

TensorShape rvq_input_shape(RvqInputLayout layout)
{
    validate_rvq_input_layout(layout);
    return layout != RvqInputLayout::output_major ? TensorShape{{latent_width, codebook_width}}
                                                  : TensorShape{{codebook_width, latent_width}};
}

TensorShape positions_shape(unsigned books)
{
    require(books > 0 && books <= 12, "invalid position book count");
    return {{transformer_width, books}};
}

TensorShape feedback_shape(unsigned books)
{
    require(books > 0 && books <= 12, "invalid feedback book count");
    return {{latent_width, alphabet, books}};
}

std::size_t encoded_bytes(TensorShape shape, TensorEncoding encoding)
{
    const auto elements = shape.elements();
    switch (encoding)
    {
    case TensorEncoding::float32:
        return checked_mul(elements, 4);
    case TensorEncoding::bf16:
        return checked_mul(elements, 2);
    case TensorEncoding::int8:
        return elements;
    case TensorEncoding::bf8:
    {
        require(shape.rank() == 3 && shape.axes[0] % 32 == 0, "invalid BF8 contract shape");
        const auto scales =
            checked_mul(checked_mul(shape.axes[0] / 32, aligned(shape.axes[1], 8)), shape.axes[2]);
        return checked_add(aligned(elements, 256), aligned(checked_mul(scales, 2), 256));
    }
    }
    throw Error("invalid tensor encoding");
}

unsigned residual_convolution(bool synthesis, unsigned stage, unsigned unit,
                              ResidualConvolution which)
{
    residual_indices(stage, unit);
    require(which == ResidualConvolution::dilated || which == ResidualConvolution::pointwise,
            "invalid residual convolution");
    return 1 + stage * 7 + (synthesis ? 1u : 0u) + unit * 2 +
           (which == ResidualConvolution::pointwise ? 1u : 0u);
}

unsigned residual_activation(bool synthesis, unsigned stage, unsigned unit,
                             ResidualActivation which)
{
    residual_indices(stage, unit);
    require(which == ResidualActivation::before_dilated ||
                which == ResidualActivation::before_pointwise,
            "invalid residual activation");
    return stage * 7 + (synthesis ? 1u : 0u) + unit * 2 +
           (which == ResidualActivation::before_pointwise ? 1u : 0u);
}

unsigned resample_convolution(bool synthesis, unsigned stage)
{
    require(stage < stages, "invalid DAC stage");
    return 1 + stage * 7 + (synthesis ? 0u : 6u);
}

unsigned resample_activation(bool synthesis, unsigned stage)
{
    require(stage < stages, "invalid DAC stage");
    return stage * 7 + (synthesis ? 0u : 6u);
}

unsigned projection_index(TransformerStack stack, unsigned layer, unsigned operation)
{
    require(layer < layers(stack) && operation < 4, "invalid Transformer layer or projection");
    return (stack == TransformerStack::temporal ? 0 : 48) + layer * 4 + operation;
}

unsigned normalization_index(TransformerStack stack, unsigned layer, bool mlp)
{
    require(layer < layers(stack), "invalid Transformer layer");
    return (stack == TransformerStack::temporal ? 0 : 25) + layer * 2 + (mlp ? 1 : 0);
}

unsigned final_normalization_index(TransformerStack stack)
{
    (void)layers(stack);
    return stack == TransformerStack::temporal ? 24 : 31;
}
} // namespace tsac::model_contract
