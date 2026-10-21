#pragma once
#include "apple-arm/dsp.hpp"
#include "common/probability.hpp"

namespace tsac::apple_arm
{
void add_bf16(std::span<std::uint16_t>, std::span<const std::uint16_t>);
ProbabilityBf8l bf8l(std::span<const float>);
void bf8l(std::span<const float>, ProbabilityBf8l&);
std::vector<std::uint16_t>
attention_bf16(std::span<const std::uint16_t>, std::span<const std::uint16_t>,
               std::span<const std::uint16_t>, std::size_t heads = 4, std::size_t head_width = 128,
               float scale = 0.0883883461356163f, bool prefill = false, bool tiled_keys = false);

void attention_bf16_into(std::span<const std::uint16_t>, std::span<const std::uint16_t>,
                         std::span<const std::uint16_t>, AttentionWorkspace&, std::size_t,
                         std::size_t, float, bool, bool);

struct Kernels
{
    using Conv1d = apple_arm::Conv1d;
    const Execution* execution = nullptr;

    void bind(Conv1d& convolution) const
    {
        convolution.set_execution(execution);
    }

    Conv1d convolution(const ConvolutionWeights& weights, const Limits& limits, std::size_t stride,
                       std::size_t dilation, bool transposed) const
    {
        auto result =
            Conv1d::from_weights(weights, limits, stride, dilation, transposed, execution);
        bind(result);
        return result;
    }

    static void add(std::span<std::uint16_t> left, std::span<const std::uint16_t> right)
    {
        add_bf16(left, right);
    }

    static ProbabilityBf8l bf8l(std::span<const float> input)
    {
        return apple_arm::bf8l(input);
    }

    static void bf8l_into(std::span<const float> input, ProbabilityBf8l& output)
    {
        apple_arm::bf8l(input, output);
    }

    static auto attention(std::span<const std::uint16_t> query, std::span<const std::uint16_t> keys,
                          std::span<const std::uint16_t> values, std::size_t heads,
                          std::size_t width, float scale, bool prefill, bool tiled)
    {
        return attention_bf16(query, keys, values, heads, width, scale, prefill, tiled);
    }

    static std::span<const std::uint16_t>
    attention_into(std::span<const std::uint16_t> query, std::span<const std::uint16_t> keys,
                   std::span<const std::uint16_t> values, std::size_t heads, std::size_t width,
                   float scale, bool prefill, bool tiled, AttentionWorkspace& workspace)
    {
        attention_bf16_into(query, keys, values, workspace, heads, width, scale, prefill, tiled);
        return workspace.output;
    }

    Signal snake_dac(std::span<const float> input, SignalShape shape, std::span<const float> alpha,
                     const Limits& limits) const
    {
        return apple_arm::snake_dac(input, shape, alpha, limits, execution);
    }

    void snake_dac_into(std::span<const float> input, SignalShape shape,
                        std::span<const float> alpha, Signal& output, const Limits& limits) const
    {
        apple_arm::snake_dac_into(input, shape, alpha, output, limits, execution);
    }

    static void linear_column(std::span<const std::int8_t> weights,
                              std::span<const float> weight_scales, std::span<const float> bias,
                              std::size_t input_width, std::size_t output_width,
                              std::span<const std::int8_t> input, std::span<const float> scales,
                              std::span<std::uint16_t> output);
};
} // namespace tsac::apple_arm
