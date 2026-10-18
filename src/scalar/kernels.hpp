#pragma once
#include "common/probability.hpp"
#include "scalar/dsp.hpp"

namespace tsac::scalar
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
    using Conv1d = scalar::Conv1d;

    static void bind(Conv1d&)
    {
    }

    static void add(std::span<std::uint16_t> left, std::span<const std::uint16_t> right)
    {
        add_bf16(left, right);
    }

    static ProbabilityBf8l bf8l(std::span<const float> input)
    {
        return scalar::bf8l(input);
    }

    static void bf8l_into(std::span<const float> input, ProbabilityBf8l& output)
    {
        scalar::bf8l(input, output);
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

    static Signal snake_dac(std::span<const float> input, SignalShape shape,
                            std::span<const float> alpha, const Limits& limits)
    {
        return scalar::snake_dac(input, shape, alpha, limits);
    }

    void snake_dac_into(std::span<const float> input, SignalShape shape,
                        std::span<const float> alpha, Signal& output, const Limits& limits) const
    {
        scalar::snake_dac_into(input, shape, alpha, output, limits);
    }

    static void linear_column(std::span<const std::int8_t> weights,
                              std::span<const float> weight_scales, std::span<const float> bias,
                              std::size_t input_width, std::size_t output_width,
                              std::span<const std::int8_t> input, std::span<const float> scales,
                              std::span<std::uint16_t> output);
};
} // namespace tsac::scalar
