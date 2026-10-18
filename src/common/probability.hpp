#pragma once
#include "common/common.hpp"
#include "common/numeric/bf16.hpp"

namespace tsac
{
struct AttentionWorkspace
{
    std::vector<std::uint16_t> output;
    std::vector<float> numerator;
};

std::uint16_t probability_add_bf16(std::uint16_t left, std::uint16_t right);
// Interleaved real/imaginary pairs, with one coefficient pair per input pair.
std::vector<std::uint16_t> probability_rotary_bf16(std::span<const std::uint16_t> values,
                                                   std::span<const std::uint16_t> coefficients);
void probability_rotary_bf16_into(std::span<const std::uint16_t> values,
                                  std::span<const std::uint16_t> coefficients,
                                  std::span<std::uint16_t> output);

// Original BF16 GELU with its FMA polynomial and exponential approximation.
std::uint16_t probability_gelu_bf16(std::uint16_t value);
// In-place batch evaluation using a lookup over all BF16 bit patterns.
void probability_gelu_bf16(std::span<std::uint16_t> values);

std::vector<std::uint16_t> probability_softmax_bf16(std::span<const std::uint16_t> values);
void probability_softmax_bf16_into(std::span<const std::uint16_t> values,
                                   std::span<std::uint16_t> output, std::span<float> exponents);

std::vector<std::uint16_t> probability_layer_norm_bf16(std::span<const std::uint16_t> values,
                                                       std::span<const std::uint16_t> gain,
                                                       std::span<const std::uint16_t> bias,
                                                       float epsilon = 1e-5f);
void probability_layer_norm_bf16_into(std::span<const std::uint16_t> values,
                                      std::span<const std::uint16_t> gain,
                                      std::span<const std::uint16_t> bias,
                                      std::span<std::uint16_t> output, std::span<float> expanded,
                                      std::span<float> squares, float epsilon = 1e-5f);

struct ProbabilityBf8l
{
    std::vector<std::int8_t> values;
    std::vector<std::uint16_t> scales; // One BF16 scale per consecutive 256 values.
};

} // namespace tsac
