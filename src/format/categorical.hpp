#pragma once
#include "format/range.hpp"

namespace tsac
{
// Reproduce the original host categorical reduction (64-value balanced blocks,
// binary-carry accumulation), including x86 FTZ/DAZ at each float operation.
float probability_sum(std::span<const float> probabilities);

// Original bisection coding. Probabilities are in symbol-index order; initial
// mass is exactly 1, and right-subtree mass uses subtraction, never re-summing.
void encode_categorical(RangeEncoder& encoder, std::span<const float> probabilities,
                        std::size_t symbol);
std::size_t decode_categorical(RangeDecoder& decoder, std::span<const float> probabilities);

// Explicit scalar float32 multiply, divide, nearest-even integer conversion,
// and [1,32767] clamp used at each tree node. Does not use ambient int rounding.
std::uint16_t conditional_q15(float left_mass, float mass);
} // namespace tsac
