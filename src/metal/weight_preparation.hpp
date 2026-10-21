#pragma once

#include "common/common.hpp"
#include "models/types.hpp"

namespace tsac::metal
{
// CPU preparation writes directly into the caller's final execution allocation.
void prepare_dac_weights(const ConvolutionWeights& weights, bool transposed,
                         std::span<float> output, const Limits& limits);
void pack_linear_weights(std::span<const std::int8_t> source, std::size_t input_width,
                         std::size_t output_width, std::size_t rows, std::span<std::int8_t> output);
} // namespace tsac::metal
