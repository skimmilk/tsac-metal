#pragma once
#include "common/common.hpp"
#include "weights/tensor_weights.hpp"

namespace tsac::weights
{
void normalize_search(std::span<const float, 8> input, std::span<float, 8> output);
void project_codebook(std::uint16_t symbol, std::span<const float> codebook,
                      std::span<const float> weights, std::span<const float> bias,
                      std::span<float> output, bool fused = true);
std::vector<float> expand(const EncodedTensorView& tensor);
} // namespace tsac::weights
