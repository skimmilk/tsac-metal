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
// Expand a contiguous range into caller-owned storage. BF8 ranges must contain
// complete axis-zero columns, including when they cross an outer-plane boundary.
void expand_into(const EncodedTensorView& tensor, std::size_t first, std::span<float> values);
} // namespace tsac::weights
