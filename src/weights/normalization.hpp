#pragma once

#include "common/common.hpp"

#include <bit>

namespace tsac
{
inline constexpr float snake_epsilon = std::bit_cast<float>(std::uint32_t{0x3089705f});
// Columns are contiguous: [fan_in, columns], axis 0 fastest.
// Scalar float32 reference: W[:,j] = v[:,j] * (g[j] / sqrt(sum(v[:,j]^2))).
// Accumulates squares in increasing input order; rejects nonfinite values and
// zero, underflowed, or overflowed norms. Shared by RVQ and convolution weights.
std::vector<float> fold_weight_norm(std::span<const float> v, std::span<const float> g,
                                    std::size_t fan_in, const Limits& limits = {});
// DAC runtime sequence: RMS normalize, multiply by 1/sqrt(fan_in), then gain.
// Squares reduce through a balanced adjacent-pair tree, padded with zeros.
std::vector<float> fold_dac_weight_norm(std::span<const float> v, std::span<const float> g,
                                        std::size_t fan_in, const Limits& limits = {});
// The output may alias v exactly. This permits bounded expansion/folding without
// changing the reduction tree or any FP32 normalization boundary.
void fold_dac_weight_norm_into(std::span<const float> v, std::span<const float> g,
                               std::size_t fan_in, std::span<float> output,
                               const Limits& limits = {});
} // namespace tsac
