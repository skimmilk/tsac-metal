#include "common/numeric.hpp"
#include "probability_numeric.hpp"

namespace tsac::apple_arm
{
void add_bf16(std::span<std::uint16_t> left, std::span<const std::uint16_t> right)
{
    const NearestRounding rounding;
    detail::require(left.size() == right.size(), "BF16 residual shape mismatch");
    // Widen BF16 bits, add in float32, then round each result back once. Load
    // both operands before storing so a residual may also be added to itself.
    std::size_t i = 0;
    for (; i + 4 <= left.size(); i += 4)
    {
        const auto a = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(left.data() + i), 16));
        const auto b = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(right.data() + i), 16));
        vst1_u16(left.data() + i, round_bf16(vaddq_f32(a, b)));
    }
    for (; i < left.size(); ++i)
    {
        left[i] = probability_add_bf16(left[i], right[i]);
    }
}

void bf8l(std::span<const float> values, ProbabilityBf8l& result)
{
    const NearestRounding rounding;
    detail::require(values.size() % 256 == 0, "BF8L activations require 256-value blocks");
    detail::require_finite(values, "BF8L activations must be finite");
    result.scales.clear();
    result.values.resize(values.size());
    result.scales.reserve(values.size() / 256);
    // BF8L shares one BF16 scale across 256 activations: max(abs(x))/127,
    // floored at 1e-35. Round the scale before computing its reciprocal, then
    // quantize against that rounded scale. Flush subnormals with their sign at
    // the same boundaries as the scalar path; int8 results saturate after ties
    // are rounded to even.
    for (std::size_t offset = 0; offset < values.size(); offset += 256)
    {
        auto maxima = vdupq_n_f32(0);
        for (std::size_t i = 0; i < 256; i += 4)
        {
            maxima =
                vmaxq_f32(maxima, vabsq_f32(flush_vector(vld1q_f32(values.data() + offset + i))));
        }
        const float maximum = vmaxvq_f32(maxima);
        const auto scale_bits = probability_bf16(std::max(divide(maximum, 127.0f), 1e-35f));
        result.scales.push_back(scale_bits);
        const float reciprocal = divide(1.0f, probability_float(scale_bits));
        for (std::size_t i = 0; i < 256; i += 8)
        {
            const auto a = flush_vector(
                vmulq_n_f32(flush_vector(vld1q_f32(values.data() + offset + i)), reciprocal));
            const auto b = flush_vector(
                vmulq_n_f32(flush_vector(vld1q_f32(values.data() + offset + i + 4)), reciprocal));
            // FCVTNS is nearest-even; saturating narrowing implements the same
            // signed-byte clamp. Preserve DAZ/FTZ around the float multiply.
            const auto integers =
                vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(a)), vqmovn_s32(vcvtnq_s32_f32(b)));
            vst1_s8(result.values.data() + offset + i, vqmovn_s16(integers));
        }
    }
}

ProbabilityBf8l bf8l(std::span<const float> values)
{
    ProbabilityBf8l result;
    bf8l(values, result);
    return result;
}
} // namespace tsac::apple_arm
