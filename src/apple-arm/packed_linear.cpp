#include "apple-arm/kernels.hpp"

#include <arm_neon.h>

namespace tsac::apple_arm
{
namespace
{
int32x4_t dot_block(const std::int8_t* weights, const std::int8_t* input)
{
    auto d0 = vdupq_n_s32(0), d1 = d0, d2 = d0, d3 = d0;
    for (std::size_t i = 0; i < 256; i += 16)
    {
        const auto x = vld1q_s8(input + i);
        d0 = vdotq_laneq_s32(d0, vld1q_s8(weights + 4 * i), x, 0);
        d1 = vdotq_laneq_s32(d1, vld1q_s8(weights + 4 * i + 16), x, 1);
        d2 = vdotq_laneq_s32(d2, vld1q_s8(weights + 4 * i + 32), x, 2);
        d3 = vdotq_laneq_s32(d3, vld1q_s8(weights + 4 * i + 48), x, 3);
    }
    return vaddq_s32(vaddq_s32(d0, d1), vaddq_s32(d2, d3));
}
} // namespace

void Kernels::linear_column(std::span<const std::int8_t> weights_, std::span<const float> scales_,
                            std::span<const float> bias_, std::size_t input_, std::size_t output_,
                            std::span<const std::int8_t> values, std::span<const float> scales,
                            std::span<std::uint16_t> output)
{
    for (std::size_t out = 0; out < output_; out += 4)
    {
        auto sums = vdupq_n_f32(0);
        for (std::size_t block = 0; block < input_ / 256; ++block)
        {
            const auto dots = dot_block(weights_.data() + out * input_ + block * 1024,
                                        values.data() + block * 256);
            const auto scale =
                vmulq_n_f32(vld1q_f32(scales_.data() + block * output_ + out), scales[block]);
            // Keep floating block FMAs ordered; add bias last, then round to BF16 once.
            sums = vfmaq_f32(sums, vcvtq_f32_s32(dots), scale);
        }
        const auto bits = vreinterpretq_u32_f32(vaddq_f32(sums, vld1q_f32(bias_.data() + out)));
        const auto rounded = vaddq_u32(vaddq_u32(bits, vdupq_n_u32(0x7fff)),
                                       vandq_u32(vshrq_n_u32(bits, 16), vdupq_n_u32(1)));
        vst1_u16(output.data() + out, vshrn_n_u32(rounded, 16));
    }
}
} // namespace tsac::apple_arm
