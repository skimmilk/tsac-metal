#pragma once
#include "common/internal.hpp"
#include "common/numeric/probability.hpp"
#include "common/probability.hpp"
#include "format/categorical.hpp"

#include <algorithm>
#include <arm_neon.h>
#include <array>
#include <cmath>
#include <limits>

namespace tsac::apple_arm
{
using detail::probability::divide;
using detail::probability::NearestRounding;
using detail::probability::original_exp;

inline float32x4_t flush_vector(float32x4_t x)
{
    const auto bits = vreinterpretq_u32_f32(x);
    const auto tiny = vceqq_u32(vandq_u32(bits, vdupq_n_u32(0x7f800000)), vdupq_n_u32(0));
    return vreinterpretq_f32_u32(vbslq_u32(tiny, vandq_u32(bits, vdupq_n_u32(0x80000000)), bits));
}

inline uint16x4_t round_bf16(float32x4_t x)
{
    const auto bits = vreinterpretq_u32_f32(x);
    return vshrn_n_u32(vaddq_u32(vaddq_u32(bits, vdupq_n_u32(0x7fff)),
                                 vandq_u32(vshrq_n_u32(bits, 16), vdupq_n_u32(1))),
                       16);
}

inline float32x4_t expand4(const std::uint16_t* source)
{
    return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(source), 16));
}
} // namespace tsac::apple_arm
