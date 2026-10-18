#pragma once
#include "common/internal.hpp"
#include "common/numeric/float_math.hpp"
#include "common/probability.hpp"
#include "format/categorical.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace tsac::detail::probability
{
inline float original_exp(float z)
{
    z = 87.0f < z ? 87.0f : z; // AVX min/max choose the second operand for NaNs.
    z = -88.0f > z ? -88.0f : z;
    volatile float exponent = z * 0x1.715476p+0f;
    const auto n = std::isfinite(exponent) ? static_cast<std::int32_t>(std::nearbyint(exponent))
                                           : std::numeric_limits<std::int32_t>::min();
    z = std::fma(static_cast<float>(n), -0x1.630000p-1f, z);
    z = std::fma(static_cast<float>(n), 0x1.bd0106p-13f, z);
    float p = std::fma(0x1.a01a02p-13f, z, 0x1.6c16c2p-10f);
    p = std::fma(p, z, 0x1.111112p-7f);
    p = std::fma(p, z, 0x1.555556p-5f);
    p = std::fma(p, z, 0x1.555556p-3f);
    volatile float square = z * z;
    p = std::fma(p, z, 0.5f);
    z = std::fma(p, square, z);
    volatile float plus_one = z + 1.0f;
    return std::bit_cast<float>(std::bit_cast<std::uint32_t>(static_cast<float>(plus_one)) +
                                (static_cast<std::uint32_t>(n) << 23));
}
} // namespace tsac::detail::probability
