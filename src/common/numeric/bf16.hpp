#pragma once
#include <bit>
#include <cstdint>

namespace tsac
{
// Original CPU conversion uses raw ties-to-even integer rounding. In particular,
// it does not canonicalize NaNs; a signaling NaN can round to infinity.
inline std::uint16_t probability_bf16(float value)
{
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return static_cast<std::uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

inline float probability_float(std::uint16_t value)
{
    return std::bit_cast<float>(std::uint32_t(value) << 16);
}

} // namespace tsac
