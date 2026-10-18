#pragma once
#include "common/internal.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>

namespace tsac::detail
{
inline void require_finite(std::span<const float> values, const char* message)
{
    // An unsigned maximum is an associative integer reduction: compilers can
    // vectorize the entire scan without changing floating-point semantics.
    // Clearing the sign accepts signed zero/subnormals and rejects every NaN
    // payload as well as either infinity, without FP exceptions for sNaNs.
    std::uint32_t maximum = 0;
    for (float value : values)
    {
        maximum = std::max(maximum, std::bit_cast<std::uint32_t>(value) & 0x7fffffffu);
    }
    require(maximum < 0x7f800000u, message);
}
} // namespace tsac::detail
