#include "common/numeric.hpp"
#include "scalar/probability_numeric.hpp"

namespace tsac::scalar
{
void add_bf16(std::span<std::uint16_t> left, std::span<const std::uint16_t> right)
{
    const NearestRounding rounding;
    detail::require(left.size() == right.size(), "BF16 addition shape mismatch");
    for (std::size_t i = 0; i < left.size(); ++i)
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
    for (std::size_t offset = 0; offset < values.size(); offset += 256)
    {
        float maximum = 0;
        for (std::size_t i = 0; i < 256; ++i)
        {
            maximum = std::max(maximum, std::abs(detail::probability::flush(values[offset + i])));
        }
        const auto scale = probability_bf16(std::max(divide(maximum, 127.0f), 1e-35f));
        result.scales.push_back(scale);
        const float reciprocal = divide(1.0f, probability_float(scale));
        for (std::size_t i = 0; i < 256; ++i)
        {
            const auto value = detail::probability::multiply(values[offset + i], reciprocal);
            result.values[offset + i] =
                static_cast<std::int8_t>(std::clamp(std::nearbyint(value), -128.0f, 127.0f));
        }
    }
}

ProbabilityBf8l bf8l(std::span<const float> values)
{
    ProbabilityBf8l result;
    bf8l(values, result);
    return result;
}
} // namespace tsac::scalar
