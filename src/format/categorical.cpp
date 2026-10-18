#include "format/categorical.hpp"

#include "common/internal.hpp"
#include "common/numeric/float_math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace tsac
{
namespace
{
using namespace detail::probability;

float pair_sum(std::span<const float> input, std::size_t width)
{
    std::array<float, 64> values{};
    std::copy(input.begin(), input.end(), values.begin());
    while (width > 1)
    {
        width /= 2;
        for (std::size_t i = 0; i < width; ++i)
        {
            values[i] = add(values[2 * i], values[2 * i + 1]);
        }
    }
    return values[0];
}

void validate(std::span<const float> probabilities)
{
    detail::require(!probabilities.empty(), "empty categorical alphabet");
    detail::require(std::all_of(probabilities.begin(), probabilities.end(),
                                [](float x)
                                {
                                    return std::isfinite(x) && x >= 0;
                                }),
                    "categorical probabilities must be finite and nonnegative");
}
} // namespace

float probability_sum(std::span<const float> probabilities)
{
    const NearestRounding rounding;
    if (probabilities.size() <= 8)
    {
        return pair_sum(probabilities, 8);
    }
    // One slot per size_t bit suffices even at the maximum representable span.
    std::array<float, std::numeric_limits<std::size_t>::digits> partial{};
    std::size_t blocks = 0;
    for (std::size_t offset = 0; offset < probabilities.size();)
    {
        const auto count = std::min<std::size_t>(64, probabilities.size() - offset);
        float value = pair_sum(probabilities.subspan(offset, count), 64);
        ++blocks;
        const auto level = static_cast<std::size_t>(std::countr_zero(blocks));
        for (std::size_t i = 0; i < level; ++i)
        {
            value = add(value, partial[i]);
            partial[i] = 0;
        }
        partial[level] = value;
        offset += count;
    }
    float value = partial[0];
    for (std::size_t i = 1; i < static_cast<std::size_t>(std::bit_width(blocks)); ++i)
    {
        value = add(value, partial[i]);
    }
    return value;
}

std::uint16_t conditional_q15(float left_mass, float mass)
{
    const NearestRounding rounding;
    const float value = divide(multiply(left_mass, 32768.0f), mass);
    // CVTSS2SI returns INT64_MIN for NaN, infinities or overflow; the original
    // consumes its low signed 32 bits, which are zero, then clamps to one.
    if (!std::isfinite(value) || value >= 0x1p63f || value < -0x1p63f)
    {
        return 1;
    }
    const auto integer = static_cast<std::int64_t>(std::nearbyint(value));
    const auto low = std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(integer));
    return static_cast<std::uint16_t>(std::clamp(low, 1, 32767));
}

void encode_categorical(RangeEncoder& encoder, std::span<const float> probabilities,
                        std::size_t symbol)
{
    const NearestRounding rounding;
    validate(probabilities);
    detail::require(symbol < probabilities.size(), "categorical symbol outside alphabet");
    std::size_t base = 0, count = probabilities.size();
    float mass = 1;
    while (count > 1)
    {
        const auto half = count / 2;
        const float left = probability_sum(probabilities.subspan(base, half));
        const bool bit = symbol >= base + half;
        encoder.put(conditional_q15(left, mass), bit);
        if (bit)
        {
            base += half;
            count -= half;
            mass = subtract(mass, left);
        }
        else
        {
            count = half;
            mass = left;
        }
    }
}

std::size_t decode_categorical(RangeDecoder& decoder, std::span<const float> probabilities)
{
    const NearestRounding rounding;
    validate(probabilities);
    std::size_t base = 0, count = probabilities.size();
    float mass = 1;
    while (count > 1)
    {
        const auto half = count / 2;
        const float left = probability_sum(probabilities.subspan(base, half));
        if (decoder.get(conditional_q15(left, mass)))
        {
            base += half;
            count -= half;
            mass = subtract(mass, left);
        }
        else
        {
            count = half;
            mass = left;
        }
    }
    return base;
}
} // namespace tsac
