#include "common/numeric.hpp"

#include "common/numeric/bf16.hpp"
#include "common/probability.hpp"

#include <array>
#include <bit>
#include <iostream>

namespace
{
void check(bool condition)
{
    if (!condition)
    {
        throw tsac::Error("numeric bit compatibility failure");
    }
}

void finite_classification()
{
    constexpr std::array<std::uint32_t, 10> finite{0,          0x80000000, 1,          0x80000001,
                                                   0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
                                                   0x7f7fffff, 0xff7fffff};
    constexpr std::array<std::uint32_t, 8> nonfinite{0x7f800000, 0xff800000, 0x7fc00000,
                                                     0xffc00000, 0x7f800001, 0xff800001,
                                                     0x7fffffff, 0xffffffff};
    for (std::size_t size : {0u, 1u, 3u, 4u, 7u, 15u, 16u, 17u, 31u, 32u, 33u, 257u})
    {
        // Offset the span as well as exercising SIMD tails.
        std::vector<float> storage(size + 1);
        auto values = std::span(storage).subspan(1);
        for (std::size_t i = 0; i < size; ++i)
        {
            values[i] = std::bit_cast<float>(finite[i % finite.size()]);
        }
        tsac::detail::require_finite(values, "finite rejected");
        for (std::size_t i = 0; i < size; ++i)
        {
            const float saved = values[i];
            for (auto bits : nonfinite)
            {
                values[i] = std::bit_cast<float>(bits);
                bool rejected = false;
                try
                {
                    tsac::detail::require_finite(values, "expected nonfinite");
                }
                catch (const tsac::Error&)
                {
                    rejected = true;
                }
                check(rejected);
            }
            values[i] = saved;
        }
    }
}

void bf16_conversion()
{
    // Every BF16 payload, including NaNs, plus both sides of every float32
    // rounding midpoint. The expected value uses the captured integer rule.
    for (std::uint32_t upper = 0; upper < 65536; ++upper)
    {
        check(std::bit_cast<std::uint32_t>(
                  tsac::probability_float(static_cast<std::uint16_t>(upper))) == upper << 16);
        for (std::uint32_t lower : {0u, 0x7ffeu, 0x7fffu, 0x8000u, 0x8001u, 0xffffu})
        {
            const auto bits = (upper << 16) | lower;
            const auto expected = static_cast<std::uint16_t>(
                upper + (lower > 0x8000u || (lower == 0x8000u && (upper & 1u))));
            check(tsac::probability_bf16(std::bit_cast<float>(bits)) == expected);
        }
    }
}

void in_place_probability()
{
    std::vector<std::uint16_t> input(512), gain(512), bias(512), coefficients(128);
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] = tsac::probability_bf16(static_cast<float>(static_cast<int>(i % 37) - 18) / 16);
        gain[i] = tsac::probability_bf16(1.0f + static_cast<float>(i % 7) / 16);
        bias[i] = tsac::probability_bf16(static_cast<float>(i % 11) / 32);
    }
    const auto expected = tsac::probability_layer_norm_bf16(input, gain, bias);
    std::array<float, 512> values, squares;
    tsac::probability_layer_norm_bf16_into(input, gain, bias, input, values, squares);
    check(input == expected);
    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
        coefficients[i] = tsac::probability_bf16(static_cast<float>(i % 13) / 16);
    }
    auto head = std::span(input).first(128);
    const auto rotated = tsac::probability_rotary_bf16(head, coefficients);
    tsac::probability_rotary_bf16_into(head, coefficients, head);
    check(std::equal(head.begin(), head.end(), rotated.begin()));
    const auto softmax = tsac::probability_softmax_bf16(head);
    tsac::probability_softmax_bf16_into(head, head, std::span(values).first(head.size()));
    check(std::equal(head.begin(), head.end(), softmax.begin()));
}
} // namespace

int main()
{
    try
    {
        finite_classification();
        bf16_conversion();
        in_place_probability();
        std::cout << "Numeric classification and BF16 bits passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
