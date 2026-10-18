#include "common/numeric/probability.hpp"

namespace tsac
{
using namespace detail::probability;

std::uint16_t probability_add_bf16(std::uint16_t left, std::uint16_t right)
{
    const NearestRounding rounding;
    volatile float sum = probability_float(left) + probability_float(right);
    return probability_bf16(sum);
}

std::vector<std::uint16_t> probability_rotary_bf16(std::span<const std::uint16_t> values,
                                                   std::span<const std::uint16_t> coefficients)
{
    detail::require(values.size() == coefficients.size() && values.size() % 2 == 0,
                    "rotary pair shape mismatch");
    std::vector<std::uint16_t> result(values.size());
    probability_rotary_bf16_into(values, coefficients, result);
    return result;
}

void probability_rotary_bf16_into(std::span<const std::uint16_t> values,
                                  std::span<const std::uint16_t> coefficients,
                                  std::span<std::uint16_t> result)
{
    const NearestRounding rounding;
    detail::require(values.size() == coefficients.size() && values.size() % 2 == 0 &&
                        result.size() == values.size(),
                    "rotary pair shape mismatch");
    for (std::size_t i = 0; i < values.size(); i += 2)
    {
        const float real = probability_float(values[i]),
                    imaginary = probability_float(values[i + 1]);
        const float cosine = probability_float(coefficients[i]),
                    sine = probability_float(coefficients[i + 1]);
        volatile float imaginary_sine = imaginary * sine;
        volatile float real_sine = real * sine;
        result[i] = probability_bf16(std::fma(real, cosine, -imaginary_sine));
        result[i + 1] = probability_bf16(std::fma(imaginary, cosine, real_sine));
    }
}

std::uint16_t probability_gelu_bf16(std::uint16_t value)
{
    const NearestRounding rounding;
    const float x = probability_float(value);
    // Keep the original separately rounded multiply/FMA boundaries. Ordinary
    // tanh or erf GELU changes BF16 outputs, even when float32 errors are small.
    volatile float cubic = 0x1.2444f2p-5f * x;
    float z = std::fma(cubic, x, 0x1.988454p-1f);
    z = z * x;
    volatile float half = x * 0.5f;
    z = z * -2.0f;
    const float exponential = original_exp(z);
    volatile float denominator = exponential + 1.0f;
    volatile float sigmoid = 2.0f / denominator;
    volatile float tanh = sigmoid - 1.0f;
    volatile float plus = tanh + 1.0f;
    volatile float result = half * plus;
    return probability_bf16(result);
}

void probability_gelu_bf16(std::span<std::uint16_t> values)
{
    // All 65,536 BF16 bit patterns have deterministic BF16 outputs. Build the
    // 128-KiB table once with the reference arithmetic (including NaNs and
    // infinities), then share it read-only across independent model lanes.
    // Function-local static initialization synchronizes the first construction;
    // subsequent calls need only one table lookup per activation.
    static const auto table = []
    {
        std::array<std::uint16_t, 65536> result{};
        for (std::size_t i = 0; i < result.size(); ++i)
        {
            result[i] = probability_gelu_bf16(static_cast<std::uint16_t>(i));
        }
        return result;
    }();
    for (auto& value : values)
    {
        value = table[value];
    }
}

std::vector<std::uint16_t> probability_softmax_bf16(std::span<const std::uint16_t> input)
{
    detail::require(!input.empty(), "empty softmax input");
    std::vector<float> exponents(input.size());
    std::vector<std::uint16_t> result(input.size());
    probability_softmax_bf16_into(input, result, exponents);
    return result;
}

void probability_softmax_bf16_into(std::span<const std::uint16_t> input,
                                   std::span<std::uint16_t> result, std::span<float> exponents)
{
    const NearestRounding rounding;
    detail::require(!input.empty() && result.size() == input.size() &&
                        exponents.size() == input.size(),
                    "softmax shape mismatch");
    float maximum = -std::numeric_limits<float>::infinity();
    for (auto value : input)
    {
        const float x = probability_float(value);
        detail::require(std::isfinite(x), "softmax input must be finite");
        maximum = std::max(maximum, x);
    }
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        volatile float shifted = probability_float(input[i]) - maximum;
        exponents[i] = original_exp(shifted);
    }
    const float total = probability_sum(exponents);
    volatile float reciprocal = 1.0f / total;
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        volatile float scaled = exponents[i] * reciprocal;
        result[i] = probability_bf16(scaled);
    }
}

std::vector<std::uint16_t> probability_layer_norm_bf16(std::span<const std::uint16_t> input,
                                                       std::span<const std::uint16_t> gain,
                                                       std::span<const std::uint16_t> bias,
                                                       float epsilon)
{
    detail::require(!input.empty() && input.size() == gain.size() && input.size() == bias.size(),
                    "layer normalization shape mismatch");
    detail::require(std::isfinite(epsilon) && epsilon > 0, "invalid layer normalization epsilon");
    std::vector<float> values(input.size()), squares(input.size());
    std::vector<std::uint16_t> result(input.size());
    probability_layer_norm_bf16_into(input, gain, bias, result, values, squares, epsilon);
    return result;
}

void probability_layer_norm_bf16_into(std::span<const std::uint16_t> input,
                                      std::span<const std::uint16_t> gain,
                                      std::span<const std::uint16_t> bias,
                                      std::span<std::uint16_t> result, std::span<float> values,
                                      std::span<float> squares, float epsilon)
{
    const NearestRounding rounding;
    detail::require(!input.empty() && input.size() == gain.size() && input.size() == bias.size() &&
                        result.size() == input.size() && values.size() == input.size() &&
                        squares.size() == input.size(),
                    "layer normalization shape mismatch");
    detail::require(std::isfinite(epsilon) && epsilon > 0, "invalid layer normalization epsilon");
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        values[i] = probability_float(input[i]);
        squares[i] = values[i] * values[i];
    }
    volatile float reciprocal = 1.0f / static_cast<float>(values.size());
    volatile float mean = probability_sum(values) * reciprocal;
    float variance = std::fma(probability_sum(squares), static_cast<float>(reciprocal), 0.0f);
    variance = std::fma(static_cast<float>(mean), -static_cast<float>(mean), variance);
    variance += epsilon;
    volatile float inverse = 1.0f / std::sqrt(variance);
    volatile float offset = -mean * inverse;
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        const float normalized =
            std::fma(values[i], static_cast<float>(inverse), static_cast<float>(offset));
        result[i] = probability_bf16(
            std::fma(normalized, probability_float(gain[i]), probability_float(bias[i])));
    }
}

} // namespace tsac
