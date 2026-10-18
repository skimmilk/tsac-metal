#include "common/numeric/bf16.hpp"
#include "scalar/kernels.hpp"

#include <cmath>

namespace tsac::scalar
{
void Kernels::linear_column(std::span<const std::int8_t> weights_, std::span<const float> scales_,
                            std::span<const float> bias_, std::size_t input_, std::size_t output_,
                            std::span<const std::int8_t> values, std::span<const float> scales,
                            std::span<std::uint16_t> output)
{
    for (std::size_t out = 0; out < output_; ++out)
    {
        float sum = 0;
        for (std::size_t block = 0; block < input_ / 256; ++block)
        {
            std::int32_t dot = 0;
            for (std::size_t i = 0; i < 256; ++i)
            {
                const auto index =
                    out / 4 * 4 * input_ + block * 1024 + (i / 4) * 16 + (out % 4) * 4 + i % 4;
                dot += std::int32_t(weights_[index]) * values[block * 256 + i];
            }
            const float scale = scales_[block * output_ + out] * scales[block];
            sum = std::fma(static_cast<float>(dot), scale, sum);
        }
        output[out] = probability_bf16(sum + bias_[out]);
    }
}
} // namespace tsac::scalar
