#include "scalar/dsp.hpp"

#include "common/internal.hpp"
#include "common/snake.hpp"

#include <cmath>
#include <limits>

namespace tsac::scalar
{
float Conv1d::sample(std::span<const float> input, SignalShape shape, std::size_t group,
                     std::size_t frame, std::size_t channel) const
{
    const auto& c = config_;
    const bool ordinary = c.kind == ConvolutionKind::ordinary;
    const auto stored_channels = ordinary ? c.input_channels : c.output_channels;
    const auto column_size = stored_channels * c.kernel;
    float sum = 0;
    for (std::size_t tap = 0; tap < c.kernel; ++tap)
    {
        const auto source = source_frame(frame, tap);
        if (!source || *source >= shape.frames)
        {
            continue;
        }
        const auto base = (group * shape.frames + *source) * c.input_channels;
        for (std::size_t in = 0; in < c.input_channels; ++in)
        {
            const auto index = ordinary ? in + tap * stored_channels + channel * column_size
                                        : channel + tap * stored_channels + in * column_size;
            sum += input[base + in] * weights_[index];
        }
    }
    return sum + bias_[channel];
}

void Conv1d::evaluate(std::span<const float> input, SignalShape shape, Signal& output,
                      const Limits&) const
{
    for (std::size_t group = 0; group < shape.groups; ++group)
    {
        for (std::size_t frame = 0; frame < output.shape.frames; ++frame)
        {
            for (std::size_t channel = 0; channel < config_.output_channels; ++channel)
            {
                output.values[(group * output.shape.frames + frame) * config_.output_channels +
                              channel] = sample(input, shape, group, frame, channel);
            }
        }
    }
}

namespace
{
float dac_sine_squared(float phase)
{
    const auto constant = [](std::uint32_t bits)
    {
        return std::bit_cast<float>(bits);
    };
    const float reduced = phase * constant(0x3f22f983);
    const double lower = std::floor(double(reduced));
    const double fraction = double(reduced) - lower;
    const double nearest =
        lower + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lower, 2.0) != 0));
    const float quadrant = static_cast<float>(nearest);
    // x86 VCVTPS2DQ returns the indefinite integer for values out of range.
    const auto q = nearest >= -2147483648.0 && nearest < 2147483648.0
                       ? static_cast<std::int32_t>(nearest)
                       : std::numeric_limits<std::int32_t>::min();
    float x = std::fma(quadrant, constant(0xbfc90000), phase);
    x = std::fma(quadrant, constant(0xb9fdaa22), x);
    const float x2 = x * x;
    float value;
    if ((q & 1) == 0)
    {
        float polynomial = std::fma(constant(0xb94ca1f9), x2, constant(0x3c08839e));
        polynomial = std::fma(polynomial, x2, constant(0xbe2aaaa3));
        const float cube = x * x2;
        value = std::fma(cube, polynomial, x);
    }
    else
    {
        float polynomial = std::fma(constant(0x37ccf5ce), x2, constant(0xbab6061a));
        polynomial = std::fma(polynomial, x2, constant(0x3d2aaaa5));
        polynomial = std::fma(polynomial, x2, constant(0xbf000000));
        value = std::fma(polynomial, x2, 1.0f);
    }
    // The original quadrant sign XOR cancels when sine is squared.
    return value * value;
}
} // namespace

void scalar_snake_frames(std::span<const float> input, std::span<const float> alpha,
                         std::span<float> output, bool original)
{
    for (std::size_t base = 0; base < input.size(); base += alpha.size())
    {
        std::size_t channel = 0;
        for (; channel < alpha.size(); ++channel)
        {
            const auto i = base + channel;
            const auto a = alpha[channel];
            const float phase = a * input[i];
            detail::require(std::isfinite(phase), "nonfinite Snake phase");
            float squared;
            if (original)
            {
                squared = dac_sine_squared(phase);
            }
            else
            {
                const float s = std::sin(phase);
                squared = s * s;
            }
            output[i] = input[i] + squared / (a + snake_epsilon);
        }
    }
}

Signal snake(std::span<const float> input, SignalShape shape, std::span<const float> alpha,
             const Limits& limits)
{
    return detail::activate(input, shape, alpha, limits,
                            [](auto x, auto a, auto y)
                            {
                                scalar_snake_frames(x, a, y, false);
                            });
}

void snake_dac_into(std::span<const float> input, SignalShape shape, std::span<const float> alpha,
                    Signal& output, const Limits& limits)
{
    detail::activate_into(input, shape, alpha, limits, output,
                          [](auto x, auto a, auto y)
                          {
                              scalar_snake_frames(x, a, y, true);
                          });
}

Signal snake_dac(std::span<const float> input, SignalShape shape, std::span<const float> alpha,
                 const Limits& limits)
{
    Signal output;
    snake_dac_into(input, shape, alpha, output, limits);
    return output;
}
} // namespace tsac::scalar
