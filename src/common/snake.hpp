#pragma once
#include "common/numeric.hpp"
#include "weights/normalization.hpp"

namespace tsac::scalar
{
void scalar_snake_frames(std::span<const float>, std::span<const float>, std::span<float>, bool);
}

namespace tsac::detail
{
template <class Evaluate>
void activate_into(std::span<const float> input, SignalShape shape, std::span<const float> alpha,
                   const Limits& limits, Signal& output, Evaluate evaluate)
{
    const auto count = shape.elements(limits);
    require(input.size() == count && alpha.size() == shape.channels, "Snake shape mismatch");
    require_finite(input, "nonfinite DSP value");
    require_finite(alpha, "nonfinite DSP value");
    for (float a : alpha)
    {
        require(std::isfinite(a + snake_epsilon) && a + snake_epsilon != 0,
                "invalid Snake denominator");
    }
    output.shape = shape;
    output.values.resize(count);
    evaluate(input, alpha, std::span(output.values));
    require_finite(output.values, "nonfinite DSP value");
}

template <class Evaluate>
Signal activate(std::span<const float> input, SignalShape shape, std::span<const float> alpha,
                const Limits& limits, Evaluate evaluate)
{
    Signal output;
    activate_into(input, shape, alpha, limits, output, evaluate);
    return output;
}
} // namespace tsac::detail
