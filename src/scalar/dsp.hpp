#pragma once
#include "common/dsp.hpp"

namespace tsac::scalar
{
class Conv1d final : public Convolution<Conv1d>
{
public:
    Conv1d(Conv1dConfig config, std::span<const float> weights, std::span<const float> bias,
           const Limits& limits = {})
        : Convolution(config, weights, bias, limits)
    {
    }

    void evaluate(std::span<const float> input, SignalShape shape, Signal& output,
                  const Limits&) const;

private:
    friend class Convolution<Conv1d>;
    Conv1d() = default;
    float sample(std::span<const float> input, SignalShape shape, std::size_t group,
                 std::size_t frame, std::size_t channel) const;
};

Signal snake(std::span<const float> input, SignalShape shape, std::span<const float> alpha,
             const Limits& limits = {});
void snake_dac_into(std::span<const float> input, SignalShape shape, std::span<const float> alpha,
                    Signal& output, const Limits& limits = {});
Signal snake_dac(std::span<const float> input, SignalShape shape, std::span<const float> alpha,
                 const Limits& limits = {});
} // namespace tsac::scalar
