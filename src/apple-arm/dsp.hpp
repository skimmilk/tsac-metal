#pragma once
#include "apple-arm/execution.hpp"
#include "common/dsp.hpp"

namespace tsac::apple_arm
{
class Conv1d final : public Convolution<Conv1d>
{
public:
    Conv1d(Conv1dConfig config, std::span<const float> weights, std::span<const float> bias,
           const Limits& limits = {});
    static Conv1d from_prepared(Conv1dConfig config, const weights::View& weights,
                                const weights::View& bias, const Limits& limits);

    void set_execution(const Execution* execution)
    {
        execution_ = execution;
    }

    void evaluate(std::span<const float> input, SignalShape shape, Signal& output,
                  const Limits& limits) const;

private:
    Conv1d() = default;

    struct Phase
    {
        std::size_t residue, weight_offset;
        std::vector<std::size_t> taps;
    };

    struct Tile
    {
        std::size_t group, first, step, frames;
        const Phase* phase;
    };

    void prepare_transposed(std::span<const float> weights);
    void lower_tile(std::span<const float> input, SignalShape shape, const Tile& tile,
                    std::span<float> columns) const;
    void apply_tile(std::span<const float> input, SignalShape shape, Signal& output,
                    const Tile& tile) const;
    std::size_t tile_size(std::size_t frames, const Limits& limits) const;
    const Execution* execution_ = nullptr;
    weights::View prepared_;
    std::vector<Phase> phases_;
};

void snake_dac_into(std::span<const float> input, SignalShape shape, std::span<const float> alpha,
                    Signal& output, const Limits& limits = {},
                    const Execution* execution = nullptr);
Signal snake_dac(std::span<const float> input, SignalShape shape, std::span<const float> alpha,
                 const Limits& limits = {}, const Execution* execution = nullptr);
} // namespace tsac::apple_arm
