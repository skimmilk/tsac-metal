#pragma once
#include "common/numeric.hpp"
#include "models/types.hpp"
#include "weights/preparation.hpp"

#include <algorithm>

namespace tsac
{
namespace detail
{
inline std::size_t weight_count(const Conv1dConfig& config, const Limits& limits)
{
    detail::require(config.input_channels && config.output_channels && config.kernel &&
                        config.stride && config.dilation,
                    "empty convolution dimension");
    detail::require(config.kind == ConvolutionKind::ordinary ||
                        config.kind == ConvolutionKind::transposed,
                    "invalid convolution kind");
    // Validate receptive span and padding even before an input is supplied.
    checked_add(checked_mul(config.kernel - 1, config.dilation), 1);
    checked_mul(config.padding, 2);
    const auto count =
        checked_mul(checked_mul(config.input_channels, config.kernel), config.output_channels);
    detail::require(count <= limits.tensor_elements, "convolution weights exceed element limit");
    return count;
}

inline void prepare_weight_tiles(std::size_t columns, std::size_t count,
                                 const ParallelExecutor* execution,
                                 const std::function<void(std::size_t, std::size_t)>& prepare)
{
    constexpr std::size_t tile = 32;
    const auto jobs = (columns + tile - 1) / tile;
    const auto task = [&](std::size_t job)
    {
        const auto first = job * tile;
        prepare(first, std::min(tile, columns - first));
    };
    if (execution && execution->parallel() && count >= 65536)
    {
        execution->parallel_for(jobs, task);
    }
    else
    {
        for (std::size_t job = 0; job < jobs; ++job)
        {
            task(job);
        }
    }
}
} // namespace detail

template <class Derived>
Convolution<Derived>::Convolution(Conv1dConfig config, std::span<const float> weights,
                                  std::span<const float> bias, const Limits& limits)
    : config_(config)
{
    detail::require(weights.size() == detail::weight_count(config, limits) &&
                        bias.size() == config.output_channels,
                    "convolution weight or bias shape mismatch");
    detail::require_finite(weights, "nonfinite DSP value");
    detail::require_finite(bias, "nonfinite DSP value");
    weights_.assign(weights.begin(), weights.end());
    bias_.assign(bias.begin(), bias.end());
}

template <class Derived>
Derived Convolution<Derived>::weight_normalized(Conv1dConfig config, std::span<const float> v,
                                                std::span<const float> g,
                                                std::span<const float> bias, const Limits& limits)
{
    const auto count = detail::weight_count(config, limits);
    const auto columns =
        config.kind == ConvolutionKind::ordinary ? config.output_channels : config.input_channels;
    detail::require(v.size() == count && g.size() == columns &&
                        bias.size() == config.output_channels,
                    "normalized convolution shape mismatch");
    detail::require_finite(bias, "nonfinite DSP value");
    return Derived(config, fold_weight_norm(v, g, count / columns, limits), bias, limits);
}

template <class Derived>
Derived Convolution<Derived>::from_prepared(Conv1dConfig config, const weights::View& weights,
                                            const weights::View& bias, const Limits& limits)
{
    detail::require(weights.as<float>().size() == detail::weight_count(config, limits) &&
                        bias.as<float>().size() == config.output_channels &&
                        config.kind == ConvolutionKind::ordinary,
                    "invalid prepared convolution");
    return Derived(config, weights.as<float>(), bias.as<float>(), limits);
}

template <class Derived>
Derived Convolution<Derived>::from_weights(const ConvolutionWeights& c, const Limits& limits,
                                           std::size_t stride, std::size_t dilation,
                                           bool transposed, const ParallelExecutor* preparation)
{
    const auto shape = c.values.shape().axes;
    const auto first = shape[0], kernel = shape[1], last = shape[2];
    const Conv1dConfig config{transposed ? last : first,
                              transposed ? first : last,
                              kernel,
                              stride,
                              stride == 1 ? (kernel - 1) * dilation / 2 : (stride + 1) / 2,
                              dilation,
                              transposed ? ConvolutionKind::transposed : ConvolutionKind::ordinary};
    detail::require(c.values.shape().elements() == detail::weight_count(config, limits),
                    "invalid stored convolution shape");
    const auto bias = weights::View(c.bias);
    detail::require(c.gain.size() == last && c.bias.size() == config.output_channels,
                    "normalized convolution shape mismatch");
    detail::require_finite(c.bias.values(), "nonfinite DSP value");
    Derived result;
    result.config_ = config;
    result.bias_ = weights::Array<float>(bias);
    result.prepare_dac_weights(c, limits, preparation);
    return result;
}

template <class Derived>
void Convolution<Derived>::prepare_dac_weights(const ConvolutionWeights& c, const Limits& limits,
                                               const ParallelExecutor* execution)
{
    const auto axes = c.values.shape().axes;
    const auto fan_in = axes[0] * axes[1];
    weights_.resize(c.values.shape().elements());
    const auto gain = c.gain.values();
    detail::prepare_weight_tiles(
        axes[2], weights_.size(), execution,
        [&](std::size_t first, std::size_t width)
        {
            // Ordinary Apple convolutions and the scalar reference consume the
            // stored column layout directly. Expand/fold in the final allocation.
            auto values = std::span(weights_).subspan(first * fan_in, width * fan_in);
            weights::expand_into(c.values, first * fan_in, values);
            fold_dac_weight_norm_into(values, gain.subspan(first, width), fan_in, values, limits);
        });
}

template <class Derived> std::size_t Convolution<Derived>::output_frames(std::size_t frames) const
{
    detail::require(frames > 0, "empty convolution input");
    const auto& config = config_;
    const auto span = checked_add(checked_mul(config.kernel - 1, config.dilation), 1);
    const auto padding = checked_mul(config.padding, 2);
    if (config.kind == ConvolutionKind::ordinary)
    {
        const auto padded = checked_add(frames, padding);
        detail::require(padded >= span, "convolution has no output frames");
        return checked_add((padded - span) / config.stride, 1);
    }
    const auto full = checked_add(checked_mul(frames - 1, config.stride), span);
    detail::require(full > padding, "transposed convolution has no output frames");
    return full - padding;
}

template <class Derived>
std::optional<std::size_t> Convolution<Derived>::source_frame(std::size_t frame,
                                                              std::size_t tap) const
{
    const auto& c = config_;
    if (c.kind == ConvolutionKind::ordinary)
    {
        const auto position = frame * c.stride + tap * c.dilation;
        if (position < c.padding)
        {
            return std::nullopt;
        }
        return position - c.padding;
    }
    const auto position = frame + c.padding, offset = tap * c.dilation;
    if (position < offset || (position - offset) % c.stride)
    {
        return std::nullopt;
    }
    return (position - offset) / c.stride;
}

template <class Derived>
Signal Convolution<Derived>::apply(std::span<const float> input, SignalShape shape,
                                   const Limits& limits) const
{
    Signal output;
    apply_into(input, shape, output, limits);
    return output;
}

template <class Derived>
void Convolution<Derived>::apply_into(std::span<const float> input, SignalShape shape,
                                      Signal& output, const Limits& limits) const
{
    detail::require(shape.channels == config_.input_channels, "convolution input channel mismatch");
    detail::require(input.size() == shape.elements(limits), "convolution input shape mismatch");
    output.shape = {config_.output_channels, output_frames(shape.frames), shape.groups};
    const auto count = output.shape.elements(limits);
    detail::require_finite(input, "nonfinite DSP value");
    output.values.resize(count);
    static_cast<const Derived&>(*this).evaluate(input, shape, output, limits);
    detail::require_finite(output.values, "nonfinite DSP value");
}

} // namespace tsac
