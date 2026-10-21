#include "audio_dac.hpp"

#include "common/internal.hpp"
#include "models/contract.hpp"

#include <array>
#include <optional>
#include <utility>

namespace tsac::metal
{
void AudioDac::convolution(const ConvolutionWeights& weights, const Limits& limits,
                           std::size_t stride, std::size_t dilation, bool transposed)
{
    const auto axes = weights.values.shape().axes;
    const auto first = axes[0], kernel = axes[1], last = axes[2];
    const Conv1dConfig config{transposed ? last : first,
                              transposed ? first : last,
                              kernel,
                              stride,
                              stride == 1 ? (kernel - 1) * dilation / 2 : (stride + 1) / 2,
                              dilation,
                              transposed ? ConvolutionKind::transposed : ConvolutionKind::ordinary};
    operations_.emplace_back(MetalConvolution::from_weights(config, weights, limits));
}

void AudioDac::activation(const TensorView<float>& alpha)
{
    operations_.emplace_back(Activation{Buffer::import_weights(weights::View(alpha))});
}

template <class Model>
void AudioDac::residual(const Model& model, unsigned stage, unsigned unit, const Limits& limits)
{
    operations_.emplace_back(SaveResidual{});
    activation(model.residual_activation(stage, unit, ResidualActivation::before_dilated));
    convolution(model.residual_convolution(stage, unit, ResidualConvolution::dilated), limits, 1,
                std::array<unsigned, 3>{1, 3, 9}[unit]);
    activation(model.residual_activation(stage, unit, ResidualActivation::before_pointwise));
    convolution(model.residual_convolution(stage, unit, ResidualConvolution::pointwise), limits);
    operations_.emplace_back(AddResidual{});
}

AudioDac AudioDac::from_model(DacAnalysisModel model, const Limits& limits)
{
    AudioDac result;
    result.channels_ = model_contract::channels(model.variant());
    result.convolution(model.entry_convolution(), limits);
    for (unsigned stage = 0; stage < 4; ++stage)
    {
        for (unsigned unit = 0; unit < 3; ++unit)
        {
            result.residual(model, stage, unit, limits);
        }
        result.activation(model.resample_activation(stage));
        result.convolution(model.resample_convolution(stage), limits,
                           std::array<unsigned, 4>{2, 4, 8, 8}[stage]);
    }
    result.activation(model.final_activation());
    result.convolution(model.exit_convolution(), limits);
    return result;
}

AudioDac AudioDac::from_model(DacSynthesisModel model, const Limits& limits)
{
    AudioDac result;
    result.channels_ = model_contract::channels(model.variant());
    result.synthesis_ = true;
    result.convolution(model.entry_convolution(), limits);
    for (unsigned stage = 0; stage < 4; ++stage)
    {
        result.activation(model.resample_activation(stage));
        result.convolution(model.resample_convolution(stage), limits,
                           std::array<unsigned, 4>{8, 8, 4, 2}[stage], 1, true);
        for (unsigned unit = 0; unit < 3; ++unit)
        {
            result.residual(model, stage, unit, limits);
        }
    }
    result.activation(model.final_activation());
    result.convolution(model.exit_convolution(), limits);
    return result;
}

void AudioDac::preflight(SignalShape shape, const Limits& limits) const
{
    shape.elements(limits);
    detail::require(shape.channels == (synthesis_ ? 1024 : channels_),
                    "Metal DAC channel mismatch");
    if (synthesis_)
    {
        shape.channels = 1536;
        shape.elements(limits);
        for (auto stride : {8u, 8u, 4u, 2u})
        {
            shape.channels /= 2;
            shape.frames = checked_mul(shape.frames, stride);
            shape.elements(limits);
        }
    }
    else
    {
        detail::require(shape.frames % 512 == 0, "Metal analysis requires padded PCM");
        shape.channels = 64;
        shape.elements(limits);
        for (auto stride : {2u, 4u, 8u, 8u})
        {
            shape.channels *= 2;
            shape.frames /= stride;
            shape.elements(limits);
        }
    }
}

struct AudioDac::Recorder
{
    DacGraph& graph;
    DacTensor x;
    std::optional<DacTensor> skip;
    const Limits& limits;
    bool preserve_input = false;

    explicit Recorder(DacGraph& target, const DacTensor& input, const Limits& policy)
        : graph(target), x(input), limits(policy)
    {
    }

    void operator()(const MetalConvolution& convolution)
    {
        auto next = graph.convolution(convolution, x, limits);
        graph.release(x);
        x = std::move(next);
    }

    void operator()(const Activation& activation)
    {
        auto next = graph.snake(x, activation.alpha, limits);
        if (!preserve_input)
        {
            graph.release(x);
        }
        x = std::move(next);
        preserve_input = false;
    }

    void operator()(SaveResidual)
    {
        skip = x;
        preserve_input = true;
    }

    void operator()(AddResidual)
    {
        if (!skip)
        {
            throw Error("Metal DAC residual has no saved input");
        }
        auto next = graph.residual(*skip, x, false, limits);
        graph.release(*skip);
        graph.release(x);
        skip.reset();
        x = std::move(next);
    }
};

Signal AudioDac::apply(std::span<const float> input, SignalShape shape, const Limits& limits) const
{
    detail::require(input.size() == shape.elements(limits), "Metal DAC input shape mismatch");
    DacGraph graph;
    return graph.download(record(graph, graph.upload(input, shape, limits), limits), limits);
}

DacTensor AudioDac::record(DacGraph& graph, const DacTensor& input, const Limits& limits) const
{
    preflight(input.shape, limits);
    detail::require(input.buffer.size() >= input.shape.elements(limits) * sizeof(float),
                    "Metal DAC buffer shape mismatch");
    Recorder recorder(graph, input, limits);
    for (const auto& operation : operations_)
    {
        std::visit(recorder, operation);
    }
    return recorder.x;
}
} // namespace tsac::metal
