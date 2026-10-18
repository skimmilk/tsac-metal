#include "chunks.hpp"
#include "common/analysis.hpp"
#include "common/numeric.hpp"
#include "models/contract.hpp"

#include <algorithm>
#include <array>
#include <utility>

namespace tsac
{
namespace analysis_detail
{
using Alpha = weights::Array<float>;

inline void observe(const AnalysisObserver& observer, std::string_view name, const Signal& signal)
{
    if (observer)
    {
        observer(name, signal);
    }
}

template <class Kernels> struct ResidualUnit
{
    Alpha alpha0;
    typename Kernels::Conv1d conv0;
    Alpha alpha1;
    typename Kernels::Conv1d conv1;

    void apply_into(const Signal& input, Signal& x, Signal& temporary, const Limits& limits,
                    const AnalysisObserver& observer, std::size_t stage, std::size_t unit,
                    const Kernels& kernels) const
    {
        kernels.snake_dac_into(input.values, input.shape, alpha0, x, limits);
        const auto trace = [&](int operation)
        {
            if (observer)
            {
                auto label = std::string("encoder.block.") + std::to_string(stage) + ".block." +
                             std::to_string(unit);
                if (operation >= 0)
                {
                    label += ".block." + std::to_string(operation);
                }
                observer(label, x);
            }
        };
        trace(0);
        conv0.apply_into(x.values, x.shape, temporary, limits);
        std::swap(x, temporary);
        trace(1);
        kernels.snake_dac_into(x.values, x.shape, alpha1, temporary, limits);
        std::swap(x, temporary);
        trace(2);
        conv1.apply_into(x.values, x.shape, temporary, limits);
        std::swap(x, temporary);
        trace(3);
        detail::require(x.shape.channels == input.shape.channels &&
                            x.shape.frames == input.shape.frames &&
                            x.shape.groups == input.shape.groups,
                        "analysis residual shape mismatch");
        {
            for (std::size_t i = 0; i < x.values.size(); ++i)
            {
                x.values[i] = input.values[i] + x.values[i];
            }
        }
        detail::require_finite(x.values, "nonfinite analysis residual");
        trace(-1);
    }
};

template <class Kernels> struct EncoderBlock
{
    std::vector<analysis_detail::ResidualUnit<Kernels>> residuals;
    Alpha alpha;
    typename Kernels::Conv1d downsample;
    std::size_t width;
};
} // namespace analysis_detail

using analysis_detail::Alpha;

template <class Kernels> struct DacAnalysis<Kernels>::Impl
{
    Kernels kernels;
    unsigned channels;
    typename Kernels::Conv1d entry;
    std::vector<analysis_detail::EncoderBlock<Kernels>> blocks;
    Alpha final_alpha;
    typename Kernels::Conv1d exit;

    void preflight(SignalShape shape, const Limits& limits) const
    {
        shape.elements(limits);
        shape = {64, entry.output_frames(shape.frames), shape.groups};
        shape.elements(limits);
        for (const auto& block : blocks)
        {
            shape = {block.width, block.downsample.output_frames(shape.frames), shape.groups};
            shape.elements(limits);
        }
        shape = {latent_channels, exit.output_frames(shape.frames), shape.groups};
        shape.elements(limits);
    }
};

template <class Kernels>
DacAnalysis<Kernels>::DacAnalysis(std::shared_ptr<const Impl> impl) : impl_(std::move(impl))
{
}

template <class Kernels>
DacAnalysis<Kernels> DacAnalysis<Kernels>::from_model(DacAnalysisModel model, const Limits& limits,
                                                      Kernels kernels)
{
    const auto conv = [&](const ConvolutionWeights& weights, std::size_t stride = 1,
                          std::size_t dilation = 1, bool transposed = false)
    {
        auto result = Kernels::Conv1d::from_weights(weights, limits, stride, dilation, transposed);
        kernels.bind(result);
        return result;
    };
    auto entry = conv(model.entry_convolution());
    std::vector<analysis_detail::EncoderBlock<Kernels>> blocks;
    std::size_t width = 64;
    for (unsigned stage = 0; stage < 4; ++stage)
    {
        const auto stride = std::array<unsigned, 4>{2, 4, 8, 8}[stage];
        std::vector<analysis_detail::ResidualUnit<Kernels>> residuals;
        for (unsigned unit = 0; unit < 3; ++unit)
        {
            auto alpha0 =
                Alpha(model.residual_activation(stage, unit, ResidualActivation::before_dilated));
            auto conv0 = conv(model.residual_convolution(stage, unit, ResidualConvolution::dilated),
                              1, std::array<unsigned, 3>{1, 3, 9}[unit]);
            auto alpha1 =
                Alpha(model.residual_activation(stage, unit, ResidualActivation::before_pointwise));
            auto conv1 =
                conv(model.residual_convolution(stage, unit, ResidualConvolution::pointwise));
            residuals.push_back(
                {std::move(alpha0), std::move(conv0), std::move(alpha1), std::move(conv1)});
        }
        auto block_alpha = Alpha(model.resample_activation(stage));
        auto resample = conv(model.resample_convolution(stage), stride);
        width *= 2;
        blocks.push_back(
            {std::move(residuals), std::move(block_alpha), std::move(resample), width});
    }
    auto final_alpha = Alpha(model.final_activation());
    auto exit = conv(model.exit_convolution());
    auto impl = std::make_shared<Impl>(Impl{kernels, model_contract::channels(model.variant()),
                                            std::move(entry), std::move(blocks),
                                            std::move(final_alpha), std::move(exit)});
    return DacAnalysis<Kernels>(std::move(impl));
}

template <class Kernels> unsigned DacAnalysis<Kernels>::channels() const
{
    return impl_->channels;
}

template <class Kernels> std::size_t DacAnalysis<Kernels>::output_frames(std::size_t pcm_frames)
{
    detail::require(pcm_frames > 0 && pcm_frames % hop == 0,
                    "analysis PCM must be a nonempty multiple of 512 samples");
    return pcm_frames / hop;
}

template <class Kernels>
Signal DacAnalysis<Kernels>::apply(std::span<const float> pcm, SignalShape shape,
                                   const Limits& limits, const AnalysisObserver& observer) const
{
    detail::require(shape.channels == channels(), "analysis input channel mismatch");
    const auto frames = output_frames(shape.frames);
    detail::require(pcm.size() == shape.elements(limits), "analysis input shape mismatch");
    impl_->preflight(shape, limits);
    const auto& kernels = impl_->kernels;
    Signal scratch, temporary;
    auto x = impl_->entry.apply(pcm, shape, limits);
    analysis_detail::observe(observer, "encoder.block.0", x);
    for (std::size_t stage = 0; stage < impl_->blocks.size(); ++stage)
    {
        const auto& block = impl_->blocks[stage];
        const auto trace = [&](unsigned operation)
        {
            if (observer)
            {
                observer(std::string("encoder.block.") + std::to_string(stage + 1) + ".block." +
                             std::to_string(operation),
                         x);
            }
        };
        for (std::size_t unit = 0; unit < block.residuals.size(); ++unit)
        {
            block.residuals[unit].apply_into(x, scratch, temporary, limits, observer, stage + 1,
                                             unit + 0, kernels);
            std::swap(x, scratch);
        }
        kernels.snake_dac_into(x.values, x.shape, block.alpha, scratch, limits);
        std::swap(x, scratch);
        trace(3);
        block.downsample.apply_into(x.values, x.shape, scratch, limits);
        std::swap(x, scratch);
        trace(4);
    }
    kernels.snake_dac_into(x.values, x.shape, impl_->final_alpha, scratch, limits);
    std::swap(x, scratch);
    analysis_detail::observe(observer, "encoder.block.5", x);
    impl_->exit.apply_into(x.values, x.shape, scratch, limits);
    std::swap(x, scratch);
    analysis_detail::observe(observer, "encoder.block.6", x);
    detail::require(x.shape.frames == frames, "analysis output length mismatch");
    return x;
}

template <class Kernels>
Signal DacAnalysis<Kernels>::apply_chunked(std::span<const float> pcm, SignalShape shape,
                                           std::size_t chunk_frames, const Limits& limits,
                                           const AnalysisObserver& observer) const
{
    detail::require(shape.channels == channels(), "analysis input channel mismatch");
    const auto frames = output_frames(shape.frames);
    detail::require(pcm.size() == shape.elements(limits), "analysis input shape mismatch");
    detail::require(chunk_frames > 0, "empty analysis chunk");
    detail::require_finite(pcm, "nonfinite analysis input");
    Signal output{{latent_channels, frames, shape.groups}, {}};
    const auto count = output.shape.elements(limits);
    const auto context = left_context_frames + right_context_frames;
    const auto max_frames =
        chunk_frames >= frames ? frames : std::min(frames, checked_add(chunk_frames, context));
    impl_->preflight({channels(), checked_mul(max_frames, hop), 1}, limits);
    output.values.resize(count);
    for (std::size_t group = 0; group < shape.groups; ++group)
    {
        for (std::size_t begin = 0; begin < frames;)
        {
            const auto central = std::min(chunk_frames, frames - begin);
            const auto end = begin + central;
            const auto left = begin > left_context_frames ? begin - left_context_frames : 0;
            const auto right = end + std::min(right_context_frames, frames - end);
            const SignalShape chunk_shape{channels(), (right - left) * hop, 1};
            const auto source = (group * shape.frames + left * hop) * channels();
            const auto chunk = pcm.subspan(source, chunk_shape.elements(limits));
            const auto chunk_observer =
                detail::chunk_observer(observer, "group." + std::to_string(group) + ".chunk." +
                                                     std::to_string(begin) + ".");
            const auto evaluated = apply(chunk, chunk_shape, limits, chunk_observer);
            const auto retained = central * latent_channels;
            const auto offset = (begin - left) * latent_channels;
            const auto destination = (group * frames + begin) * latent_channels;
            std::copy_n(evaluated.values.begin() + static_cast<std::ptrdiff_t>(offset), retained,
                        output.values.begin() + static_cast<std::ptrdiff_t>(destination));
            begin = end;
        }
    }
    return output;
}
} // namespace tsac
