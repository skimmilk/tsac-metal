#include "chunks.hpp"
#include "common/numeric.hpp"
#include "common/synthesis.hpp"
#include "models/contract.hpp"

#include <algorithm>
#include <array>
#include <utility>

namespace tsac
{
namespace synthesis_detail
{
using Alpha = weights::Array<float>;

inline void observe(const SynthesisObserver& observer, std::string_view name, const Signal& signal)
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
                    const SynthesisObserver& observer, std::size_t stage, std::size_t unit,
                    const Kernels& kernels) const
    {
        kernels.snake_dac_into(input.values, input.shape, alpha0, x, limits);
        const auto trace = [&](int operation)
        {
            if (observer)
            {
                auto label = std::string("decoder.model.") + std::to_string(stage) + ".block." +
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
                        "synthesis residual shape mismatch");
        {
            for (std::size_t i = 0; i < x.values.size(); ++i)
            {
                x.values[i] = input.values[i] + x.values[i];
            }
        }
        detail::require_finite(x.values, "nonfinite synthesis residual");
        trace(-1);
    }
};

template <class Kernels> struct DecoderBlock
{
    Alpha alpha;
    typename Kernels::Conv1d upsample;
    std::vector<synthesis_detail::ResidualUnit<Kernels>> residuals;
    std::size_t width;
};
} // namespace synthesis_detail

using synthesis_detail::Alpha;

template <class Kernels> struct DacSynthesis<Kernels>::Impl
{
    Kernels kernels;
    unsigned channels;
    typename Kernels::Conv1d entry;
    std::vector<synthesis_detail::DecoderBlock<Kernels>> blocks;
    Alpha final_alpha;
    typename Kernels::Conv1d exit;

    void preflight(SignalShape shape, const Limits& limits) const
    {
        shape.elements(limits);
        shape = {1536, entry.output_frames(shape.frames), shape.groups};
        shape.elements(limits);
        for (const auto& block : blocks)
        {
            shape = {block.width, block.upsample.output_frames(shape.frames), shape.groups};
            shape.elements(limits);
        }
        shape = {channels, exit.output_frames(shape.frames), shape.groups};
        shape.elements(limits);
    }
};

template <class Kernels>
DacSynthesis<Kernels>::DacSynthesis(std::shared_ptr<const Impl> impl) : impl_(std::move(impl))
{
}

template <class Kernels>
DacSynthesis<Kernels> DacSynthesis<Kernels>::from_model(DacSynthesisModel model,
                                                        const Limits& limits, Kernels kernels)
{
    const auto conv = [&](const ConvolutionWeights& weights, std::size_t stride = 1,
                          std::size_t dilation = 1, bool transposed = false)
    {
        return kernels.convolution(weights, limits, stride, dilation, transposed);
    };
    auto entry = conv(model.entry_convolution());
    std::vector<synthesis_detail::DecoderBlock<Kernels>> blocks;
    std::size_t width = 1536;
    for (unsigned stage = 0; stage < 4; ++stage)
    {
        const auto stride = std::array<unsigned, 4>{8, 8, 4, 2}[stage];
        auto block_alpha = Alpha(model.resample_activation(stage));
        auto resample = conv(model.resample_convolution(stage), stride, 1, true);
        width /= 2;
        std::vector<synthesis_detail::ResidualUnit<Kernels>> residuals;
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
        blocks.push_back(
            {std::move(block_alpha), std::move(resample), std::move(residuals), width});
    }
    auto final_alpha = Alpha(model.final_activation());
    auto exit = conv(model.exit_convolution());
    auto impl = std::make_shared<Impl>(Impl{kernels, model_contract::channels(model.variant()),
                                            std::move(entry), std::move(blocks),
                                            std::move(final_alpha), std::move(exit)});
    return DacSynthesis<Kernels>(std::move(impl));
}

template <class Kernels> unsigned DacSynthesis<Kernels>::channels() const
{
    return impl_->channels;
}

template <class Kernels> std::size_t DacSynthesis<Kernels>::output_frames(std::size_t latent_frames)
{
    detail::require(latent_frames > 0, "empty synthesis input");
    return checked_mul(latent_frames, hop);
}

template <class Kernels>
Signal DacSynthesis<Kernels>::apply(std::span<const float> latent, SignalShape shape,
                                    const Limits& limits, const SynthesisObserver& observer) const
{
    detail::require(shape.channels == latent_channels, "synthesis input channel mismatch");
    detail::require(latent.size() == shape.elements(limits), "synthesis input shape mismatch");
    impl_->preflight(shape, limits);
    const auto& kernels = impl_->kernels;
    Signal scratch, temporary;
    auto x = impl_->entry.apply(latent, shape, limits);
    synthesis_detail::observe(observer, "decoder.model.0", x);
    for (std::size_t stage = 0; stage < impl_->blocks.size(); ++stage)
    {
        const auto& block = impl_->blocks[stage];
        const auto trace = [&](unsigned operation)
        {
            if (observer)
            {
                observer(std::string("decoder.model.") + std::to_string(stage + 1) + ".block." +
                             std::to_string(operation),
                         x);
            }
        };
        kernels.snake_dac_into(x.values, x.shape, block.alpha, scratch, limits);
        std::swap(x, scratch);
        trace(0);
        block.upsample.apply_into(x.values, x.shape, scratch, limits);
        std::swap(x, scratch);
        trace(1);
        for (std::size_t unit = 0; unit < block.residuals.size(); ++unit)
        {
            block.residuals[unit].apply_into(x, scratch, temporary, limits, observer, stage + 1,
                                             unit + 2, kernels);
            std::swap(x, scratch);
        }
    }
    kernels.snake_dac_into(x.values, x.shape, impl_->final_alpha, scratch, limits);
    std::swap(x, scratch);
    synthesis_detail::observe(observer, "decoder.model.5", x);
    impl_->exit.apply_into(x.values, x.shape, scratch, limits);
    std::swap(x, scratch);
    synthesis_detail::observe(observer, "decoder.model.6", x);
    detail::require(x.shape.frames == output_frames(shape.frames),
                    "synthesis output length mismatch");
    return x;
}

template <class Kernels>
Signal DacSynthesis<Kernels>::apply_chunked(std::span<const float> latent, SignalShape shape,
                                            std::size_t chunk_frames, const Limits& limits,
                                            const SynthesisObserver& observer) const
{
    detail::require(shape.channels == latent_channels, "synthesis input channel mismatch");
    detail::require(latent.size() == shape.elements(limits), "synthesis input shape mismatch");
    detail::require(chunk_frames > 0, "empty synthesis chunk");
    detail::require_finite(latent, "nonfinite synthesis input");
    Signal output{{channels(), output_frames(shape.frames), shape.groups}, {}};
    const auto count = output.shape.elements(limits);
    // Validate the largest possible evaluation before allocating output. Each
    // smaller clipped interval is bounded by the same channel/time products.
    const auto context = checked_mul(context_frames, 2);
    const auto max_frames = chunk_frames >= shape.frames
                                ? shape.frames
                                : std::min(shape.frames, checked_add(chunk_frames, context));
    impl_->preflight({latent_channels, max_frames, shape.groups}, limits);
    output.values.resize(count);
    for (std::size_t begin = 0; begin < shape.frames;)
    {
        const auto central = std::min(chunk_frames, shape.frames - begin);
        const auto end = begin + central;
        const auto left = begin > context_frames ? begin - context_frames : 0;
        const auto right = end + std::min(context_frames, shape.frames - end);
        const auto chunk = detail::signal_window(latent, shape, left, right, limits);
        const auto trace = detail::chunk_observer(observer, "chunk." + std::to_string(begin) + ".");
        const auto evaluated = apply(chunk.values, chunk.shape, limits, trace);
        detail::retain_window(evaluated, output, (begin - left) * hop, begin * hop, central * hop);
        begin = end;
    }
    return output;
}
} // namespace tsac
