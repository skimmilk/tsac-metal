#pragma once
#include "common/dsp.hpp"
#include "models/dac_synthesis_model.hpp"

#include <functional>
#include <memory>
#include <string_view>

namespace tsac
{
// Called synchronously after every convolution, Snake and residual addition.
// Names are original decoder tensor/module prefixes. Signals are borrowed only
// for the duration of the callback. Chunked evaluation prefixes each name with
// "chunk.<central latent offset>."; the signal includes that chunk's context.
using SynthesisObserver = std::function<void(std::string_view, const Signal&)>;

template <class Kernels> class DacSynthesis
{
public:
    static constexpr std::size_t latent_channels = 1024, hop = 512;
    static constexpr std::size_t context_frames = 11, default_chunk_frames = 100;

    // Validates the supported DAC tensor schema. All prepared weights and
    // activations are owned, so the source adapter can be released afterward.
    static DacSynthesis from_model(DacSynthesisModel model, const Limits& limits = {},
                                   Kernels kernels = {});
    unsigned channels() const;
    static std::size_t output_frames(std::size_t latent_frames);

    // [1024,latent_frames,groups] -> [model_channels,latent_frames*512,groups].
    // Every group is an independent recording. No final tanh or PCM rounding.
    Signal apply(std::span<const float> latent, SignalShape shape, const Limits& limits = {},
                 const SynthesisObserver& observer = {}) const;
    // Evaluate central chunks with up to eleven surrounding latent frames on
    // either side, clipped at recording boundaries, then retain central PCM.
    Signal apply_chunked(std::span<const float> latent, SignalShape shape,
                         std::size_t chunk_frames = default_chunk_frames, const Limits& limits = {},
                         const SynthesisObserver& observer = {}) const;

private:
    struct Impl;
    explicit DacSynthesis(std::shared_ptr<const Impl> impl);
    std::shared_ptr<const Impl> impl_;
};
} // namespace tsac
