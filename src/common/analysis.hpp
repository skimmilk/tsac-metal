#pragma once
#include "common/dsp.hpp"
#include "models/dac_analysis_model.hpp"

#include <functional>
#include <memory>
#include <string_view>

namespace tsac
{
using AnalysisObserver = std::function<void(std::string_view, const Signal&)>;

// Fixed DAC analysis pipeline. PCM is normalized float32, with channels fastest,
// and must already be zero-padded to a multiple of 512 samples per group.
template <class Kernels> class DacAnalysis
{
public:
    static constexpr std::size_t latent_channels = 1024, hop = 512;
    static constexpr std::size_t left_context_frames = 8, right_context_frames = 9;
    static constexpr std::size_t default_chunk_frames = 100;
    static DacAnalysis from_model(DacAnalysisModel model, const Limits& limits = {},
                                  Kernels kernels = {});
    unsigned channels() const;
    static std::size_t output_frames(std::size_t pcm_frames);
    // [model_channels,pcm_frames,groups] -> [1024,pcm_frames/512,groups].
    Signal apply(std::span<const float> pcm, SignalShape shape, const Limits& limits = {},
                 const AnalysisObserver& observer = {}) const;
    // Evaluate one independent group at a time with bounded contextual chunks.
    // Observer names are group.<group>.chunk.<central latent offset>.<module>.
    Signal apply_chunked(std::span<const float> pcm, SignalShape shape,
                         std::size_t chunk_frames = default_chunk_frames, const Limits& limits = {},
                         const AnalysisObserver& observer = {}) const;

private:
    struct Impl;
    explicit DacAnalysis(std::shared_ptr<const Impl> impl);
    std::shared_ptr<const Impl> impl_;
};
} // namespace tsac
