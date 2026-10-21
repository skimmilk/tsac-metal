#pragma once
#include "common/signal.hpp"
#include "models/types.hpp"
#include "weights/tensor_weights.hpp"

#include <memory>

namespace tsac
{
namespace metal
{
class DacGraph;
}

class MetalConvolution
{
public:
    static MetalConvolution prepare(Conv1dConfig config, std::span<const float> weights,
                                    std::span<const float> bias);
    static MetalConvolution from_prepared(Conv1dConfig config, const weights::View& weights,
                                          const weights::View& bias);
    static MetalConvolution from_weights(Conv1dConfig config, const ConvolutionWeights& weights,
                                         const Limits& limits);
    Signal apply(std::span<const float> input, SignalShape shape, const Limits& limits = {}) const;

private:
    friend class metal::DacGraph;
    struct Impl;
    explicit MetalConvolution(std::shared_ptr<const Impl> impl);
    std::shared_ptr<const Impl> impl_;
};

Signal metal_snake(std::span<const float> input, SignalShape shape, std::span<const float> alpha,
                   const Limits& limits = {});
// Eight-wide vectors. Similarity vectors/codebooks already use the captured
// RMS/L2 normalization; ties select the lowest index. Projection uses the
// original unnormalized vectors and folded8x1024 output weights.
std::vector<std::uint16_t> metal_rvq_search(std::span<const float> vectors,
                                            std::span<const float> normalized_codebook,
                                            const Limits& limits = {});
std::vector<float> metal_residual(std::span<const float> left, std::span<const float> right,
                                  bool subtract = false, const Limits& limits = {});
std::vector<std::uint16_t>
metal_rvq_search_raw_projected(std::span<const float> vectors,
                               std::span<const float> normalized_codebook,
                               const Limits& limits = {});
std::vector<float> metal_rvq_project(std::span<const std::uint16_t> tokens,
                                     std::span<const float> codebook,
                                     std::span<const float> folded_weights,
                                     std::span<const float> bias, const Limits& limits = {});
} // namespace tsac
