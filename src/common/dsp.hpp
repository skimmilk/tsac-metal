#pragma once
#include "common/signal.hpp"
#include "weights/normalization.hpp"
#include "weights/tensor_weights.hpp"

#include <bit>
#include <memory>
#include <optional>

namespace tsac
{
struct ConvolutionWeights;

// Geometry, validation and model preparation shared by concrete CPU convolutions.
// Stored axes are [input,kernel,output] (ordinary) or [output,kernel,input].
template <class Derived> class Convolution
{
public:
    // Normalize contiguous columns over the first two stored axes, including
    // [output_channels,kernel] for transposed weights. Gain is per stored column.
    static Derived weight_normalized(Conv1dConfig config, std::span<const float> v,
                                     std::span<const float> g, std::span<const float> bias,
                                     const Limits& limits = {});
    static Derived from_prepared(Conv1dConfig config, const weights::View& weights,
                                 const weights::View& bias, const Limits& limits);
    static Derived from_weights(const ConvolutionWeights& conv, const Limits& limits,
                                std::size_t stride = 1, std::size_t dilation = 1,
                                bool transposed = false);
    std::size_t output_frames(std::size_t input_frames) const;
    // Scalar sums in tap/channel order; Apple uses BLAS reductions. Both add
    // bias once after all products. Out-of-bounds input is zero. Owns weights.
    Signal apply(std::span<const float> input, SignalShape shape, const Limits& limits = {}) const;
    // Input must not overlap output storage. Reuses output capacity.
    void apply_into(std::span<const float> input, SignalShape shape, Signal& output,
                    const Limits& limits = {}) const;

protected:
    std::optional<std::size_t> source_frame(std::size_t frame, std::size_t tap) const;
    Conv1dConfig config_;
    std::vector<float> weights_;
    weights::Array<float> bias_;

private:
    friend Derived;
    Convolution() = default;
    // Weights are already folded; bias has exactly output_channels elements.
    Convolution(Conv1dConfig config, std::span<const float> weights, std::span<const float> bias,
                const Limits& limits = {});
};

} // namespace tsac

#include "common/dsp.tpp"
