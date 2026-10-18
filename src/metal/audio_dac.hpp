#pragma once
#include "metal/dac_graph.hpp"
#include "models/dac_analysis_model.hpp"
#include "models/dac_synthesis_model.hpp"

#include <variant>

namespace tsac::metal
{
// GPU-only topology. Prepared operations own their GPU resources; no CPU DSP
// objects, execution switches or per-layer host readbacks are retained.
class AudioDac
{
public:
    static AudioDac from_model(DacAnalysisModel model, const Limits& limits);
    static AudioDac from_model(DacSynthesisModel model, const Limits& limits);
    Signal apply(std::span<const float> input, SignalShape shape, const Limits& limits) const;
    // Append to the caller's graph; the returned tensor shares its backing storage.
    DacTensor record(DacGraph& graph, const DacTensor& input, const Limits& limits) const;

private:
    struct Activation
    {
        Buffer alpha;
    };

    struct SaveResidual
    {
    };

    struct AddResidual
    {
    };

    using Operation = std::variant<MetalConvolution, Activation, SaveResidual, AddResidual>;
    std::vector<Operation> operations_;
    unsigned channels_ = 0;
    bool synthesis_ = false;

    void convolution(const ConvolutionWeights& weights, const Limits& limits,
                     std::size_t stride = 1, std::size_t dilation = 1, bool transposed = false);
    void activation(const TensorView<float>& alpha);
    template <class Model>
    void residual(const Model& model, unsigned stage, unsigned unit, const Limits& limits);
    void preflight(SignalShape shape, const Limits& limits) const;
    struct Recorder;
};
} // namespace tsac::metal
