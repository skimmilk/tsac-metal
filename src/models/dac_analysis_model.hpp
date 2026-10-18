#pragma once
#include "models/types.hpp"

namespace tsac
{
class DacAnalysisModel
{
public:
    DacAnalysisModel(ModelDirectory directory, ModelVariant variant);
    ModelVariant variant() const;
    ConvolutionWeights entry_convolution() const;
    ConvolutionWeights residual_convolution(unsigned stage, unsigned unit,
                                            ResidualConvolution which) const;
    ConvolutionWeights resample_convolution(unsigned stage) const;
    ConvolutionWeights exit_convolution() const;
    TensorView<float> residual_activation(unsigned stage, unsigned unit,
                                          ResidualActivation which) const;
    TensorView<float> resample_activation(unsigned stage) const;
    TensorView<float> final_activation() const;

private:
    struct State;
    std::shared_ptr<const State> state_;
};
} // namespace tsac
