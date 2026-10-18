#include "models/dac_synthesis_model.hpp"

#include "access.hpp"

namespace tsac
{
struct DacSynthesisModel::State
{
    ModelDirectory directory;
    ModelVariant variant;
    const model_layout::Dac& locations;
    std::shared_ptr<const MappedRegion> source, shared;

    State(ModelDirectory d, ModelVariant v)
        : directory(std::move(d)), variant(v),
          locations(model_contract::channels(v) == 1 ? adapter_generated::mono_synthesis
                                                     : adapter_generated::stereo_synthesis)
    {
        source = model_access::prefix(directory,
                                      model_access::file(v, ModelFile::mono_dac_synthesis_source),
                                      model_access::extent({locations.values, locations.gains}));
        shared = model_access::prefix(directory,
                                      model_access::file(v, ModelFile::mono_dac_synthesis_shared),
                                      model_access::extent({locations.biases, locations.alphas}));
    }

    ConvolutionWeights convolution(unsigned index) const
    {
        const auto shape = model_contract::dac_shape(true, variant, index);
        const auto r = locations.values[index];
        const bool transposed = true && index > 0 && index < 29 && (index - 1) % 7 == 0;
        return {
            EncodedTensorView(source, model_access::relative(*source, r), r.bytes, shape,
                              model_contract::dac_encoding(true, index)),
            model_access::tensor<float>(source, locations.gains[index], {{1, 1, shape.axes[2]}}),
            model_access::tensor<float>(shared, locations.biases[index],
                                        {{shape.axes[transposed ? 0 : 2]}})};
    }

    TensorView<float> activation(unsigned index) const
    {
        return model_access::tensor<float>(shared, locations.alphas[index],
                                           model_contract::activation_shape(true, index));
    }
};

DacSynthesisModel::DacSynthesisModel(ModelDirectory directory, ModelVariant variant)
    : state_(std::make_shared<State>(std::move(directory), variant))
{
}

ModelVariant DacSynthesisModel::variant() const
{
    return state_->variant;
}

ConvolutionWeights DacSynthesisModel::entry_convolution() const
{
    return state_->convolution(0);
}

ConvolutionWeights DacSynthesisModel::exit_convolution() const
{
    return state_->convolution(29);
}

ConvolutionWeights DacSynthesisModel::residual_convolution(unsigned stage, unsigned unit,
                                                           ResidualConvolution which) const
{
    return state_->convolution(model_contract::residual_convolution(true, stage, unit, which));
}

ConvolutionWeights DacSynthesisModel::resample_convolution(unsigned stage) const
{
    return state_->convolution(model_contract::resample_convolution(true, stage));
}

TensorView<float> DacSynthesisModel::residual_activation(unsigned stage, unsigned unit,
                                                         ResidualActivation which) const
{
    return state_->activation(model_contract::residual_activation(true, stage, unit, which));
}

TensorView<float> DacSynthesisModel::resample_activation(unsigned stage) const
{
    return state_->activation(model_contract::resample_activation(true, stage));
}

TensorView<float> DacSynthesisModel::final_activation() const
{
    return state_->activation(28);
}
} // namespace tsac
