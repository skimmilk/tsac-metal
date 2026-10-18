#pragma once
#include "models/types.hpp"

namespace tsac
{
class RvqDecoderModel
{
public:
    RvqDecoderModel(ModelDirectory directory, ModelVariant variant, unsigned codebooks);
    ModelVariant variant() const;
    unsigned codebooks() const;
    TensorView<float> codebook(unsigned book) const;
    FloatProjectionWeights output_projection(unsigned book) const;

private:
    struct State;
    std::shared_ptr<const State> state_;
};
} // namespace tsac
