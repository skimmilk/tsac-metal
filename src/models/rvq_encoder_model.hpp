#pragma once
#include "models/types.hpp"

namespace tsac
{
class RvqEncoderModel
{
public:
    RvqEncoderModel(ModelDirectory directory, ModelVariant variant, RvqInputLayout layout,
                    unsigned codebooks);
    ModelVariant variant() const;
    RvqInputLayout input_layout() const;
    unsigned codebooks() const;
    TensorView<float> codebook(unsigned book) const;
    FloatProjectionWeights output_projection(unsigned book) const;
    TensorView<float> search_vectors(unsigned book) const;
    FloatProjectionWeights input_projection(unsigned book) const;

private:
    struct State;
    std::shared_ptr<const State> state_;
};
} // namespace tsac
