#pragma once
#include "models/types.hpp"

namespace tsac
{
class TransformerModel
{
public:
    TransformerModel(ModelDirectory directory, ModelVariant variant, unsigned codebooks);
    ModelVariant variant() const;
    unsigned codebooks() const;
    LinearWeights attention_qkv(TransformerStack stack, unsigned layer) const;
    LinearWeights attention_output(TransformerStack stack, unsigned layer) const;
    LinearWeights mlp_input(TransformerStack stack, unsigned layer) const;
    LinearWeights mlp_output(TransformerStack stack, unsigned layer) const;
    LinearWeights temporal_projection() const;
    LinearWeights codebook_projection() const;
    LinearWeights output_projection() const;
    NormalizationWeights attention_normalization(TransformerStack stack, unsigned layer) const;
    NormalizationWeights mlp_normalization(TransformerStack stack, unsigned layer) const;
    NormalizationWeights final_normalization(TransformerStack stack) const;
    TensorView<std::uint16_t> positions() const;
    TensorView<std::uint16_t> rotations() const;
    TensorView<std::uint16_t> feedback_embeddings() const;

private:
    struct State;
    std::shared_ptr<const State> state_;
};
} // namespace tsac
