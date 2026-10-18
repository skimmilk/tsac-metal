#include "model.hpp"

#include "common/numeric.hpp"
#include "models/contract.hpp"

namespace tsac::metal::transformer
{
std::shared_ptr<const PreparedModel> PreparedModel::load(TransformerModel transformer,
                                                         const Limits& limits)
{
    detail::require(available(), "MetalTransformer requires a Metal 4-capable GPU and macOS 26");
    const auto norm = [&](const NormalizationWeights& n)
    {
        return Norm{Buffer::import_weights(weights::View(n.gain)),
                    Buffer::import_weights(weights::View(n.bias))};
    };
    const auto linear = [&](const LinearWeights& weights)
    {
        return MetalLinear::from_weights(weights);
    };
    const auto stack = [&](TransformerStack kind)
    {
        std::vector<Layer> prepared;
        prepared.reserve(model_contract::layers(kind));
        for (unsigned layer = 0; layer < model_contract::layers(kind); ++layer)
        {
            prepared.push_back({norm(transformer.attention_normalization(kind, layer)),
                                norm(transformer.mlp_normalization(kind, layer)),
                                linear(transformer.attention_qkv(kind, layer)),
                                linear(transformer.attention_output(kind, layer)),
                                linear(transformer.mlp_input(kind, layer)),
                                linear(transformer.mlp_output(kind, layer))});
        }
        return Stack{std::move(prepared), norm(transformer.final_normalization(kind))};
    };
    auto constants = weights::View(transformer.positions());
    constants.offset = 0;
    constants.bytes = constants.mapping.size();
    auto result = std::make_shared<PreparedModel>(
        PreparedModel{model_contract::channels(transformer.variant()),
                      transformer.codebooks(),
                      limits,
                      stack(TransformerStack::temporal),
                      stack(TransformerStack::codebook),
                      linear(transformer.temporal_projection()),
                      linear(transformer.codebook_projection()),
                      linear(transformer.output_projection()),
                      Buffer::import_weights(constants),
                      Buffer::import_weights(weights::View(transformer.feedback_embeddings())),
                      Buffer::import_weights(weights::View(transformer.positions())),
                      Buffer::import_weights(weights::View(transformer.rotations())),
                      {}});
    std::vector<Buffer> resources{result->constants, result->embeddings};
    const auto add_projection = [&](const MetalLinear& l)
    {
        const auto buffers = l.resources();
        resources.insert(resources.end(), buffers.begin(), buffers.end());
    };
    for (const auto* s : {&result->temporal, &result->codebook})
    {
        for (const auto& l : s->layers)
        {
            add_projection(l.qkv);
            add_projection(l.attention_out);
            add_projection(l.mlp_in);
            add_projection(l.mlp_out);
        }
    }
    add_projection(result->temporal_projection);
    add_projection(result->codebook_projection);
    add_projection(result->output_projection);
    result->residency = ResidencySet::create(resources);
    return result;
}

} // namespace tsac::metal::transformer
