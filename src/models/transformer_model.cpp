#include "models/transformer_model.hpp"

#include "access.hpp"

namespace tsac
{
struct TransformerModel::State
{
    ModelDirectory directory;
    ModelVariant variant;
    unsigned count;
    const model_layout::Transformer& locations;
    std::shared_ptr<const MappedRegion> source, shared, embeddings;

    State(ModelDirectory d, ModelVariant variant, unsigned books)
        : directory(std::move(d)), variant(variant), count(model_access::books(variant, books)),
          locations(model_contract::channels(variant) == 1 ? adapter_generated::mono_transformer
                                                           : adapter_generated::stereo_transformer)
    {
        if (!adapter_generated::has_transformer)
        {
            throw Error("Transformer assets were disabled");
        }
        std::size_t source_end = 0, shared_end = 0;
        for (const auto& row : locations.projections)
        {
            source_end = std::max(source_end, model_access::end(row[0]));
            shared_end =
                std::max({shared_end, model_access::end(row[1]), model_access::end(row[2])});
        }
        for (const auto& row : locations.norms)
        {
            for (auto r : row)
            {
                shared_end = std::max(shared_end, model_access::end(r));
            }
        }
        for (auto r : locations.constants)
        {
            shared_end = std::max(shared_end, model_access::end(r));
        }
        source = model_access::prefix(
            directory, model_access::file(variant, ModelFile::mono_transformer_source), source_end);
        const auto file = model_access::file(variant, ModelFile::mono_transformer_shared);
        shared = model_access::prefix(directory, file, shared_end);
        embeddings = detail::ModelDirectoryAccess::map(
            directory, file, locations.embeddings.offset,
            checked_mul(count, model_contract::embedding_book_bytes));
    }

    LinearWeights projection(unsigned index) const
    {
        const auto shape = model_contract::projection_shape(index);
        const auto& row = locations.projections[index];
        return {model_access::tensor<std::int8_t>(source, row[0], shape),
                model_access::tensor<float>(shared, row[1], {{shape.axes[1], shape.axes[0] / 256}}),
                model_access::tensor<float>(shared, row[2], {{shape.axes[1]}})};
    }

    NormalizationWeights norm(unsigned index) const
    {
        const auto& row = locations.norms[index];
        return {model_access::tensor<std::uint16_t>(shared, row[0], model_contract::normalization),
                model_access::tensor<std::uint16_t>(shared, row[1], model_contract::normalization)};
    }
};

TransformerModel::TransformerModel(ModelDirectory directory, ModelVariant variant,
                                   unsigned codebooks)
    : state_(std::make_shared<State>(std::move(directory), variant, codebooks))
{
}

ModelVariant TransformerModel::variant() const
{
    return state_->variant;
}

unsigned TransformerModel::codebooks() const
{
    return state_->count;
}

LinearWeights TransformerModel::attention_qkv(TransformerStack stack, unsigned layer) const
{
    return state_->projection(model_contract::projection_index(stack, layer, 0));
}

LinearWeights TransformerModel::attention_output(TransformerStack stack, unsigned layer) const
{
    return state_->projection(model_contract::projection_index(stack, layer, 1));
}

LinearWeights TransformerModel::mlp_input(TransformerStack stack, unsigned layer) const
{
    return state_->projection(model_contract::projection_index(stack, layer, 2));
}

LinearWeights TransformerModel::mlp_output(TransformerStack stack, unsigned layer) const
{
    return state_->projection(model_contract::projection_index(stack, layer, 3));
}

LinearWeights TransformerModel::temporal_projection() const
{
    return state_->projection(60);
}

LinearWeights TransformerModel::codebook_projection() const
{
    return state_->projection(61);
}

LinearWeights TransformerModel::output_projection() const
{
    return state_->projection(62);
}

NormalizationWeights TransformerModel::attention_normalization(TransformerStack stack,
                                                               unsigned layer) const
{
    return state_->norm(model_contract::normalization_index(stack, layer, false));
}

NormalizationWeights TransformerModel::mlp_normalization(TransformerStack stack,
                                                         unsigned layer) const
{
    return state_->norm(model_contract::normalization_index(stack, layer, true));
}

NormalizationWeights TransformerModel::final_normalization(TransformerStack stack) const
{
    return state_->norm(model_contract::final_normalization_index(stack));
}

TensorView<std::uint16_t> TransformerModel::positions() const
{
    auto range = state_->locations.constants[0];
    range.bytes = std::size_t{512} * state_->count * 2;
    return model_access::tensor<std::uint16_t>(state_->shared, range,
                                               model_contract::positions_shape(state_->count));
}

TensorView<std::uint16_t> TransformerModel::rotations() const
{
    return model_access::tensor<std::uint16_t>(state_->shared, state_->locations.constants[1],
                                               model_contract::rotary);
}

TensorView<std::uint16_t> TransformerModel::feedback_embeddings() const
{
    const model_layout::Range range{
        state_->locations.embeddings.offset,
        checked_mul(state_->count, model_contract::embedding_book_bytes)};
    return model_access::tensor<std::uint16_t>(state_->embeddings, range,
                                               model_contract::feedback_shape(state_->count));
}
} // namespace tsac
