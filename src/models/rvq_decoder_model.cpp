#include "models/rvq_decoder_model.hpp"

#include "access.hpp"

namespace tsac
{
struct RvqDecoderModel::State
{
    ModelDirectory directory;
    ModelVariant variant;
    unsigned count;
    const std::span<const std::array<model_layout::Range, 7>> locations;
    std::shared_ptr<const MappedRegion> shared;

    State(ModelDirectory d, ModelVariant variant, unsigned books)
        : directory(std::move(d)), variant(variant), count(model_access::books(variant, books)),
          locations(model_access::book_locations(variant))
    {

        std::size_t shared_end = 0;

        for (unsigned book = 0; book < count; ++book)
        {
            const auto& row = locations[book];
            for (unsigned column : {0u, 1u, 2u})
            {
                shared_end = std::max(shared_end, model_access::end(row[column]));
            }
        }
        shared = model_access::prefix(
            directory, model_access::file(variant, ModelFile::mono_rvq_shared), shared_end);
    }
};

RvqDecoderModel::RvqDecoderModel(ModelDirectory directory, ModelVariant variant, unsigned codebooks)
    : state_(std::make_shared<State>(std::move(directory), variant, codebooks))
{
}

ModelVariant RvqDecoderModel::variant() const
{
    return state_->variant;
}

unsigned RvqDecoderModel::codebooks() const
{
    return state_->count;
}

TensorView<float> RvqDecoderModel::codebook(unsigned book) const
{
    model_access::book(book, state_->count);
    return model_access::tensor<float>(state_->shared, state_->locations[book][0],
                                       model_contract::rvq_vectors);
}

FloatProjectionWeights RvqDecoderModel::output_projection(unsigned book) const
{
    model_access::book(book, state_->count);
    return {model_access::tensor<float>(state_->shared, state_->locations[book][1],
                                        model_contract::rvq_output_weights),
            model_access::tensor<float>(state_->shared, state_->locations[book][2],
                                        model_contract::rvq_output_bias)};
}

} // namespace tsac
