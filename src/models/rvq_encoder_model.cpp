#include "models/rvq_encoder_model.hpp"

#include "access.hpp"

namespace tsac
{
struct RvqEncoderModel::State
{
    ModelDirectory directory;
    ModelVariant variant;
    unsigned count;
    const std::span<const std::array<model_layout::Range, 7>> locations;
    std::shared_ptr<const MappedRegion> shared;
    RvqInputLayout layout;
    std::shared_ptr<const MappedRegion> search, input;

    State(ModelDirectory d, ModelVariant variant, RvqInputLayout layout, unsigned books)
        : directory(std::move(d)), variant(variant), count(model_access::books(variant, books)),
          locations(model_access::book_locations(variant)), layout(layout)
    {
        model_contract::validate_rvq_input_layout(layout);
        std::size_t shared_end = 0;
        std::size_t search_end = 0, input_end = 0;
        for (unsigned book = 0; book < count; ++book)
        {
            const auto& row = locations[book];
            for (unsigned column : {0u, 1u, 2u})
            {
                shared_end = std::max(shared_end, model_access::end(row[column]));
            }
            search_end =
                std::max({search_end, model_access::end(row[3]), model_access::end(row[4])});
            input_end = std::max(
                input_end, model_access::end(row[layout != RvqInputLayout::output_major ? 5 : 6]));
        }
        shared = model_access::prefix(
            directory, model_access::file(variant, ModelFile::mono_rvq_shared), shared_end);
        search = model_access::prefix(
            directory, model_access::file(variant, ModelFile::mono_rvq_encode_shared), search_end);
        input = model_access::prefix(
            directory,
            model_access::file(variant, layout != RvqInputLayout::output_major
                                            ? ModelFile::mono_rvq_encode_apple
                                            : ModelFile::mono_rvq_encode_metal),
            input_end);
    }
};

RvqEncoderModel::RvqEncoderModel(ModelDirectory directory, ModelVariant variant,
                                 RvqInputLayout layout, unsigned codebooks)
    : state_(std::make_shared<State>(std::move(directory), variant, layout, codebooks))
{
}

ModelVariant RvqEncoderModel::variant() const
{
    return state_->variant;
}

RvqInputLayout RvqEncoderModel::input_layout() const
{
    return state_->layout;
}

unsigned RvqEncoderModel::codebooks() const
{
    return state_->count;
}

TensorView<float> RvqEncoderModel::codebook(unsigned book) const
{
    model_access::book(book, state_->count);
    return model_access::tensor<float>(state_->shared, state_->locations[book][0],
                                       model_contract::rvq_vectors);
}

FloatProjectionWeights RvqEncoderModel::output_projection(unsigned book) const
{
    model_access::book(book, state_->count);
    return {model_access::tensor<float>(state_->shared, state_->locations[book][1],
                                        model_contract::rvq_output_weights),
            model_access::tensor<float>(state_->shared, state_->locations[book][2],
                                        model_contract::rvq_output_bias)};
}

TensorView<float> RvqEncoderModel::search_vectors(unsigned book) const
{
    model_access::book(book, state_->count);
    return model_access::tensor<float>(state_->search, state_->locations[book][3],
                                       model_contract::rvq_vectors);
}

FloatProjectionWeights RvqEncoderModel::input_projection(unsigned book) const
{
    model_access::book(book, state_->count);
    const bool input_major = state_->layout == RvqInputLayout::input_major;
    return {model_access::tensor<float>(state_->input, state_->locations[book][input_major ? 5 : 6],
                                        model_contract::rvq_input_shape(state_->layout)),
            model_access::tensor<float>(state_->search, state_->locations[book][4],
                                        model_contract::rvq_input_bias)};
}
} // namespace tsac
