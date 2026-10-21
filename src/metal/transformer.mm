#include "metal/transformer.hpp"

#include "metal/transformer/session.hpp"

namespace tsac
{
struct MetalTransformer::Impl
{
    explicit Impl(std::shared_ptr<const metal::transformer::PreparedModel> model)
        : prepared(std::move(model)), state(std::make_unique<metal::transformer::Session>(prepared))
    {
    }

    metal::transformer::Session& session()
    {
        if (!state)
        {
            state = std::make_unique<metal::transformer::Session>(prepared);
        }
        return *state;
    }

    std::shared_ptr<const metal::transformer::PreparedModel> prepared;
    std::unique_ptr<metal::transformer::Session> state;
};

MetalTransformer::MetalTransformer(std::unique_ptr<Impl> impl) : impl_(std::move(impl))
{
}

MetalTransformer::~MetalTransformer() = default;
MetalTransformer::MetalTransformer(MetalTransformer&&) noexcept = default;
MetalTransformer& MetalTransformer::operator=(MetalTransformer&&) noexcept = default;

MetalTransformer MetalTransformer::from_model(TransformerModel model, const Limits& limits)
{
    return MetalTransformer(
        std::make_unique<Impl>(metal::transformer::PreparedModel::load(std::move(model), limits)));
}

MetalTransformer MetalTransformer::new_session() const
{
    return MetalTransformer(std::make_unique<Impl>(impl_->prepared));
}

void MetalTransformer::clear_session()
{
    impl_->state.reset();
}

Bytes MetalTransformer::encode(Header header, std::span<const std::uint16_t> tokens,
                               const EntropyObserver& observer)
{
    return impl_->session().encode(header, tokens, observer);
}

std::vector<std::uint16_t> MetalTransformer::decode(ByteView stream,
                                                    const EntropyObserver& observer)
{
    return impl_->session().decode(stream, observer);
}
} // namespace tsac
