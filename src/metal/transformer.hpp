#pragma once
#include "format/entropy.hpp"
#include "models/transformer_model.hpp"

#include <memory>

namespace tsac
{
// Metal 4 probability-model session. Shares immutable GPU weights; owns caches,
// and one exclusive recording owner. Calls complete before returning; observers
// run in entropy order after their associated GPU work completes. A session may
// be reused after success or failure. Concurrent calls require separate sessions.
class MetalTransformer final
{
public:
    static MetalTransformer from_model(TransformerModel model, const Limits& limits = {});
    ~MetalTransformer();
    MetalTransformer(MetalTransformer&&) noexcept;
    MetalTransformer& operator=(MetalTransformer&&) noexcept;
    MetalTransformer(const MetalTransformer&) = delete;
    MetalTransformer& operator=(const MetalTransformer&) = delete;

    // Independent mutable state sharing the same immutable GPU model/residency.
    MetalTransformer new_session() const;
    // Release stream state, KV caches, scratch and command storage while keeping
    // immutable prepared weights. Execution state is recreated on the next call.
    void clear_session();

    // Normal framing, with causal known-token lookahead and host range encoding.
    Bytes encode(Header header, std::span<const std::uint16_t> tokens,
                 const EntropyObserver& observer = {});
    // Normal framing, with tokens and the serial range state resident on GPU.
    std::vector<std::uint16_t> decode(ByteView stream, const EntropyObserver& observer = {});

private:
    struct Impl;
    explicit MetalTransformer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
} // namespace tsac
