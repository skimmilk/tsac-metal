#pragma once
#include "tsac/observer.hpp"

#include "format/container.hpp"

namespace tsac
{
// Stateful predictor boundary. All lanes are independent; symbols still share
// one arithmetic stream in step/codebook/lane/group order. predict() borrows
// a 1024-entry distribution until commit() or the next predict(). reset() must
// clear all caches, and commit() supplies exactly one selected current token.
class TokenPredictor
{
public:
    virtual ~TokenPredictor() = default;
    virtual void reset(const Header& header) = 0;

    // Encoder-only borrowed tokens; cleared before pack_normal returns, including
    // exceptions. Backends may use bounded lookahead without changing stream order.
    virtual void known_tokens(std::span<const std::uint16_t>)
    {
    }

    virtual std::span<const float> predict(const EntropyPosition& position) = 0;
    virtual void commit(const EntropyPosition& position, std::uint16_t token) = 0;
};

std::uint16_t select_entropy_batch(std::size_t frames, int requested = 0);
void visit_entropy_positions(const Header& header,
                             const std::function<void(const EntropyPosition&)>& visit,
                             const Limits& limits = {});
Bytes pack_normal(Header header, std::span<const std::uint16_t> tokens, TokenPredictor& predictor,
                  const Limits& limits = {}, const EntropyObserver& observer = {});
// CRC is checked before returning tokens to any synthesis/output operation.
std::vector<std::uint16_t> unpack_normal(ByteView bytes, TokenPredictor& predictor,
                                         const Limits& limits = {},
                                         const EntropyObserver& observer = {});
} // namespace tsac
