#include "format/entropy.hpp"

#include "common/internal.hpp"
#include "format/categorical.hpp"

#include <algorithm>
#include <limits>

namespace tsac
{
std::uint16_t select_entropy_batch(std::size_t frames, int requested)
{
    detail::require(requested >= 0 && requested <= 65535, "invalid entropy batch size");
    if (requested)
    {
        return static_cast<std::uint16_t>(requested);
    }
    auto batch = frames / 1024 + (frames % 1024 != 0);
    if (batch > 8)
    {
        batch = checked_add(batch, 7) & ~std::size_t{7};
    }
    return static_cast<std::uint16_t>(std::clamp<std::size_t>(batch, 1, 128));
}

void visit_entropy_positions(const Header& header,
                             const std::function<void(const EntropyPosition&)>& visit,
                             const Limits& limits)
{
    validate_header(header, limits);
    detail::require(header.method == 1, "entropy traversal requires method 1");
    const auto frames_per_lane = header.frames / header.batch;
    const auto extra = header.frames % header.batch;
    const auto steps = frames_per_lane + (extra != 0);
    for (std::size_t step = 0; step < steps; ++step)
    {
        for (std::size_t book = 0; book < header.codebooks; ++book)
        {
            for (std::size_t lane = 0; lane < std::min<std::size_t>(header.batch, header.frames);
                 ++lane)
            {
                if (step >= frames_per_lane + (lane < extra))
                {
                    continue;
                }
                const auto frame = lane * frames_per_lane + std::min(lane, extra) + step;
                for (std::size_t group = 0; group < header.groups; ++group)
                {
                    visit({frame, group, book, lane, step,
                           (frame * header.groups + group) * header.codebooks + book});
                }
            }
        }
    }
}

Bytes pack_normal(Header header, std::span<const std::uint16_t> tokens, TokenPredictor& predictor,
                  const Limits& limits, const EntropyObserver& observer)
{
    header.method = 1;
    validate_header(header, limits);
    detail::require(header.frames <= std::numeric_limits<std::uint32_t>::max(),
                    "normal frame count exceeds uint32");
    detail::require(tokens.size() == header.symbol_count(),
                    "normal token count does not match header");
    header.token_crc = token_crc32(tokens); // Validate every symbol before any prediction.
    detail::require(limits.file_bytes >= 19, "normal framing exceeds file limit");
    RangeEncoder encoder(limits.file_bytes - 18);
    predictor.reset(header);

    struct ClearKnown
    {
        TokenPredictor& predictor;

        ~ClearKnown()
        {
            predictor.known_tokens({});
        }
    } clear{predictor};

    predictor.known_tokens(tokens);
    visit_entropy_positions(
        header,
        [&](const EntropyPosition& position)
        {
            if (observer.before_symbol)
            {
                observer.before_symbol(position);
            }
            const auto probabilities = predictor.predict(position);
            detail::require(probabilities.size() == 1024, "probability model alphabet mismatch");
            const auto token = tokens[position.token_offset];
            const auto before = observer.after_symbol ? encoder.bit_position() : 0;
            encode_categorical(encoder, probabilities, token);
            predictor.commit(position, token);
            if (observer.after_symbol)
            {
                observer.after_symbol(position, encoder.bit_position() - before);
            }
        },
        limits);
    const auto payload = encoder.finish();
    Bytes bytes{'F',
                'B',
                'A',
                'Z',
                0,
                1,
                static_cast<std::uint8_t>(header.groups | ((header.model_channels - 1) << 7)),
                header.codebooks};
    const auto be = [&](std::size_t value, unsigned count)
    {
        for (unsigned i = count; i > 0; --i)
        {
            bytes.push_back(static_cast<std::uint8_t>(value >> ((i - 1) * 8)));
        }
    };
    be(header.frames, 4);
    be(header.batch, 2);
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    be(header.token_crc, 4);
    return bytes;
}

std::vector<std::uint16_t> unpack_normal(ByteView bytes, TokenPredictor& predictor,
                                         const Limits& limits, const EntropyObserver& observer)
{
    const auto header = parse_header(bytes, limits);
    detail::require(header.method == 1, "unpack_normal requires method 1");
    std::vector<std::uint16_t> tokens(header.symbol_count());
    RangeDecoder decoder(bytes.subspan(header.payload_offset, header.payload_bytes));
    predictor.reset(header);
    visit_entropy_positions(
        header,
        [&](const EntropyPosition& position)
        {
            if (observer.before_symbol)
            {
                observer.before_symbol(position);
            }
            const auto probabilities = predictor.predict(position);
            detail::require(probabilities.size() == 1024, "probability model alphabet mismatch");
            const auto token =
                static_cast<std::uint16_t>(decode_categorical(decoder, probabilities));
            tokens[position.token_offset] = token;
            predictor.commit(position, token);
            if (observer.after_symbol)
            {
                observer.after_symbol(position, 0);
            }
        },
        limits);
    verify_token_crc(header, tokens);
    return tokens;
}
} // namespace tsac
