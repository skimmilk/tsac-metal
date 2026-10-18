#include "format/container.hpp"

#include "common/internal.hpp"

#include <algorithm>

namespace tsac
{
void validate_header(const Header& header, const Limits& limits)
{
    detail::require(header.method <= 1, "unsupported method");
    detail::require(header.groups > 0 && header.groups <= 127, "invalid channel groups");
    detail::require(header.model_channels == 1 || header.model_channels == 2,
                    "invalid model channels");
    detail::require(header.codebooks > 0 &&
                        header.codebooks <= (header.model_channels == 1 ? 9 : 12),
                    "invalid codebook count");
    detail::require(header.method == 0 || header.batch > 0, "zero entropy batch size");
    detail::require(header.symbol_count() <= limits.symbols, "symbol count exceeds limit");
    detail::require(header.pcm16_bytes() <= limits.pcm_bytes, "decoded PCM size exceeds limit");
}

std::size_t Header::symbol_count() const
{
    return checked_mul(checked_mul(frames, groups), codebooks);
}

std::size_t Header::channels() const
{
    return checked_mul(groups, model_channels);
}

std::size_t Header::samples_per_channel() const
{
    return checked_mul(frames, 512);
}

std::size_t Header::pcm16_bytes() const
{
    return checked_mul(checked_mul(samples_per_channel(), channels()), 2);
}

Header parse_header(ByteView bytes, const Limits& limits)
{
    detail::require(bytes.size() <= limits.file_bytes, "input exceeds file size limit");
    detail::require(detail::be32(bytes, 0) == 0x4642415a, "invalid FBAZ magic");
    detail::slice(bytes, 0, 8);
    detail::require(bytes[4] == 0, "unsupported TSAC version");
    Header header;
    header.method = bytes[5];
    header.groups = bytes[6] & 127;
    header.model_channels = 1 + (bytes[6] >> 7);
    header.codebooks = bytes[7];
    detail::require(header.method <= 1, "unsupported TSAC method");
    detail::require(header.groups && header.codebooks, "zero groups or codebooks");
    if (header.method == 0)
    {
        header.payload_bytes = bytes.size() - 8;
        const auto bits_per_frame = checked_mul(checked_mul(header.groups, header.codebooks), 10);
        const auto bits = checked_mul(header.payload_bytes, 8);
        header.frames = bits / bits_per_frame;
        const auto padding = bits % bits_per_frame;
        detail::require(padding < 8, "incomplete fast frame or excess padding");
        if (padding)
        {
            detail::require((bytes.back() & ((1u << padding) - 1)) == 0, "nonzero fast padding");
        }
    }
    else
    {
        detail::require(bytes.size() >= 19, "truncated normal stream");
        header.frames = detail::be32(bytes, 8);
        header.batch = static_cast<std::uint16_t>((std::uint16_t(bytes[12]) << 8) | bytes[13]);
        header.payload_offset = 14;
        header.payload_bytes = bytes.size() - 18;
        header.token_crc = detail::be32(bytes, bytes.size() - 4);
    }
    validate_header(header, limits);
    return header;
}

std::vector<std::uint16_t> unpack_fast(ByteView bytes, const Limits& limits)
{
    const auto header = parse_header(bytes, limits);
    detail::require(header.method == 0, "normal-mode unpacking requires the probability model");
    std::vector<std::uint16_t> indices;
    indices.reserve(header.symbol_count());
    std::uint32_t buffer = 0;
    unsigned bits = 0;
    for (auto byte : bytes.subspan(8))
    {
        buffer = (buffer << 8) | byte;
        bits += 8;
        while (bits >= 10)
        {
            bits -= 10;
            indices.push_back(static_cast<std::uint16_t>((buffer >> bits) & 1023));
        }
        buffer &= (1u << bits) - 1;
    }
    detail::require(indices.size() == header.symbol_count(), "fast symbol count mismatch");
    return indices;
}

Bytes pack_fast(const Header& header, std::span<const std::uint16_t> indices, const Limits& limits)
{
    validate_header(header, limits);
    detail::require(header.method == 0, "pack_fast requires method 0");
    detail::require(indices.size() == header.symbol_count(), "symbol count does not match header");
    const auto payload = checked_add(checked_mul(indices.size(), 10), 7) / 8;
    detail::require(checked_add(payload, 8) <= limits.file_bytes, "output exceeds file size limit");
    const auto channel_flags =
        static_cast<std::uint8_t>(header.groups | ((header.model_channels - 1) << 7));
    Bytes bytes{'F', 'B', 'A', 'Z', 0, 0, channel_flags, header.codebooks};
    bytes.reserve(payload + 8);
    std::uint32_t buffer = 0;
    unsigned bits = 0;
    for (auto index : indices)
    {
        detail::require(index < 1024, "index outside 10-bit alphabet");
        buffer = (buffer << 10) | index;
        bits += 10;
        while (bits >= 8)
        {
            bits -= 8;
            bytes.push_back(static_cast<std::uint8_t>(buffer >> bits));
        }
        buffer &= (1u << bits) - 1;
    }
    if (bits)
    {
        bytes.push_back(static_cast<std::uint8_t>(buffer << (8 - bits)));
    }
    return bytes;
}

std::uint32_t token_crc32(std::span<const std::uint16_t> indices)
{
    std::uint32_t crc = 0xffffffff;
    for (auto index : indices)
    {
        detail::require(index < 1024, "index outside 10-bit alphabet");
        for (unsigned shift = 0; shift < 32; shift += 8)
        {
            crc ^= ((std::uint32_t(index) >> shift) & 255) << 24;
            for (unsigned bit = 0; bit < 8; ++bit)
            {
                crc = (crc << 1) ^ ((crc & 0x80000000) ? 0x04c11db7 : 0);
            }
        }
    }
    return crc;
}

void verify_token_crc(const Header& header, std::span<const std::uint16_t> indices)
{
    detail::require(header.method == 1 && indices.size() == header.symbol_count(),
                    "CRC requires a normal header and complete indices");
    detail::require(token_crc32(indices) == header.token_crc, "invalid token CRC32");
}

void visit_entropy_order(const Header& header, const std::function<void(std::size_t)>& visit,
                         const Limits& limits)
{
    validate_header(header, limits);
    detail::require(header.method == 1, "entropy traversal requires method 1");
    const auto frames_per_lane = header.frames / header.batch;
    const auto extra_frames = header.frames % header.batch;
    const auto steps = frames_per_lane + (extra_frames != 0);
    // Empty lanes (B > T) never visit a symbol and need not be iterated here.
    const auto lanes = std::min<std::size_t>(header.batch, header.frames);
    for (std::size_t step = 0; step < steps; ++step)
    {
        for (std::size_t book = 0; book < header.codebooks; ++book)
        {
            for (std::size_t lane = 0; lane < lanes; ++lane)
            {
                if (step >= frames_per_lane + (lane < extra_frames))
                {
                    continue;
                }
                const auto start = lane * frames_per_lane + std::min(lane, extra_frames);
                for (std::size_t group = 0; group < header.groups; ++group)
                {
                    visit(((start + step) * header.groups + group) * header.codebooks + book);
                }
            }
        }
    }
}
} // namespace tsac
