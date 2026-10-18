#pragma once
#include "common/common.hpp"

#include <functional>

namespace tsac
{
struct Header
{
    std::uint8_t method = 0;
    std::uint8_t groups = 1;
    std::uint8_t model_channels = 1;
    std::uint8_t codebooks = 1;
    std::size_t frames = 0;
    std::uint16_t batch = 1;
    std::size_t payload_offset = 8;
    std::size_t payload_bytes = 0;
    std::uint32_t token_crc = 0; // Only present in method 1.
    std::size_t symbol_count() const;
    std::size_t channels() const;
    std::size_t samples_per_channel() const;
    std::size_t pcm16_bytes() const;
};

// Validate dimensions and policy limits before allocating indices or PCM.
void validate_header(const Header& header, const Limits& limits = {});
Header parse_header(ByteView bytes, const Limits& limits = {});
std::vector<std::uint16_t> unpack_fast(ByteView bytes, const Limits& limits = {});
// Header dimensions must agree with the entire time/group/codebook-major array.
Bytes pack_fast(const Header& header, std::span<const std::uint16_t> indices,
                const Limits& limits = {});
std::uint32_t token_crc32(std::span<const std::uint16_t> indices);
void verify_token_crc(const Header& header, std::span<const std::uint16_t> indices);
// Callback receives the flat time/group/codebook-major index in entropy order.
void visit_entropy_order(const Header& header, const std::function<void(std::size_t)>& visit,
                         const Limits& limits = {});
} // namespace tsac
