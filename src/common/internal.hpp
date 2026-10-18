#pragma once
#include "common/common.hpp"

#include <bit>
#include <limits>

namespace tsac::detail
{
inline void require(bool condition, const char* message)
{
    if (!condition)
    {
        throw Error(message);
    }
}

inline ByteView slice(ByteView data, std::size_t offset, std::size_t size)
{
    require(offset <= data.size() && size <= data.size() - offset, "truncated input");
    return data.subspan(offset, size);
}

inline std::uint32_t le32(ByteView data, std::size_t offset)
{
    auto p = slice(data, offset, 4);
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) | (std::uint32_t(p[2]) << 16) |
           (std::uint32_t(p[3]) << 24);
}

inline std::uint32_t be32(ByteView data, std::size_t offset)
{
    auto p = slice(data, offset, 4);
    return (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16) | (std::uint32_t(p[2]) << 8) |
           std::uint32_t(p[3]);
}

inline float bf16(ByteView data, std::size_t offset)
{
    auto p = slice(data, offset, 2);
    return std::bit_cast<float>((std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8)) << 16);
}

inline std::size_t align256(std::size_t size)
{
    return checked_add(size, 255) / 256 * 256;
}

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
} // namespace tsac::detail
