#include "common/common.hpp"

#include "common/internal.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace tsac
{
std::size_t checked_add(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
    {
        throw Error("size addition overflow");
    }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b)
{
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
    {
        throw Error("size multiplication overflow");
    }
    return a * b;
}

Bytes read_file(const std::filesystem::path& path, const Limits& limits)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw Error("cannot open input: " + path.string());
    }
    const auto end = file.tellg();
    if (end < 0 || static_cast<std::uintmax_t>(end) > limits.file_bytes ||
        static_cast<std::uintmax_t>(end) >
            static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max()))
    {
        throw Error("input exceeds file size limit");
    }
    Bytes bytes(static_cast<std::size_t>(end));
    file.seekg(0);
    if (!bytes.empty() && !file.read(reinterpret_cast<char*>(bytes.data()),
                                     static_cast<std::streamsize>(bytes.size())))
    {
        throw Error("cannot read complete input");
    }
    // A growing file should not silently become a different input.
    if (file.peek() != std::char_traits<char>::eof())
    {
        throw Error("input changed while reading");
    }
    return bytes;
}

void write_file(const std::filesystem::path& path, ByteView bytes)
{
    detail::require(bytes.size() <=
                        static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()),
                    "output too large");
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        throw Error("cannot open output: " + path.string());
    }
    if (!bytes.empty())
    {
        file.write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
    }
    file.close();
    if (!file)
    {
        throw Error("cannot write complete output");
    }
}

std::string sha256(ByteView bytes)
{
    detail::require(bytes.size() <= std::numeric_limits<std::uint64_t>::max() / 8,
                    "SHA256 input too large");
    constexpr std::array<std::uint32_t, 64> constants{
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};
    std::array<std::uint32_t, 8> state{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                       0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    const auto compress = [&](ByteView block)
    {
        std::array<std::uint32_t, 64> words;
        for (std::size_t i = 0; i < 16; ++i)
        {
            words[i] = detail::be32(block, i * 4);
        }
        for (std::size_t i = 16; i < 64; ++i)
        {
            const auto x = words[i - 15], y = words[i - 2];
            const auto s0 = std::rotr(x, 7) ^ std::rotr(x, 18) ^ (x >> 3);
            const auto s1 = std::rotr(y, 17) ^ std::rotr(y, 19) ^ (y >> 10);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }
        auto [a, b, c, d, e, f, g, h] = state;
        for (std::size_t i = 0; i < 64; ++i)
        {
            const auto s1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
            const auto choice = (e & f) ^ (~e & g);
            const auto t1 = h + s1 + choice + constants[i] + words[i];
            const auto s0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + s0 + majority;
        }
        const std::array<std::uint32_t, 8> next{a, b, c, d, e, f, g, h};
        for (std::size_t i = 0; i < 8; ++i)
        {
            state[i] += next[i];
        }
    };
    const auto complete = bytes.size() / 64 * 64;
    for (std::size_t offset = 0; offset < complete; offset += 64)
    {
        compress(bytes.subspan(offset, 64));
    }
    std::array<std::uint8_t, 128> tail{};
    const auto remaining = bytes.size() - complete;
    std::copy(bytes.begin() + static_cast<std::ptrdiff_t>(complete), bytes.end(), tail.begin());
    tail[remaining] = 0x80;
    const auto padded = remaining < 56 ? 64u : 128u;
    const auto bits = static_cast<std::uint64_t>(bytes.size()) * 8;
    for (unsigned i = 0; i < 8; ++i)
    {
        tail[padded - 1 - i] = static_cast<std::uint8_t>(bits >> (8 * i));
    }
    compress(ByteView(tail).first(64));
    if (padded == 128)
    {
        compress(ByteView(tail).subspan(64));
    }
    std::array<std::uint8_t, 32> digest;
    for (std::size_t i = 0; i < digest.size(); ++i)
    {
        digest[i] = static_cast<std::uint8_t>(state[i / 4] >> (24 - (i % 4) * 8));
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (auto value : digest)
    {
        out << std::setw(2) << unsigned(value);
    }
    return out.str();
}
} // namespace tsac
