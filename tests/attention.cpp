#if TSAC_ENABLE_APPLE_ARM
#include "apple-arm/kernels.hpp"
using tsac::apple_arm::attention_bf16;
#else
#include "scalar/kernels.hpp"
using tsac::scalar::attention_bf16;
#endif

#include <iostream>

using namespace tsac;

namespace
{
void check(bool condition, const char* message)
{
    if (!condition)
    {
        throw Error(message);
    }
}

std::vector<std::uint16_t> tiled_keys(std::span<const std::uint16_t> keys, std::size_t heads,
                                      std::size_t coordinates)
{
    const auto width = checked_mul(heads, coordinates);
    check(width > 0 && keys.size() % width == 0, "invalid attention test geometry");
    const auto frames = keys.size() / width;
    std::vector<std::uint16_t> tiled(keys.begin(), keys.end());
    for (std::size_t frame = 0; frame + 4 <= frames; frame += 4)
    {
        for (std::size_t head = 0; head < heads; ++head)
        {
            for (std::size_t j = 0; j < coordinates; ++j)
            {
                for (std::size_t lane = 0; lane < 4; ++lane)
                {
                    tiled[frame * width + head * coordinates * 4 + j * 4 + lane] =
                        keys[(frame + lane) * width + head * coordinates + j];
                }
            }
        }
    }
    return tiled;
}

void hash_values(std::uint64_t& checksum, std::span<const std::uint16_t> values)
{
    for (auto value : values)
    {
        checksum = (checksum ^ (value & 255u)) * 1099511628211ull;
        checksum = (checksum ^ (value >> 8)) * 1099511628211ull;
    }
}

void geometry(std::size_t heads, std::size_t coordinates, std::size_t frames,
              std::uint64_t& checksum)
{
    const auto width = heads * coordinates;
    std::vector<std::uint16_t> query(width), keys(frames * width), values(keys.size());
    for (std::size_t i = 0; i < width; ++i)
    {
        query[i] = probability_bf16(float(int((i * 37 + 19) % 257) - 128) / 128);
    }
    for (std::size_t i = 0; i < keys.size(); ++i)
    {
        keys[i] = probability_bf16(float(int((i * 73 + i / 17) % 257) - 128) / 64);
        values[i] = probability_bf16(float(int((i * 31 + i / 7) % 127) - 63) / 32);
    }
    const auto tiled = tiled_keys(keys, heads, coordinates);
    for (bool prefill : {false, true})
    {
        const auto row = attention_bf16(query, keys, values, heads, coordinates,
                                        0.0883883461356163f, prefill, false);
        const auto blocked = attention_bf16(query, tiled, values, heads, coordinates,
                                            0.0883883461356163f, prefill, true);
        check(row == blocked, "attention cache layouts changed BF16 bits");
        hash_values(checksum, row);
    }
}
} // namespace

int main()
{
    try
    {
        std::uint64_t checksum = 14695981039346656037ull;
        // Include SIMD remainders, cache tile boundaries, the first streaming
        // softmax merge, and the replay-specific value rounding path.
        for (std::size_t heads : {1u, 2u, 4u})
        {
            for (std::size_t coordinates : {3u, 4u, 7u, 16u, 128u})
            {
                for (std::size_t frames : {1u, 3u, 4u, 15u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 65u,
                                           127u, 128u, 129u, 257u})
                {
                    geometry(heads, coordinates, frames, checksum);
                }
            }
        }
        // Captured from the original CPU implementation before decomposition.
        std::cout << "attention checksum=" << checksum << '\n';
        check(checksum == 15737425383056432415ull,
              "attention differs from captured original BF16 results");
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
