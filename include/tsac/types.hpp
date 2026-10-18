#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tsac
{
using Bytes = std::vector<std::uint8_t>;
using ByteView = std::span<const std::uint8_t>;

struct Error : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

// Policy limits are independent of format limits and checked before allocation.
struct Limits
{
    std::size_t file_bytes = std::size_t{512} * 1024 * 1024;
    std::size_t symbols = std::size_t{64} * 1024 * 1024;
    std::size_t pcm_bytes = std::size_t{1024} * 1024 * 1024;
    std::size_t cache_bytes = std::size_t{1024} * 1024 * 1024;
    std::size_t tensor_elements = std::size_t{128} * 1024 * 1024;
    std::size_t tensors = 10000;
    std::size_t metadata_bytes = std::size_t{1024} * 1024;
    std::size_t name_bytes = 4096;
};
} // namespace tsac
