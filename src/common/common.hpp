#pragma once
#include "tsac/types.hpp"

#include <filesystem>

namespace tsac
{
std::size_t checked_add(std::size_t a, std::size_t b);
std::size_t checked_mul(std::size_t a, std::size_t b);
Bytes read_file(const std::filesystem::path& path, const Limits& limits = {});
void write_file(const std::filesystem::path& path, ByteView bytes);
std::string sha256(ByteView bytes);
} // namespace tsac
