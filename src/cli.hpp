#pragma once
#include "common/common.hpp"

namespace tsac
{
struct CliOptions
{
    char command = 0;
    std::filesystem::path input, output;
    int threads = 0, quality = 0, channels = 0, batch = 0;
    unsigned verbose = 0;
    bool fast = false, separate = false, help = false;
    bool metal = false;
};

CliOptions parse_cli(std::span<const std::string_view> arguments);
std::string_view cli_help();
} // namespace tsac
