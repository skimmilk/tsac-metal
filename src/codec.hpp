#pragma once
#include "cli.hpp"
#include "format/container.hpp"
#include "weights/model_directory.hpp"

namespace tsac
{
std::filesystem::path executable_directory();
std::vector<std::uint16_t> decode_tokens(ByteView stream, const Limits& limits = {});

struct CodecStatistics
{
    std::size_t frames = 0, channels = 0, stream_bytes = 0;
    double elapsed_seconds = 0, peak_memory_gb = 0;
    double bitrate_kbps() const;
};

Bytes compress_audio(const CliOptions& options, CodecStatistics& statistics,
                     const Limits& limits = {});
Bytes decompress_audio(ByteView stream, const CliOptions& options, CodecStatistics& statistics,
                       const Limits& limits = {});
CodecStatistics run_codec(const CliOptions& options, const Limits& limits = {});
} // namespace tsac
