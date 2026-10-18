#pragma once
#include "common/signal.hpp"

namespace tsac
{
// Native PCM16 input. Output is interleaved [input_channels,samples,1], padded
// with zeros to a 512-sample boundary. FFmpeg uses argv, never a command shell.
Signal read_audio(const std::filesystem::path& path, int requested_channels = 0,
                  const Limits& limits = {});
Signal pcm16_input(ByteView bytes, unsigned channels, const Limits& limits = {});
Signal wave_input(ByteView bytes, const Limits& limits = {});
Signal group_audio(const Signal& interleaved, bool separate, const Limits& limits = {});
// 44.1 kHz PCM16 RIFF/WAVE. Groups are interleaved after model channels:
// channel = model_channel + model_channels * group. Preserves all padded samples.
// Float samples are scaled by 32768, rounded to nearest with even ties, and clipped.
Bytes pcm16_wave(std::span<const float> samples, SignalShape shape, const Limits& limits = {});
} // namespace tsac
