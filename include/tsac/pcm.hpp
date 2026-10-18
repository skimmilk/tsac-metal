#pragma once
#include "tsac/types.hpp"

#include <mdspan>

namespace tsac
{
// Borrowed normalized float32 PCM at 44.1 kHz, indexed [channel, frame].
// Positive strides support interleaved, planar and frame-sliced recordings.
using PcmView = std::mdspan<const float, std::dextents<std::size_t, 2>, std::layout_stride>;
using PcmChannelView = std::mdspan<const float, std::dextents<std::size_t, 1>, std::layout_stride>;

PcmView interleaved_pcm(std::span<const float> samples, std::size_t channels);
PcmChannelView channel_view(PcmView audio, std::size_t channel);
PcmView frame_view(PcmView audio, std::size_t first, std::size_t count);

struct PcmAudio
{
    std::vector<float> samples;
    unsigned channels = 1;

    PcmView view() const&;
    PcmView view() const&& = delete;
};
} // namespace tsac
