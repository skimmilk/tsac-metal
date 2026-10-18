#pragma once
#include "tsac/audio_analyzer.hpp"
#include "tsac/model_directory.hpp"

namespace tsac
{
class MetalAnalyzer final : public AudioAnalyzer
{
public:
    explicit MetalAnalyzer(ModelDirectory directory, const Limits& limits = {});
    Bytes encode(PcmView audio, const EncodeOptions& options = {},
                 const EntropyObserver& observer = {}) override;
    PcmAudio decode(ByteView stream, const EntropyObserver& observer = {}) override;

private:
    ModelDirectory directory_;
    Limits limits_;
};
} // namespace tsac
