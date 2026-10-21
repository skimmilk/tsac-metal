#pragma once
#include "tsac/audio_analyzer.hpp"
#include "tsac/model_directory.hpp"

namespace tsac
{
class MetalAnalyzer final : public AudioAnalyzer
{
public:
    // One idle prepared model bundle is retained between matching calls.
    // Concurrent calls lease independent sessions; model files must stay immutable.
    explicit MetalAnalyzer(ModelDirectory directory, const Limits& limits = {});
    Bytes encode(PcmView audio, const EncodeOptions& options = {},
                 const EntropyObserver& observer = {}) override;
    PcmAudio decode(ByteView stream, const EntropyObserver& observer = {}) override;

private:
    struct Cache;
    std::shared_ptr<Cache> cache_;
    ModelDirectory directory_;
    Limits limits_;
};
} // namespace tsac
