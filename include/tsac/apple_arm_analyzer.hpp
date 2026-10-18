#pragma once
#include "tsac/audio_analyzer.hpp"
#include "tsac/model_directory.hpp"

namespace tsac
{
class AppleArmAnalyzer final : public AudioAnalyzer
{
public:
    // Zero keeps automatic BLAS threading; 1–256 bounds native CPU workers.
    // Retains one idle prepared session for repeated matching recordings.
    // Concurrent calls lease separate sessions; model files must stay immutable.
    explicit AppleArmAnalyzer(ModelDirectory directory, const Limits& limits = {},
                              unsigned threads = 0);
    Bytes encode(PcmView audio, const EncodeOptions& options = {},
                 const EntropyObserver& observer = {}) override;
    PcmAudio decode(ByteView stream, const EntropyObserver& observer = {}) override;

private:
    struct Cache;
    std::shared_ptr<Cache> cache_;
    ModelDirectory directory_;
    Limits limits_;
    [[maybe_unused]] unsigned threads_; // Native support can be disabled at build time.
};
} // namespace tsac
