#pragma once
#include "tsac/audio_analyzer.hpp"
#include "tsac/model_directory.hpp"

namespace tsac
{
// Portable reference and research implementation using standard C++ arithmetic.
class ScalarAnalyzer final : public AudioAnalyzer
{
public:
    explicit ScalarAnalyzer(ModelDirectory directory, const Limits& limits = {});
    Bytes encode(PcmView audio, const EncodeOptions& options = {},
                 const EntropyObserver& observer = {}) override;
    PcmAudio decode(ByteView stream, const EntropyObserver& observer = {}) override;

private:
    ModelDirectory directory_;
    Limits limits_;
};
} // namespace tsac
