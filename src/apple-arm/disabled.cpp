#include "tsac/apple_arm_analyzer.hpp"

namespace tsac
{
AppleArmAnalyzer::AppleArmAnalyzer(ModelDirectory directory, const Limits& limits, unsigned threads)
    : directory_(std::move(directory)), limits_(limits), threads_(threads)
{
    throw Error("Apple ARM support was disabled at build time");
}

Bytes AppleArmAnalyzer::encode(PcmView, const EncodeOptions&, const EntropyObserver&)
{
    throw Error("Apple ARM support was disabled at build time");
}

PcmAudio AppleArmAnalyzer::decode(ByteView, const EntropyObserver&)
{
    throw Error("Apple ARM support was disabled at build time");
}
} // namespace tsac
