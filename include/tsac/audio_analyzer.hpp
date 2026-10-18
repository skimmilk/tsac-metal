#pragma once

#include "tsac/observer.hpp"
#include "tsac/pcm.hpp"

#include <optional>

namespace tsac
{
enum class CodingMethod
{
    fast,
    normal
};

struct EncodeOptions
{
    CodingMethod method = CodingMethod::normal;
    unsigned codebooks = 0; // Maximum for the selected model when zero.
    bool separate_channels = false;
    std::optional<std::uint16_t> entropy_batch; // Automatic when absent.
};

// Complete, synchronous recording operations. Input views are borrowed only
// during the call; returned streams/PCM own their storage. Encoding zero-pads
// to 512 samples. Decoding retains padding: TSAC has no original sample count.
// Calls on one instance must be exclusive. Sequential reuse after success or
// failure is supported. Observers run synchronously in entropy order after the
// corresponding work completes. Implementations never print or perform file I/O.
class AudioAnalyzer
{
public:
    virtual ~AudioAnalyzer() = default;
    virtual Bytes encode(PcmView audio, const EncodeOptions& options = {},
                         const EntropyObserver& observer = {}) = 0;
    virtual PcmAudio decode(ByteView stream, const EntropyObserver& observer = {}) = 0;
};
} // namespace tsac
