#pragma once
#include <bit>
#include <cfenv>
#include <cstdint>

namespace tsac::detail::probability
{
// Match the codec's nearest-even arithmetic without leaking an FP rounding
// policy to callers. Batch helpers hold one guard around the whole vector loop.
class NearestRounding
{
public:
    NearestRounding() : previous_(std::fegetround())
    {
        if (previous_ != FE_TONEAREST)
        {
            std::fesetround(FE_TONEAREST);
        }
    }

    ~NearestRounding()
    {
        if (previous_ != FE_TONEAREST)
        {
            std::fesetround(previous_);
        }
    }

private:
    int previous_;
};

inline float flush(float x)
{
    const auto bits = std::bit_cast<std::uint32_t>(x);
    return (bits & 0x7f800000u) == 0 ? std::bit_cast<float>(bits & 0x80000000u) : x;
}

inline float add(float x, float y)
{
    volatile float v = flush(x) + flush(y);
    return flush(v);
}

inline float subtract(float x, float y)
{
    volatile float v = flush(x) - flush(y);
    return flush(v);
}

inline float multiply(float x, float y)
{
    volatile float v = flush(x) * flush(y);
    return flush(v);
}

inline float divide(float x, float y)
{
    volatile float v = flush(x) / flush(y);
    return flush(v);
}

} // namespace tsac::detail::probability
