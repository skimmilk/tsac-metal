#pragma once
#include "common/common.hpp"

namespace tsac
{
// Integer layer only: callers supply Q15 probabilities of zero in [1,32767].
// No float categorical tree or neural predictions are implied by this API.
class RangeEncoder
{
public:
    explicit RangeEncoder(std::size_t max_bytes = std::size_t{64} * 1024 * 1024)
        : max_bytes_(max_bytes)
    {
    }

    void put(std::uint16_t probability_zero, bool bit);
    // Original unfinalized arithmetic position, including delayed carry bytes.
    // Does not count physical termination bytes; valid only before finish().
    std::size_t bit_position() const;
    Bytes finish(); // One-shot. Includes all physical termination bytes.
private:
    void emit(std::uint32_t value);
    void flush_pending(std::uint32_t value);
    void shift();
    std::uint32_t range_ = 0xff0000, low_ = 0;
    std::uint8_t pending_byte_ = 0xff;
    std::size_t pending_count_ = 0, max_bytes_;
    bool finished_ = false;
    Bytes bytes_;
};

class RangeDecoder
{
public:
    // The input is borrowed; it must outlive the decoder.
    explicit RangeDecoder(ByteView bytes);
    bool get(std::uint16_t probability_zero);

    // Zero extension is part of original termination; it is not an integrity check.
    std::size_t zero_extension_bytes() const
    {
        return zero_bytes_;
    }

private:
    std::uint32_t read();
    ByteView bytes_;
    std::size_t offset_ = 0, zero_bytes_ = 0;
    std::uint32_t range_ = 0xff0000, code_ = 0;
};
} // namespace tsac
