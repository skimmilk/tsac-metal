#include "format/range.hpp"

#include "common/internal.hpp"

#include <bit>
#include <utility>

namespace tsac
{
namespace
{
// A 24-bit interval emits bytes until its width reaches this threshold.
// The split uses Q15 probabilities; these constants follow entropy-coding.md.
constexpr std::uint32_t minimum_range = 0xff01;
constexpr unsigned probability_bits = 15;

std::uint32_t split(std::uint32_t range, std::uint16_t p0)
{
    detail::require(p0 > 0 && p0 < 32768, "Q15 probability must be in [1,32767]");
    detail::require(range >= minimum_range, "unnormalized range state");
    return static_cast<std::uint32_t>((std::uint64_t(range) * p0) >> probability_bits);
}
} // namespace

void RangeEncoder::emit(std::uint32_t value)
{
    detail::require(bytes_.size() < max_bytes_, "range output exceeds limit");
    bytes_.push_back(static_cast<std::uint8_t>(value));
}

void RangeEncoder::flush_pending(std::uint32_t value)
{
    const auto carry = (value >> 8) & 255;
    if (pending_count_)
    {
        emit(std::uint32_t(pending_byte_) + carry);
        while (pending_count_ > 1)
        {
            emit(carry - 1);
            --pending_count_;
        }
    }
    pending_byte_ = static_cast<std::uint8_t>(value);
    pending_count_ = 1;
}

void RangeEncoder::shift()
{
    if ((low_ >> 16) == 0xff)
    {
        pending_count_ = checked_add(pending_count_, 1);
        detail::require(pending_count_ <= max_bytes_, "pending range output exceeds limit");
    }
    else
    {
        flush_pending(low_ >> 16);
    }
    low_ = (low_ & 0xffff) << 8;
    range_ <<= 8;
}

void RangeEncoder::put(std::uint16_t probability_zero, bool bit)
{
    detail::require(!finished_, "range encoder already finished");
    const auto range0 = split(range_, probability_zero);
    if (bit)
    {
        low_ += range0;
        range_ -= range0;
    }
    else
    {
        range_ = range0;
    }
    while (range_ < minimum_range)
    {
        shift();
    }
}

std::size_t RangeEncoder::bit_position() const
{
    detail::require(!finished_, "range encoder already finished");
    const auto whole_bytes = checked_add(pending_count_, bytes_.size());
    return checked_add(checked_mul(whole_bytes, 8),
                       static_cast<std::size_t>(24 - std::bit_width(range_)));
}

Bytes RangeEncoder::finish()
{
    detail::require(!finished_, "range encoder already finished");
    if (range_ < 0x10000)
    {
        shift();
    }
    const auto n = std::bit_width(range_) - 1;
    detail::require(n >= 16 && n <= 23, "invalid final range");
    const std::uint32_t unit = 1u << n;
    const auto val = (low_ & (unit - 1)) == 0 ? low_ : (low_ + unit) & (0u - unit);
    detail::require(val >= low_ && val < low_ + range_, "invalid range termination");
    if ((val >> 16) == 0xff)
    {
        pending_count_ = checked_add(pending_count_, 1);
        detail::require(pending_count_ <= max_bytes_, "pending range output exceeds limit");
    }
    else
    {
        flush_pending(val >> 16);
    }
    if (pending_count_)
    {
        flush_pending(0);
    }
    finished_ = true;
    return std::move(bytes_);
}

std::uint32_t RangeDecoder::read()
{
    if (offset_ < bytes_.size())
    {
        return bytes_[offset_++];
    }
    zero_bytes_ = checked_add(zero_bytes_, 1);
    return 0;
}

RangeDecoder::RangeDecoder(ByteView bytes) : bytes_(bytes)
{
    // Keep calls sequenced: operand evaluation order must not reorder reads.
    code_ = read() << 16;
    code_ |= read() << 8;
    code_ |= read();
    detail::require(code_ < range_, "range code outside initial interval");
}

bool RangeDecoder::get(std::uint16_t probability_zero)
{
    const auto range0 = split(range_, probability_zero);
    const bool bit = code_ >= range0;
    if (bit)
    {
        code_ -= range0;
        range_ -= range0;
    }
    else
    {
        range_ = range0;
    }
    while (range_ < minimum_range)
    {
        range_ <<= 8;
        code_ = (code_ << 8) + read();
    }
    detail::require(code_ < range_, "range code outside interval");
    return bit;
}
} // namespace tsac
