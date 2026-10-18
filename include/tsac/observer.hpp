#pragma once
#include <cstddef>
#include <functional>

namespace tsac
{
struct EntropyPosition
{
    std::size_t frame, group, codebook, lane, step, token_offset;
};

struct EntropyObserver
{
    std::function<void(const EntropyPosition&)> before_symbol;
    std::function<void(const EntropyPosition&, std::size_t)> after_symbol;
};
} // namespace tsac
