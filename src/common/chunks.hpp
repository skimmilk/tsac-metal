#pragma once
#include "common/signal.hpp"

#include <algorithm>
#include <functional>
#include <string_view>

namespace tsac::detail
{
using SignalObserver = std::function<void(std::string_view, const Signal&)>;

inline SignalObserver chunk_observer(const SignalObserver& observer, std::string prefix)
{
    if (!observer)
    {
        return {};
    }
    return [observer, prefix = std::move(prefix)](std::string_view name, const Signal& signal)
    {
        observer(prefix + std::string(name), signal);
    };
}

inline Signal signal_window(std::span<const float> input, SignalShape shape, std::size_t left,
                            std::size_t right, const Limits& limits)
{
    Signal result{{shape.channels, right - left, shape.groups}, {}};
    result.values.resize(result.shape.elements(limits));
    const auto count = result.shape.frames * shape.channels;
    for (std::size_t group = 0; group < shape.groups; ++group)
    {
        std::copy_n(input.data() + (group * shape.frames + left) * shape.channels, count,
                    result.values.data() + group * count);
    }
    return result;
}

inline void retain_window(const Signal& input, Signal& output, std::size_t source,
                          std::size_t destination, std::size_t frames)
{
    const auto channels = input.shape.channels;
    for (std::size_t group = 0; group < input.shape.groups; ++group)
    {
        std::copy_n(input.values.data() + (group * input.shape.frames + source) * channels,
                    frames * channels,
                    output.values.data() + (group * output.shape.frames + destination) * channels);
    }
}
} // namespace tsac::detail
