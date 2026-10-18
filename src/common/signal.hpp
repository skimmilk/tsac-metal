#pragma once
#include "common/common.hpp"

namespace tsac
{
// [channels,frames,groups], axis 0 fastest. Groups are independent recordings,
// not convolution groups; every group uses the same weights and alpha values.
struct SignalShape
{
    std::size_t channels = 0, frames = 0, groups = 1;
    std::size_t elements(const Limits& limits = {}) const;
};

struct Signal
{
    SignalShape shape;
    std::vector<float> values;
};

enum class ConvolutionKind
{
    ordinary,
    transposed
};

struct Conv1dConfig
{
    std::size_t input_channels = 0, output_channels = 0, kernel = 0;
    std::size_t stride = 1, padding = 0, dilation = 1;
    ConvolutionKind kind = ConvolutionKind::ordinary;
};
} // namespace tsac
