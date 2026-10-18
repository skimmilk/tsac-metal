#pragma once
#include "models/types.hpp"

namespace tsac::model_layout
{
// Generated implementation data: offsets and byte lengths only.
struct Range
{
    std::size_t offset, bytes;
};

struct Dac
{
    std::span<const Range> values, gains, biases, alphas;
};

struct Rvq
{
    std::span<const std::array<Range, 7>> books;
};

struct Transformer
{
    std::span<const std::array<Range, 3>> projections;
    std::span<const std::array<Range, 2>> norms;
    std::array<Range, 2> constants;
    Range embeddings;
};
} // namespace tsac::model_layout
