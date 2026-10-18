#pragma once

#include <metal_stdlib>

namespace tsac_probability
{

using namespace metal;

struct QuantizationShared
{
    float maxima[8], inverse;
};

} // namespace tsac_probability
