#include "../abi/bindings.h"
#include "parameters.h"

using namespace metal;
using namespace tsac_metal_abi;

kernel void convolution(device const float* input [[buffer(convolution_binding::input)]],
                        device const float* weights [[buffer(convolution_binding::weights)]],
                        device const float* bias [[buffer(convolution_binding::bias)]],
                        device float* output [[buffer(convolution_binding::output)]],
                        constant ConvolutionParams& p [[buffer(convolution_binding::parameters)]],
                        uint id [[thread_position_in_grid]])
{
    if (id >= p.output_channels * p.output_frames * p.group_count)
    {
        return;
    }
    uint channel = id % p.output_channels, frame = (id / p.output_channels) % p.output_frames,
         group = id / (p.output_channels * p.output_frames);
    float sum = 0;
    for (uint tap = 0; tap < p.kernel_size; ++tap)
    {
        uint source;
        if (!p.transposed)
        {
            uint position = frame * p.stride + tap * p.dilation;
            if (position < p.padding)
            {
                continue;
            }
            source = position - p.padding;
        }
        else
        {
            uint position = frame + p.padding, offset = tap * p.dilation;
            if (position < offset || (position - offset) % p.stride)
            {
                continue;
            }
            source = (position - offset) / p.stride;
        }
        if (source >= p.input_frames)
        {
            continue;
        }
        for (uint in = 0; in < p.input_channels; ++in)
        {
            uint weight = channel + p.output_channels * (in + p.input_channels * tap);
            sum = fma(input[(group * p.input_frames + source) * p.input_channels + in],
                      weights[weight], sum);
        }
    }
    output[id] = sum + bias[channel];
}

kernel void convolution4(device const float* input [[buffer(convolution_binding::input)]],
                         device const float4* weights [[buffer(convolution_binding::weights)]],
                         device const float4* bias [[buffer(convolution_binding::bias)]],
                         device float4* output [[buffer(convolution_binding::output)]],
                         constant ConvolutionParams& p [[buffer(convolution_binding::parameters)]],
                         uint id [[thread_position_in_grid]])
{
    uint quads = p.output_channels / 4;
    if (id >= quads * p.output_frames * p.group_count)
    {
        return;
    }
    uint channel = id % quads, frame = (id / quads) % p.output_frames,
         group = id / (quads * p.output_frames);
    float4 sum = 0;
    for (uint tap = 0; tap < p.kernel_size; ++tap)
    {
        uint source;
        if (!p.transposed)
        {
            uint position = frame * p.stride + tap * p.dilation;
            if (position < p.padding)
            {
                continue;
            }
            source = position - p.padding;
        }
        else
        {
            uint position = frame + p.padding, offset = tap * p.dilation;
            if (position < offset || (position - offset) % p.stride)
            {
                continue;
            }
            source = (position - offset) / p.stride;
        }
        if (source >= p.input_frames)
        {
            continue;
        }
        for (uint in = 0; in < p.input_channels; ++in)
        {
            float x = input[(group * p.input_frames + source) * p.input_channels + in];
            sum = fma(float4(x), weights[channel + quads * (in + p.input_channels * tap)], sum);
        }
    }
    output[id] = sum + bias[channel];
}
