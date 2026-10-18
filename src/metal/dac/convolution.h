#pragma once

#include "parameters.h"

namespace tsac_dac
{

using namespace metal;
using namespace tsac_metal_abi;

inline bool convolution_source(uint frame, uint tap, constant ConvolutionParams& p,
                               thread uint& source)
{
    if (frame >= p.output_frames)
    {
        return false;
    }
    if (!p.transposed)
    {
        uint position = frame * p.stride + tap * p.dilation;
        if (position < p.padding)
        {
            return false;
        }
        source = position - p.padding;
    }
    else
    {
        uint position = frame + p.padding, offset = tap * p.dilation;
        if (position < offset || (position - offset) % p.stride)
        {
            return false;
        }
        source = (position - offset) / p.stride;
    }
    return source < p.input_frames;
}

// Static geometry removes inner-loop predicates for selected DAC layers.
// C=0 retains the generic geometry, including transposed convolutions.
template <uint C, uint D, uint S = 1, uint P = 0, bool Transposed = false>
inline bool tile_source(uint frame, uint tap, constant ConvolutionParams& p, thread uint& source)
{
    if (!C)
    {
        return convolution_source(frame, tap, p, source);
    }
    if (Transposed)
    {
        uint position = frame + P, offset = tap * D;
        if (frame >= p.output_frames || position < offset || (position - offset) % S)
        {
            return false;
        }
        source = (position - offset) / S;
    }
    else
    {
        uint position = frame + tap * D;
        if (frame >= p.output_frames || position < 3 * D)
        {
            return false;
        }
        source = position - 3 * D;
    }
    return source < p.input_frames;
}

template <uint channel_quads, uint threads = 128, uint input_block = 8, bool pointwise = false,
          uint F = 8, uint C = 0, uint D = 0, bool Interior = false, uint Out = 0, uint K = 0,
          uint S = 0, uint P = 0, bool Transpose = false>
inline void convolution_tile(device const float* input, device const float4* weights,
                             device const float4* bias, device float4* output,
                             constant ConvolutionParams& p, uint tile, uint tid,
                             threadgroup float4* shared_weights, threadgroup float* shared_input,
                             uint3 coordinates = uint3(0))
{
    // Transposed tiles contain one output phase. Every thread then agrees
    // which taps can contribute, so absent phases skip the entire input loop.
    const uint cin = C ? C : p.input_channels, cout = C ? (Out ? Out : C) : p.output_channels,
               taps = C ? (K ? K : 7u) : p.kernel_size;
    const uint stride = C ? (S ? S : 1u) : p.stride,
               padding = C ? (Transpose ? P : 3u * D) : p.padding, dilation = C ? D : p.dilation;
    const uint transposed = C ? uint(Transpose) : p.transposed;
    constexpr uint frame_width = threads * F / channel_quads;
    // phase is the output-frame residue modulo stride: a tap contributes iff
    // (phase + padding) % stride == (tap * dilation) % stride. Grouping equal
    // residues makes this a uniform skip before any shared loads or barriers.
    // Fall back to contiguous frames when stride is sparse or rounding a phase
    // tile up to frame_width could overflow the shader's uint coordinates.
    uint phases = C ? (Transpose ? stride : 1u)
                    : (transposed && stride <= p.output_frames &&
                               stride <= (0xffffffffu - p.output_frames) / (frame_width - 1)
                           ? stride
                           : 1u);
    uint phase_frames = (p.output_frames - 1) / phases + 1;
    uint channel_tiles = (cout - 1) / (channel_quads * 4) + 1,
         frame_tiles = (phase_frames - 1) / frame_width + 1;
    uint channel_base =
        (C ? (Transpose ? coordinates.x % channel_tiles : coordinates.x) : tile % channel_tiles) *
        channel_quads;
    uint frame_base = (C ? coordinates.y : (tile / channel_tiles) % frame_tiles) * frame_width;
    uint phase = C ? (Transpose ? coordinates.x / channel_tiles : 0u)
                   : (tile / (channel_tiles * frame_tiles)) % phases;
    uint group = C ? coordinates.z : tile / (channel_tiles * frame_tiles * phases);
    uint quads = cout / 4;
    uint channel = channel_base + tid % channel_quads,
         first_frame = frame_base + (tid / channel_quads) * F;
    // Each thread owns four channels at F frames. The hot shapes use four
    // accumulators instead of eight, reducing register pressure. Input blocks
    // remain ordered, including skipping (not multiplying) padding holes.
    float4 sums[F];
    for (uint f = 0; f < F; ++f)
    {
        sums[f] = 0;
    }
    for (uint tap = 0; tap < (pointwise ? 1u : taps); ++tap)
    {
        if (phases > 1 && (phase + padding) % stride != (tap * dilation) % stride)
        {
            continue;
        }
        bool valid[F];
        for (uint f = 0; f < F; ++f)
        {
            uint unused = 0;
            valid[f] =
                Interior || (pointwise ? first_frame + f < p.output_frames
                                       : tile_source<C, D, S ? S : 1, P, Transpose>(
                                             (first_frame + f) * phases + phase, tap, p, unused));
        }
        for (uint begin = 0; begin < cin; begin += input_block)
        {
            for (uint row = tid / channel_quads; row < input_block; row += threads / channel_quads)
            {
                uint in = begin + row, out = channel_base + tid % channel_quads;
                shared_weights[row * channel_quads + tid % channel_quads] =
                    in < cin && out < quads ? weights[out + quads * (in + cin * tap)] : float4(0);
            }
            if (C)
            {
#pragma unroll
                for (uint iteration = 0;
                     iteration < (frame_width * input_block + threads - 1) / threads; ++iteration)
                {
                    const uint element = tid + iteration * threads;
                    if (element >= frame_width * input_block)
                    {
                        continue;
                    }
                    uint frame = (frame_base + element / input_block) * phases + phase,
                         in = begin + element % input_block, source = 0;
                    if (Interior)
                    {
                        source = Transpose ? (frame + padding - tap * dilation) / stride
                                           : frame + tap * D - 3 * D;
                    }
                    bool present =
                        Interior ||
                        (in < cin && (pointwise ? (source = frame, frame < p.output_frames)
                                                : tile_source<C, D, S ? S : 1, P, Transpose>(
                                                      frame, tap, p, source)));
                    shared_input[element] =
                        present ? input[(group * p.input_frames + source) * cin + in] : 0.0f;
                }
            }
            else
            {
                for (uint element = tid; element < frame_width * input_block; element += threads)
                {
                    uint frame = (frame_base + element / input_block) * phases + phase,
                         in = begin + element % input_block, source = 0;
                    if (Interior)
                    {
                        source = Transpose ? (frame + padding - tap * dilation) / stride
                                           : frame + tap * D - 3 * D;
                    }
                    bool present =
                        Interior ||
                        (in < cin && (pointwise ? (source = frame, frame < p.output_frames)
                                                : tile_source<C, D, S ? S : 1, P, Transpose>(
                                                      frame, tap, p, source)));
                    shared_input[element] =
                        present ? input[(group * p.input_frames + source) * cin + in] : 0.0f;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint row = 0; row < min(input_block, cin - begin); ++row)
            {
                float4 w = shared_weights[row * channel_quads + tid % channel_quads];
                for (uint f = 0; f < F; ++f)
                {
                    // Padding skips the FMA entirely, just as in the scalar
                    // kernel; multiplying padding by zero can change signed zero.
                    if (valid[f])
                    {
                        sums[f] = fma(
                            float4(
                                shared_input[((tid / channel_quads) * F + f) * input_block + row]),
                            w, sums[f]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (channel < quads)
    {
        float4 b = bias[channel];
        for (uint f = 0; f < F; ++f)
        {
            if ((first_frame + f) * phases + phase < p.output_frames)
            {
                output[(group * p.output_frames + (first_frame + f) * phases + phase) * quads +
                       channel] = sums[f] + b;
            }
        }
    }
}

} // namespace tsac_dac
