#include "../abi/bindings.h"
#include "convolution.h"
#include "parameters.h"

using namespace metal;
using namespace tsac_metal_abi;
using namespace tsac_dac;

// Selected only when the host retains all stride phases and K=2*S, P=S/2.
template <uint C, uint O, uint S, uint Q>
inline void convolution_transposed(device const float* x, device const float4* w,
                                   device const float4* b, device float4* y,
                                   constant ConvolutionParams& p, uint3 tile, uint tid,
                                   threadgroup float4* sw, threadgroup float* sx)
{
    constexpr uint frames = 128 * 8 / Q, channel_tiles = (O + Q * 4 - 1) / (Q * 4);
    const uint first = tile.y * frames, phase = tile.x / channel_tiles;
    // Each phase has two contributing taps, with source offsets in [-1,+1].
    // This conservative region keeps both sources valid for every phase.
    const bool inside = first >= 1 && first < p.input_frames &&
                        p.input_frames - first >= frames + 1 && p.output_frames > phase &&
                        (p.output_frames - 1 - phase) / S >= first + frames - 1;
    if (inside)
    {
        convolution_tile<Q, 128, 8, false, 8, C, 1, true, O, 2 * S, S, S / 2, true>(
            x, w, b, y, p, tile.x, tid, sw, sx, tile);
    }
    else
    {
        convolution_tile<Q, 128, 8, false, 8, C, 1, false, O, 2 * S, S, S / 2, true>(
            x, w, b, y, p, tile.x, tid, sw, sx, tile);
    }
}

kernel void conv_transpose_192_96(device const float* x [[buffer(convolution_binding::input)]],
                                  device const float4* w [[buffer(convolution_binding::weights)]],
                                  device const float4* b [[buffer(convolution_binding::bias)]],
                                  device float4* y [[buffer(convolution_binding::output)]],
                                  constant ConvolutionParams& p
                                  [[buffer(convolution_binding::parameters)]],
                                  uint3 tile [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float4 sw[256];
    threadgroup float sx[256];
    convolution_transposed<192, 96, 2, 32>(x, w, b, y, p, tile, tid, sw, sx);
}

kernel void conv_transpose_384_192(device const float* x [[buffer(convolution_binding::input)]],
                                   device const float4* w [[buffer(convolution_binding::weights)]],
                                   device const float4* b [[buffer(convolution_binding::bias)]],
                                   device float4* y [[buffer(convolution_binding::output)]],
                                   constant ConvolutionParams& p
                                   [[buffer(convolution_binding::parameters)]],
                                   uint3 tile [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float4 sw[128];
    threadgroup float sx[512];
    convolution_transposed<384, 192, 4, 16>(x, w, b, y, p, tile, tid, sw, sx);
}

kernel void conv_transpose_768_384(device const float* x [[buffer(convolution_binding::input)]],
                                   device const float4* w [[buffer(convolution_binding::weights)]],
                                   device const float4* b [[buffer(convolution_binding::bias)]],
                                   device float4* y [[buffer(convolution_binding::output)]],
                                   constant ConvolutionParams& p
                                   [[buffer(convolution_binding::parameters)]],
                                   uint3 tile [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float4 sw[256];
    threadgroup float sx[256];
    convolution_transposed<768, 384, 8, 32>(x, w, b, y, p, tile, tid, sw, sx);
}

kernel void conv_transpose_1536_768(device const float* x [[buffer(convolution_binding::input)]],
                                    device const float4* w [[buffer(convolution_binding::weights)]],
                                    device const float4* b [[buffer(convolution_binding::bias)]],
                                    device float4* y [[buffer(convolution_binding::output)]],
                                    constant ConvolutionParams& p
                                    [[buffer(convolution_binding::parameters)]],
                                    uint3 tile [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float4 sw[256];
    threadgroup float sx[256];
    convolution_transposed<1536, 768, 8, 32>(x, w, b, y, p, tile, tid, sw, sx);
}
