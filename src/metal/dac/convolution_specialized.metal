#include "../abi/bindings.h"
#include "convolution.h"
#include "parameters.h"

using namespace metal;
using namespace tsac_metal_abi;
using namespace tsac_dac;

// The hot ordinary kernels use a 3D grid and a uniform interior decision.
// Keep the checked path for padding/tails; zero-filled padding must skip FMAs.
template <uint C, uint D, uint Q = 16, uint T = 128, uint B = 16, uint F = 4>
inline void convolution_hot(device const float* x, device const float4* w, device const float4* b,
                            device float4* y, constant ConvolutionParams& p, uint3 tile, uint tid,
                            threadgroup float4* sw, threadgroup float* sx)
{
    constexpr uint frames = T * F / Q;
    const uint first = tile.y * frames;
    const bool inside = first >= 3 * D && first < p.input_frames &&
                        p.input_frames - first >= frames + 3 * D && first < p.output_frames &&
                        p.output_frames - first >= frames;
    if (inside)
    {
        convolution_tile<Q, T, B, false, F, C, D, true>(x, w, b, y, p, tile.x, tid, sw, sx, tile);
    }
    else
    {
        convolution_tile<Q, T, B, false, F, C, D>(x, w, b, y, p, tile.x, tid, sw, sx, tile);
    }
}

kernel void conv_hot_384_1(device const float* x [[buffer(convolution_binding::input)]],
                           device const float4* w [[buffer(convolution_binding::weights)]],
                           device const float4* b [[buffer(convolution_binding::bias)]],
                           device float4* y [[buffer(convolution_binding::output)]],
                           constant ConvolutionParams& p
                           [[buffer(convolution_binding::parameters)]],
                           uint3 tile [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float4 sw[256];
    threadgroup float sx[512];
    convolution_hot<384, 1>(x, w, b, y, p, tile, tid, sw, sx);
}

kernel void conv_hot_384_3(device const float* x [[buffer(convolution_binding::input)]],
                           device const float4* w [[buffer(convolution_binding::weights)]],
                           device const float4* b [[buffer(convolution_binding::bias)]],
                           device float4* y [[buffer(convolution_binding::output)]],
                           constant ConvolutionParams& p
                           [[buffer(convolution_binding::parameters)]],
                           uint3 tile [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float4 sw[256];
    threadgroup float sx[512];
    convolution_hot<384, 3>(x, w, b, y, p, tile, tid, sw, sx);
}

kernel void conv_hot_384_9(device const float* x [[buffer(convolution_binding::input)]],
                           device const float4* w [[buffer(convolution_binding::weights)]],
                           device const float4* b [[buffer(convolution_binding::bias)]],
                           device float4* y [[buffer(convolution_binding::output)]],
                           constant ConvolutionParams& p
                           [[buffer(convolution_binding::parameters)]],
                           uint3 tile [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float4 sw[256];
    threadgroup float sx[512];
    convolution_hot<384, 9>(x, w, b, y, p, tile, tid, sw, sx);
}

kernel void conv_hot_192_1(device const float* x [[buffer(convolution_binding::input)]],
                           device const float4* w [[buffer(convolution_binding::weights)]],
                           device const float4* b [[buffer(convolution_binding::bias)]],
                           device float4* y [[buffer(convolution_binding::output)]],
                           constant ConvolutionParams& p
                           [[buffer(convolution_binding::parameters)]],
                           uint3 tile [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float4 sw[256];
    threadgroup float sx[512];
    convolution_hot<192, 1>(x, w, b, y, p, tile, tid, sw, sx);
}

kernel void conv_hot_192_3(device const float* x [[buffer(convolution_binding::input)]],
                           device const float4* w [[buffer(convolution_binding::weights)]],
                           device const float4* b [[buffer(convolution_binding::bias)]],
                           device float4* y [[buffer(convolution_binding::output)]],
                           constant ConvolutionParams& p
                           [[buffer(convolution_binding::parameters)]],
                           uint3 tile [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float4 sw[256];
    threadgroup float sx[512];
    convolution_hot<192, 3>(x, w, b, y, p, tile, tid, sw, sx);
}

kernel void conv_hot_192_9(device const float* x [[buffer(convolution_binding::input)]],
                           device const float4* w [[buffer(convolution_binding::weights)]],
                           device const float4* b [[buffer(convolution_binding::bias)]],
                           device float4* y [[buffer(convolution_binding::output)]],
                           constant ConvolutionParams& p
                           [[buffer(convolution_binding::parameters)]],
                           uint3 tile [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float4 sw[256];
    threadgroup float sx[512];
    convolution_hot<192, 9>(x, w, b, y, p, tile, tid, sw, sx);
}

kernel void conv_static_768_1(device const float* x [[buffer(convolution_binding::input)]],
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
    convolution_hot<768, 1, 32, 128, 8, 8>(x, w, b, y, p, tile, tid, sw, sx);
}

kernel void conv_static_768_3(device const float* x [[buffer(convolution_binding::input)]],
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
    convolution_hot<768, 3, 32, 128, 8, 8>(x, w, b, y, p, tile, tid, sw, sx);
}

kernel void conv_static_768_9(device const float* x [[buffer(convolution_binding::input)]],
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
    convolution_hot<768, 9, 32, 128, 8, 8>(x, w, b, y, p, tile, tid, sw, sx);
}

kernel void conv_static_96_1(device const float* x [[buffer(convolution_binding::input)]],
                             device const float4* w [[buffer(convolution_binding::weights)]],
                             device const float4* b [[buffer(convolution_binding::bias)]],
                             device float4* y [[buffer(convolution_binding::output)]],
                             constant ConvolutionParams& p
                             [[buffer(convolution_binding::parameters)]],
                             uint3 tile [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float4 sw[192];
    threadgroup float sx[256];
    convolution_hot<96, 1, 24, 96, 8, 8>(x, w, b, y, p, tile, tid, sw, sx);
}

kernel void conv_static_96_3(device const float* x [[buffer(convolution_binding::input)]],
                             device const float4* w [[buffer(convolution_binding::weights)]],
                             device const float4* b [[buffer(convolution_binding::bias)]],
                             device float4* y [[buffer(convolution_binding::output)]],
                             constant ConvolutionParams& p
                             [[buffer(convolution_binding::parameters)]],
                             uint3 tile [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float4 sw[192];
    threadgroup float sx[256];
    convolution_hot<96, 3, 24, 96, 8, 8>(x, w, b, y, p, tile, tid, sw, sx);
}

kernel void conv_static_96_9(device const float* x [[buffer(convolution_binding::input)]],
                             device const float4* w [[buffer(convolution_binding::weights)]],
                             device const float4* b [[buffer(convolution_binding::bias)]],
                             device float4* y [[buffer(convolution_binding::output)]],
                             constant ConvolutionParams& p
                             [[buffer(convolution_binding::parameters)]],
                             uint3 tile [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float4 sw[192];
    threadgroup float sx[256];
    convolution_hot<96, 9, 24, 96, 8, 8>(x, w, b, y, p, tile, tid, sw, sx);
}
