#include "../abi/bindings.h"
#include "convolution.h"
#include "parameters.h"

using namespace metal;
using namespace tsac_metal_abi;
using namespace tsac_dac;

kernel void convolution_tiled(device const float* input [[buffer(convolution_binding::input)]],
                              device const float4* weights [[buffer(convolution_binding::weights)]],
                              device const float4* bias [[buffer(convolution_binding::bias)]],
                              device float4* output [[buffer(convolution_binding::output)]],
                              constant ConvolutionParams& p
                              [[buffer(convolution_binding::parameters)]],
                              uint tile [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float4 shared_weights[8 * 32];
    threadgroup float shared_input[32 * 8];
    convolution_tile<32>(input, weights, bias, output, p, tile, tid, shared_weights, shared_input);
}

kernel void convolution_tiled64(device const float* input [[buffer(convolution_binding::input)]],
                                device const float4* weights
                                [[buffer(convolution_binding::weights)]],
                                device const float4* bias [[buffer(convolution_binding::bias)]],
                                device float4* output [[buffer(convolution_binding::output)]],
                                constant ConvolutionParams& p
                                [[buffer(convolution_binding::parameters)]],
                                uint tile [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float4 shared_weights[8 * 16];
    threadgroup float shared_input[64 * 8];
    convolution_tile<16>(input, weights, bias, output, p, tile, tid, shared_weights, shared_input);
}

kernel void matched(device const float* input [[buffer(convolution_binding::input)]],
                    device const float4* weights [[buffer(convolution_binding::weights)]],
                    device const float4* bias [[buffer(convolution_binding::bias)]],
                    device float4* output [[buffer(convolution_binding::output)]],
                    constant ConvolutionParams& p [[buffer(convolution_binding::parameters)]],
                    uint tile [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float4 w[192];
    threadgroup float x[256];
    convolution_tile<24, 96, 8, false>(input, weights, bias, output, p, tile, tid, w, x);
}

kernel void point96(device const float* input [[buffer(convolution_binding::input)]],
                    device const float4* weights [[buffer(convolution_binding::weights)]],
                    device const float4* bias [[buffer(convolution_binding::bias)]],
                    device float4* output [[buffer(convolution_binding::output)]],
                    constant ConvolutionParams& p [[buffer(convolution_binding::parameters)]],
                    uint tile [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float4 w[384];
    threadgroup float x[512];
    convolution_tile<24, 96, 16, true>(input, weights, bias, output, p, tile, tid, w, x);
}

kernel void point64(device const float* input [[buffer(convolution_binding::input)]],
                    device const float4* weights [[buffer(convolution_binding::weights)]],
                    device const float4* bias [[buffer(convolution_binding::bias)]],
                    device float4* output [[buffer(convolution_binding::output)]],
                    constant ConvolutionParams& p [[buffer(convolution_binding::parameters)]],
                    uint tile [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float4 w[256];
    threadgroup float x[1024];
    convolution_tile<16, 128, 16, true>(input, weights, bias, output, p, tile, tid, w, x);
}
