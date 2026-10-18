#include "../abi/bindings.h"
#include "parameters.h"

#include <metal_stdlib>

using namespace metal;
using namespace tsac_metal_abi;

kernel void rvq_search(device const float* vectors [[buffer(rvq_search_binding::input)]],
                       device const float* codebook [[buffer(rvq_search_binding::codebook)]],
                       device ushort* selected [[buffer(rvq_search_binding::output)]],
                       constant RvqSearchParams& p [[buffer(rvq_search_binding::parameters)]],
                       uint id [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_threadgroup]])
{
    if (id >= p.vector_count)
    {
        return;
    }
    float query[8];
    for (uint i = 0; i < 8; ++i)
    {
        query[i] = vectors[id * 8 + i];
    }
    if (p.normalize_input)
    {
        float squares[8];
        for (uint i = 0; i < 8; ++i)
        {
            squares[i] = query[i] * query[i];
        }
        for (uint width = 8; width > 1; width /= 2)
        {
            for (uint i = 0; i < width / 2; ++i)
            {
                squares[i] = squares[i * 2] + squares[i * 2 + 1];
            }
        }
        if (!isfinite(squares[0]))
        {
            if (lane == 0)
            {
                selected[id] = 0xffff;
            }
            return;
        }
        float mean = squares[0] * 0.125f;
        float inverse_rms = 1.0f / sqrt(mean + as_type<float>(0x2b8cbcccu));
        for (uint i = 0; i < 8; ++i)
        {
            float normalized = query[i] * inverse_rms;
            query[i] = normalized * as_type<float>(0x3eb504f3u);
        }
    }
    float best = -INFINITY;
    ushort index = 0;
    for (uint entry = lane; entry < 1024; entry += 32)
    {
        float dot = 0;
        for (uint i = 0; i < 8; ++i)
        {
            dot = fma(codebook[entry * 8 + i], query[i], dot);
        }
        if (dot > best)
        {
            best = dot;
            index = ushort(entry);
        }
    }
    float maximum = simd_max(best);
    uint winner = simd_min(best == maximum ? uint(index) : 1024u);
    if (lane == 0)
    {
        selected[id] = ushort(winner);
    }
}

kernel void rvq_project(device const ushort* tokens [[buffer(rvq_project_binding::tokens)]],
                        device const float* codebook [[buffer(rvq_project_binding::codebook)]],
                        device const float* weights [[buffer(rvq_project_binding::weights)]],
                        device const float* bias [[buffer(rvq_project_binding::bias)]],
                        device float* output [[buffer(rvq_project_binding::output)]],
                        device atomic_uint* error [[buffer(rvq_project_binding::error)]],
                        constant RvqProjectParams& p [[buffer(rvq_project_binding::parameters)]],
                        uint id [[thread_position_in_grid]])
{
    if (id >= p.token_count * 1024)
    {
        return;
    }
    uint channel = id % 1024, frame = id / 1024;
    uint token = tokens[frame];
    if (token >= 1024)
    {
        atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        output[id] = 0;
        return;
    }
    uint offset = token * 8;
    float sum = 0;
    for (uint i = 0; i < 8; ++i)
    {
        sum = fma(weights[channel * 8 + i], codebook[offset + i], sum);
    }
    float result = sum + bias[channel];
    if (!isfinite(result))
    {
        atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    }
    output[id] = result;
}

kernel void
rvq_accumulate(device const ushort* tokens [[buffer(rvq_accumulate_binding::tokens)]],
               device const float* codebook [[buffer(rvq_accumulate_binding::codebook)]],
               device const float* weights [[buffer(rvq_accumulate_binding::weights)]],
               device const float* bias [[buffer(rvq_accumulate_binding::bias)]],
               device const float* residual [[buffer(rvq_accumulate_binding::residual)]],
               device float* output [[buffer(rvq_accumulate_binding::output)]],
               device atomic_uint* error [[buffer(rvq_accumulate_binding::error)]],
               constant RvqAccumulateParams& p [[buffer(rvq_accumulate_binding::parameters)]],
               uint id [[thread_position_in_grid]])
{
    if (id >= p.element_count)
    {
        return;
    }
    uint channel = id % 1024, token = tokens[id / 1024];
    if (token >= 1024)
    {
        atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
        output[id] = 0;
        return;
    }
    float sum = 0;
    for (uint i = 0; i < 8; ++i)
    {
        sum = fma(weights[channel * 8 + i], codebook[token * 8 + i], sum);
    }
    float contribution = sum + bias[channel];
    float result = p.subtract ? residual[id] - contribution : residual[id] + contribution;
    if (!isfinite(contribution) || !isfinite(result))
    {
        atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    }
    output[id] = result;
}
