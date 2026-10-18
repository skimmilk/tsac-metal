#include "../abi/bindings.h"
#include "numeric.h"
#include "parameters.h"
#include "quantization.h"

using namespace metal;
using namespace tsac_metal_abi;
using namespace tsac_probability;

template <class Q>
inline void quant_bf16_parallel_impl(device const ushort* x, device Q* y, device ushort* scales,
                                     device atomic_uint* error, constant QuantizationParams& p,
                                     uint block, uint lane, threadgroup QuantizationShared& shared)
{
    uint at = block * 256 + lane;
    ushort bits = p.apply_gelu ? gelu_value(x[at]) : x[at];
    bool invalid = (bits & 0x7f80u) == 0x7f80u;
    if (simd_any(invalid) && lane % 32 == 0)
    {
        atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    }
    float value = invalid ? 0.0f : flush32(fp(bits));
    float maximum = simd_max(abs(value));
    if (lane % 32 == 0)
    {
        shared.maxima[lane / 32] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 32)
    {
        maximum = simd_max(lane < 8 ? shared.maxima[lane] : 0.0f);
        if (lane == 0)
        {
            ushort s = bf(max(div32(maximum, 127.0f), 1e-35f));
            scales[block] = s;
            shared.inverse = div32(1.0f, fp(s));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    y[at] = char(clamp(int(rint(flush32(value * shared.inverse))), -128, 127));
}

kernel void quant_bf16_parallel(
    device const ushort* x [[buffer(quantization_checked_binding::input)]],
    device char* y [[buffer(quantization_checked_binding::output)]],
    device ushort* scales [[buffer(quantization_checked_binding::scales)]],
    device atomic_uint* error [[buffer(quantization_checked_binding::error)]],
    constant QuantizationParams& p [[buffer(quantization_checked_binding::parameters)]],
    uint block [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]])
{
    threadgroup QuantizationShared shared;
    quant_bf16_parallel_impl(x, y, scales, error, p, block, lane, shared);
}

kernel void quant_bf16_parallel_wide(
    device const ushort* x [[buffer(quantization_checked_binding::input)]],
    device int* y [[buffer(quantization_checked_binding::output)]],
    device ushort* scales [[buffer(quantization_checked_binding::scales)]],
    device atomic_uint* error [[buffer(quantization_checked_binding::error)]],
    constant QuantizationParams& p [[buffer(quantization_checked_binding::parameters)]],
    uint block [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]])
{
    threadgroup QuantizationShared shared;
    quant_bf16_parallel_impl(x, y, scales, error, p, block, lane, shared);
}
