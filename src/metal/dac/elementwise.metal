#include "../abi/bindings.h"
#include "parameters.h"

#include <metal_stdlib>

using namespace metal;
using namespace tsac_metal_abi;

kernel void residual(device const float* left [[buffer(residual_binding::left)]],
                     device const float* right [[buffer(residual_binding::right)]],
                     device float* output [[buffer(residual_binding::output)]],
                     device atomic_uint* error [[buffer(residual_binding::error)]],
                     constant ResidualParams& p [[buffer(residual_binding::parameters)]],
                     uint id [[thread_position_in_grid]])
{
    if (id >= p.element_count)
    {
        return;
    }
    float result = p.subtract ? left[id] - right[id] : left[id] + right[id];
    if (!isfinite(result))
    {
        atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    }
    output[id] = result;
}

kernel void check_finite(device const float* x [[buffer(check_finite_binding::input)]],
                         device atomic_uint* error [[buffer(check_finite_binding::error)]],
                         constant CheckFiniteParams& p [[buffer(check_finite_binding::parameters)]],
                         uint id [[thread_position_in_grid]])
{
    bool bad = id < p.element_count && !isfinite(x[id]);
    if (simd_any(bad) && id % 32 == 0)
    {
        atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    }
}
