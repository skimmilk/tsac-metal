#include "../abi/bindings.h"
#include "parameters.h"

#include <metal_stdlib>

using namespace metal;
using namespace tsac_metal_abi;

inline float sine_squared(float phase)
{
    float quadrant = rint(phase * as_type<float>(0x3f22f983u));
    int q =
        quadrant >= -2147483648.0f && quadrant < 2147483648.0f ? int(quadrant) : (-2147483647 - 1);
    float x = fma(quadrant, as_type<float>(0xbfc90000u), phase);
    x = fma(quadrant, as_type<float>(0xb9fdaa22u), x);
    float x2 = x * x, value;
    if ((q & 1) == 0)
    {
        float polynomial = fma(as_type<float>(0xb94ca1f9u), x2, as_type<float>(0x3c08839eu));
        polynomial = fma(polynomial, x2, as_type<float>(0xbe2aaaa3u));
        float cube = x * x2;
        value = fma(cube, polynomial, x);
    }
    else
    {
        float polynomial = fma(as_type<float>(0x37ccf5ceu), x2, as_type<float>(0xbab6061au));
        polynomial = fma(polynomial, x2, as_type<float>(0x3d2aaaa5u));
        polynomial = fma(polynomial, x2, as_type<float>(0xbf000000u));
        value = fma(polynomial, x2, 1.0f);
    }
    return value * value;
}

kernel void snake(device const float* input [[buffer(snake_binding::input)]],
                  device const float* alpha [[buffer(snake_binding::alpha)]],
                  device float* output [[buffer(snake_binding::output)]],
                  device atomic_uint* error [[buffer(snake_binding::error)]],
                  constant SnakeParams& p [[buffer(snake_binding::parameters)]],
                  uint id [[thread_position_in_grid]])
{
    if (id >= p.element_count)
    {
        return;
    }
    float a = alpha[id % p.channel_count], x = input[id];
    float squared = sine_squared(a * x), denominator = a + as_type<float>(0x3089705fu);
    float quotient = squared / denominator;
    float result = x + quotient;
    if (!isfinite(result))
    {
        atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    }
    output[id] = result;
}
