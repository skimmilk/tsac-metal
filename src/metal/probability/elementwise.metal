#include "../abi/bindings.h"
#include "numeric.h"
#include "parameters.h"

using namespace metal;
using namespace tsac_metal_abi;
using namespace tsac_probability;

kernel void add_kernel(device const ushort* x [[buffer(add_binding::input)]],
                       device const ushort* z [[buffer(add_binding::right)]],
                       device ushort* y [[buffer(add_binding::output)]],
                       constant AddParams& p [[buffer(add_binding::parameters)]],
                       uint id [[thread_position_in_grid]])
{
    if (id < p.element_count)
    {
        y[id] = bf(fp(x[id]) + fp(z[p.right_period_elements ? id % p.right_period_elements : id]));
    }
}

kernel void copy_bf16(device const ushort* x [[buffer(copy_binding::input)]],
                      device ushort* y [[buffer(copy_binding::output)]],
                      constant CopyParams& p [[buffer(copy_binding::parameters)]],
                      uint id [[thread_position_in_grid]])
{
    if (id < p.element_count)
    {
        y[id] = x[id];
    }
}
