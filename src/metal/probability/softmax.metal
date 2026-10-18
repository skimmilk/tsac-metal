#include "../abi/bindings.h"
#include "numeric.h"
#include "parameters.h"

using namespace metal;
using namespace tsac_metal_abi;
using namespace tsac_probability;

struct SoftmaxShared
{
    float maximum, reciprocal;
};

template <bool Decode>
inline void softmax1024_impl(device const ushort* x, device float* tree_output, device ushort* y,
                             device atomic_uint* error, constant SoftmaxParams&, uint lane,
                             uint column, threadgroup float* scratch,
                             threadgroup SoftmaxShared& shared, threadgroup float* tree)
{
    x += column * 1024;
    y += column * 1024;
    uint base = lane * 4;
    ushort4 bits = ushort4(x[base], x[base + 1], x[base + 2], x[base + 3]);
    bool4 invalid = (bits & ushort4(0x7f80u)) == ushort4(0x7f80u);
    if (simd_any(any(invalid)) && lane % 32 == 0)
    {
        atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    }
    bits = select(bits, ushort4(0), invalid);
    float a = fp(bits.x), b = fp(bits.y), c = fp(bits.z), d = fp(bits.w);
    float local_max = simd_max(max(max(a, b), max(c, d)));
    if (lane % 32 == 0)
    {
        scratch[lane / 32] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 32)
    {
        local_max = simd_max(lane < 8 ? scratch[lane] : -INFINITY);
        if (lane == 0)
        {
            shared.maximum = local_max;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    a = exp_original(a - shared.maximum);
    b = exp_original(b - shared.maximum);
    c = exp_original(c - shared.maximum);
    d = exp_original(d - shared.maximum);
    float sum = add32(add32(a, b), add32(c, d));
    for (uint stride = 1; stride < 32; stride *= 2)
    {
        float next = simd_shuffle_down(sum, stride);
        if (lane % (stride * 2) == 0)
        {
            sum = add32(sum, next);
        }
    }
    if (lane % 32 == 0)
    {
        scratch[lane / 32] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0)
    {
        shared.reciprocal =
            div32(1.0f, add32(add32(add32(scratch[0], scratch[1]), add32(scratch[2], scratch[3])),
                              add32(add32(scratch[4], scratch[5]), add32(scratch[6], scratch[7]))));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    y[base] = bf(a * shared.reciprocal);
    y[base + 1] = bf(b * shared.reciprocal);
    y[base + 2] = bf(c * shared.reciprocal);
    y[base + 3] = bf(d * shared.reciprocal);
    if (Decode)
    {
        // Each left subtree has power-of-two size. These adjacent-pair sums
        // reproduce probability_sum's 64-value blocks and carry tree for
        // nonnegative BF16 probabilities, including FTZ at each addition.
        tree[512 + lane * 2] = add32(fp(y[base]), fp(y[base + 1]));
        tree[513 + lane * 2] = add32(fp(y[base + 2]), fp(y[base + 3]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // The decoder reads only left children (even indices). Both children
        // are still needed for internal sums, but the root is never consumed.
        for (uint start = 256; start > 1; start /= 2)
        {
            if (lane < start)
            {
                tree[start + lane] = add32(tree[(start + lane) * 2], tree[(start + lane) * 2 + 1]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint at = (lane + 1) * 2; at < 1024; at += 512)
        {
            tree_output[column * 1024 + at] = tree[at];
        }
    }
}

kernel void softmax1024_parallel(device const ushort* x [[buffer(softmax_binding::input)]],
                                 device ushort* y [[buffer(softmax_binding::output)]],
                                 device atomic_uint* error [[buffer(softmax_binding::error)]],
                                 constant SoftmaxParams& p [[buffer(softmax_binding::parameters)]],
                                 uint lane [[thread_index_in_threadgroup]],
                                 uint column [[threadgroup_position_in_grid]])
{
    threadgroup float scratch[8];
    threadgroup SoftmaxShared shared;
    softmax1024_impl<false>(x, nullptr, y, error, p, lane, column, scratch, shared, nullptr);
}

kernel void softmax1024_decode(device const ushort* x [[buffer(softmax_decode_binding::input)]],
                               device float* tree_output [[buffer(softmax_decode_binding::tree)]],
                               device ushort* y [[buffer(softmax_decode_binding::output)]],
                               device atomic_uint* error [[buffer(softmax_decode_binding::error)]],
                               constant SoftmaxParams& p
                               [[buffer(softmax_decode_binding::parameters)]],
                               uint lane [[thread_index_in_threadgroup]],
                               uint column [[threadgroup_position_in_grid]])
{
    threadgroup float scratch[8], tree[1024];
    threadgroup SoftmaxShared shared;
    softmax1024_impl<true>(x, tree_output, y, error, p, lane, column, scratch, shared, tree);
}
