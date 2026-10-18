#include "../abi/bindings.h"
#include "numeric.h"
#include "parameters.h"

using namespace metal;
using namespace tsac_metal_abi;
using namespace tsac_probability;

// Named shared state makes the cross-thread broadcasts explicit. Thread zero
// writes each result before the group barrier; other threads then consume it.
struct NormQuantShared
{
    float sums[4], squares[4], affine[2], maxima[4], inverse[2];
};

// Contiguous four-value leaves and a binary tree preserve the original sum64
// hierarchy, including its 64-value block carries (addition is commutative).
void norm512_stats(device const ushort* x, float epsilon, uint lane, threadgroup float* sums,
                   threadgroup float* squares, threadgroup float* affine)
{
    uint base = lane * 4;
    float a = fp(x[base]), b = fp(x[base + 1]), c = fp(x[base + 2]), d = fp(x[base + 3]);
    float sum = add32(add32(a, b), add32(c, d)),
          square = add32(add32(a * a, b * b), add32(c * c, d * d));
    for (uint stride = 1; stride < 32; stride *= 2)
    {
        float next = simd_shuffle_down(sum, stride),
              next_square = simd_shuffle_down(square, stride);
        if (lane % (2 * stride) == 0)
        {
            sum = add32(sum, next);
            square = add32(square, next_square);
        }
    }
    if (lane % 32 == 0)
    {
        sums[lane / 32] = sum;
        squares[lane / 32] = square;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0)
    {
        sums[0] = add32(add32(sums[0], sums[1]), add32(sums[2], sums[3]));
        squares[0] = add32(add32(squares[0], squares[1]), add32(squares[2], squares[3]));
    }
    if (lane == 0)
    {
        // The original scalar reduction folds empty carry slots (+0) into the final
        // power-of-two block, including the sign of an all-negative-zero sum.
        float mean = add32(0.0f, sums[0]) * div32(1.0f, 512.0f);
        float variance = fma(add32(0.0f, squares[0]), div32(1.0f, 512.0f), 0.0f);
        variance = fma(mean, -mean, variance);
        variance = variance + epsilon;
        affine[0] = div32(1.0f, precise::sqrt(variance));
        affine[1] = -mean * affine[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void norm512_parallel(device const ushort* x [[buffer(normalization_binding::input)]],
                             device const ushort* gain [[buffer(normalization_binding::gain)]],
                             device const ushort* bias [[buffer(normalization_binding::bias)]],
                             device ushort* y [[buffer(normalization_binding::output)]],
                             device atomic_uint* error [[buffer(normalization_binding::error)]],
                             constant NormalizationParams& p
                             [[buffer(normalization_binding::parameters)]],
                             uint lane [[thread_index_in_threadgroup]],
                             uint column [[threadgroup_position_in_grid]])
{
    x += column * 512;
    y += column * 512;
    threadgroup float sums[4], squares[4], affine[2];
    norm512_stats(x, p.epsilon, lane, sums, squares, affine);
    bool invalid = false;
    for (uint i = lane * 4; i < lane * 4 + 4; ++i)
    {
        float normalized = fma(fp(x[i]), affine[0], affine[1]);
        ushort value = bf(fma(normalized, fp(gain[i]), fp(bias[i])));
        invalid |= (value & 0x7f80u) == 0x7f80u;
        y[i] = value;
    }
    if (simd_any(invalid) && lane % 32 == 0)
    {
        atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    }
}

template <class Q>
inline void norm_quant512_impl(device const ushort* x, device const ushort* gain,
                               device const ushort* bias, device Q* y, device ushort* scales,
                               device atomic_uint* error, constant NormalizationParams& p,
                               uint lane, uint column, threadgroup NormQuantShared& shared)
{
    x += column * 512;
    y += column * 512;
    scales += column * 2;
    norm512_stats(x, p.epsilon, lane, shared.sums, shared.squares, shared.affine);
    float values[4];
    float maximum = 0;
    bool invalid = false;
    for (uint j = 0; j < 4; ++j)
    {
        uint i = lane * 4 + j;
        float normalized = fma(fp(x[i]), shared.affine[0], shared.affine[1]);
        ushort bits = bf(fma(normalized, fp(gain[i]), fp(bias[i])));
        bool bad = (bits & 0x7f80u) == 0x7f80u;
        invalid |= bad;
        values[j] = bad ? 0.0f : flush32(fp(bits));
        maximum = max(maximum, abs(values[j]));
    }
    if (simd_any(invalid) && lane % 32 == 0)
    {
        atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    }
    maximum = simd_max(maximum);
    if (lane % 32 == 0)
    {
        shared.maxima[lane / 32] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane % 64 == 0)
    {
        uint block = lane / 64;
        maximum = max(shared.maxima[block * 2], shared.maxima[block * 2 + 1]);
        ushort scale = bf(max(div32(maximum, 127.0f), 1e-35f));
        scales[block] = scale;
        shared.inverse[block] = div32(1.0f, fp(scale));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < 4; ++j)
    {
        y[lane * 4 + j] =
            char(clamp(int(rint(flush32(values[j] * shared.inverse[lane / 64]))), -128, 127));
    }
}

kernel void norm_quant512(
    device const ushort* x [[buffer(normalization_quantized_binding::input)]],
    device const ushort* gain [[buffer(normalization_quantized_binding::gain)]],
    device const ushort* bias [[buffer(normalization_quantized_binding::bias)]],
    device char* y [[buffer(normalization_quantized_binding::output)]],
    device ushort* scales [[buffer(normalization_quantized_binding::scales)]],
    device atomic_uint* error [[buffer(normalization_quantized_binding::error)]],
    constant NormalizationParams& p [[buffer(normalization_quantized_binding::parameters)]],
    uint lane [[thread_index_in_threadgroup]], uint column [[threadgroup_position_in_grid]])
{
    threadgroup NormQuantShared shared;
    norm_quant512_impl(x, gain, bias, y, scales, error, p, lane, column, shared);
}

kernel void norm_quant512_wide(
    device const ushort* x [[buffer(normalization_quantized_binding::input)]],
    device const ushort* gain [[buffer(normalization_quantized_binding::gain)]],
    device const ushort* bias [[buffer(normalization_quantized_binding::bias)]],
    device int* y [[buffer(normalization_quantized_binding::output)]],
    device ushort* scales [[buffer(normalization_quantized_binding::scales)]],
    device atomic_uint* error [[buffer(normalization_quantized_binding::error)]],
    constant NormalizationParams& p [[buffer(normalization_quantized_binding::parameters)]],
    uint lane [[thread_index_in_threadgroup]], uint column [[threadgroup_position_in_grid]])
{
    threadgroup NormQuantShared shared;
    norm_quant512_impl(x, gain, bias, y, scales, error, p, lane, column, shared);
}
