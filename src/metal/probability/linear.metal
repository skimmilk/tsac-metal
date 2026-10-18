#include "../abi/bindings.h"
#include "numeric.h"
#include "parameters.h"

using namespace metal;
using namespace tsac_metal_abi;
using namespace tsac_probability;

kernel void linear_kernel(device const char* weights [[buffer(projection_binding::weights)]],
                          device const float* scales [[buffer(projection_binding::scales)]],
                          device const float* bias [[buffer(projection_binding::bias)]],
                          device const char* x [[buffer(projection_binding::input)]],
                          device const ushort* activation_scales
                          [[buffer(projection_binding::activation_scales)]],
                          device ushort* y [[buffer(projection_binding::output)]],
                          constant ProjectionParams& p [[buffer(projection_binding::parameters)]],
                          uint id [[thread_position_in_grid]])
{
    // One complete SIMD group cooperates on an output. Integer reduction is
    // exact; floating scales/FMA remain in their original block order.
    uint output_id = id / 32, lane = id % 32;
    if (output_id >= p.column_count * p.output_width)
    {
        return;
    }
    uint col = output_id / p.output_width, out = output_id % p.output_width,
         blocks = p.input_width / 256;
    float sum = 0;
    for (uint block = 0; block < blocks; ++block)
    {
        int partial = 0;
        // A 32-thread SIMD group covers the 256-input block in two rounds of
        // char4 loads per lane. Integer sums fit int32, so the SIMD reduction
        // is exact. Apply the scale only to the full dot, keep floating block
        // FMAs ordered, and defer bias/BF16 rounding until all blocks finish.
        for (uint i = lane * 4; i < 256; i += 128)
        {
            int4 w = int4(*reinterpret_cast<device const char4*>(weights + out * p.input_width +
                                                                 block * 256 + i));
            int4 a = int4(
                *reinterpret_cast<device const char4*>(x + col * p.input_width + block * 256 + i));
            int4 products = w * a;
            partial += products.x + products.y + products.z + products.w;
        }
        int dot = simd_sum(partial);
        float scale =
            scales[block * p.output_width + out] * fp(activation_scales[col * blocks + block]);
        sum = fma(float(dot), scale, sum);
    }
    if (lane == 0)
    {
        y[output_id] = bf(sum + bias[out]);
    }
}

// Small batches share activation loads across output rows and packed weight
// loads across columns. All partial sums stay integer until block scaling.
template <uint R>
inline void linear_small_impl(device const char* weights, device const float* scales,
                              device const float* bias, device const char* x,
                              device const ushort* activation_scales, device ushort* y,
                              constant ProjectionParams& p, uint id)
{
    uint rows = (p.output_width + R - 1) / R, out_base = (id / 32) % rows * R,
         col = (id / 32) / rows * 3, lane = id % 32, blocks = p.input_width / 256;
    if (col >= p.column_count)
    {
        return;
    }
    float sums[R][3];
    for (uint r = 0; r < R; ++r)
    {
        for (uint c = 0; c < 3; ++c)
        {
            sums[r][c] = 0;
        }
    }
    for (uint block = 0; block < blocks; ++block)
    {
        int partial[R][3];
        for (uint r = 0; r < R; ++r)
        {
            for (uint c = 0; c < 3; ++c)
            {
                partial[r][c] = 0;
            }
        }
        for (uint i = lane * 4; i < 256; i += 128)
        {
            int4 activation[3];
            for (uint c = 0; c < 3; ++c)
            {
                if (col + c < p.column_count)
                {
                    activation[c] = int4(*reinterpret_cast<device const char4*>(
                        x + (col + c) * p.input_width + block * 256 + i));
                }
            }
            for (uint r = 0; r < R; ++r)
            {
                if (out_base + r < p.output_width)
                {
                    int4 w = int4(*reinterpret_cast<device const char4*>(
                        weights + (out_base + r) * p.input_width + block * 256 + i));
                    for (uint c = 0; c < 3; ++c)
                    {
                        if (col + c < p.column_count)
                        {
                            int4 product = w * activation[c];
                            partial[r][c] += product.x + product.y + product.z + product.w;
                        }
                    }
                }
            }
        }
        for (uint r = 0; r < R; ++r)
        {
            if (out_base + r < p.output_width)
            {
                for (uint c = 0; c < 3; ++c)
                {
                    if (col + c < p.column_count)
                    {
                        int dot = simd_sum(partial[r][c]);
                        float scale = scales[block * p.output_width + out_base + r] *
                                      fp(activation_scales[(col + c) * blocks + block]);
                        sums[r][c] = fma(float(dot), scale, sums[r][c]);
                    }
                }
            }
        }
    }
    if (lane == 0)
    {
        for (uint r = 0; r < R; ++r)
        {
            if (out_base + r < p.output_width)
            {
                for (uint c = 0; c < 3; ++c)
                {
                    if (col + c < p.column_count)
                    {
                        y[(col + c) * p.output_width + out_base + r] =
                            bf(sums[r][c] + bias[out_base + r]);
                    }
                }
            }
        }
    }
}

kernel void linear_small2(device const char* w [[buffer(projection_binding::weights)]],
                          device const float* s [[buffer(projection_binding::scales)]],
                          device const float* b [[buffer(projection_binding::bias)]],
                          device const char* x [[buffer(projection_binding::input)]],
                          device const ushort* a [[buffer(projection_binding::activation_scales)]],
                          device ushort* y [[buffer(projection_binding::output)]],
                          constant ProjectionParams& p [[buffer(projection_binding::parameters)]],
                          uint id [[thread_position_in_grid]])
{
    linear_small_impl<2>(w, s, b, x, a, y, p, id);
}

kernel void linear_small4(device const char* w [[buffer(projection_binding::weights)]],
                          device const float* s [[buffer(projection_binding::scales)]],
                          device const float* b [[buffer(projection_binding::bias)]],
                          device const char* x [[buffer(projection_binding::input)]],
                          device const ushort* a [[buffer(projection_binding::activation_scales)]],
                          device ushort* y [[buffer(projection_binding::output)]],
                          constant ProjectionParams& p [[buffer(projection_binding::parameters)]],
                          uint id [[thread_position_in_grid]])
{
    linear_small_impl<4>(w, s, b, x, a, y, p, id);
}

// Interleave input quartets across output rows. Each SIMD group uses narrower
// exact integer reductions; scaled floating blocks remain strictly ordered.
// Producers sign-extend the same clipped int8 values into int32, avoiding
// repeated activation unpacking in every output row group. Weights stay int8.
template <uint R, bool residual = false>
inline void packed3(device const char4* w, device const float* s, device const float* b,
                    device const int4* x, device const ushort* a, device ushort* y,
                    constant ProjectionParams& p, uint id, device const ushort* skip = nullptr)
{
    uint lane = id % 32, row = lane % R, k = lane / R, out = id / 32 * R + row,
         blocks = p.input_width / 256;
    float3 sums = 0;
    for (uint block = 0; block < blocks; ++block)
    {
        int3 dots = 0;
        for (uint i = k; i < 64; i += 32 / R)
        {
            int4 weight = int4(w[((out / R * blocks + block) * 64 + i) * R + row]);
            int4 x0 = int4(x[block * 64 + i]), x1 = int4(x[(blocks + block) * 64 + i]),
                 x2 = int4(x[(2 * blocks + block) * 64 + i]);
            int4 d0 = weight * x0, d1 = weight * x1, d2 = weight * x2;
            dots += int3(d0.x + d0.y + d0.z + d0.w, d1.x + d1.y + d1.z + d1.w,
                         d2.x + d2.y + d2.z + d2.w);
        }
        for (uint delta = 16; delta >= R; delta /= 2)
        {
            dots += simd_shuffle_down(dots, delta);
        }
        float3 scale = s[block * p.output_width + out] *
                       float3(fp(a[block]), fp(a[blocks + block]), fp(a[2 * blocks + block]));
        sums = fma(float3(dots), scale, sums);
    }
    if (k == 0)
    {
        for (uint c = 0; c < 3; ++c)
        {
            ushort rounded = bf(sums[c] + b[out]);
            y[c * p.output_width + out] =
                residual ? bf(fp(rounded) + fp(skip[c * p.output_width + out])) : rounded;
        }
    }
}

kernel void linear_packed_wide2(device const char4* w [[buffer(projection_binding::weights)]],
                                device const float* s [[buffer(projection_binding::scales)]],
                                device const float* b [[buffer(projection_binding::bias)]],
                                device const int4* x [[buffer(projection_binding::input)]],
                                device const ushort* a
                                [[buffer(projection_binding::activation_scales)]],
                                device ushort* y [[buffer(projection_binding::output)]],
                                constant ProjectionParams& p
                                [[buffer(projection_binding::parameters)]],
                                uint id [[thread_position_in_grid]])
{
    packed3<2>(w, s, b, x, a, y, p, id);
}

kernel void linear_packed_wide8(device const char4* w [[buffer(projection_binding::weights)]],
                                device const float* s [[buffer(projection_binding::scales)]],
                                device const float* b [[buffer(projection_binding::bias)]],
                                device const int4* x [[buffer(projection_binding::input)]],
                                device const ushort* a
                                [[buffer(projection_binding::activation_scales)]],
                                device ushort* y [[buffer(projection_binding::output)]],
                                constant ProjectionParams& p
                                [[buffer(projection_binding::parameters)]],
                                uint id [[thread_position_in_grid]])
{
    packed3<8>(w, s, b, x, a, y, p, id);
}

kernel void
linear_packed_add_wide2(device const char4* w [[buffer(projection_binding::weights)]],
                        device const float* s [[buffer(projection_binding::scales)]],
                        device const float* b [[buffer(projection_binding::bias)]],
                        device const int4* x [[buffer(projection_binding::input)]],
                        device const ushort* a [[buffer(projection_binding::activation_scales)]],
                        device ushort* y [[buffer(projection_binding::output)]],
                        device const ushort* skip [[buffer(projection_binding::residual)]],
                        constant ProjectionParams& p [[buffer(projection_binding::parameters)]],
                        uint id [[thread_position_in_grid]])
{
    packed3<2, true>(w, s, b, x, a, y, p, id, skip);
}

kernel void
linear_packed_add_wide8(device const char4* w [[buffer(projection_binding::weights)]],
                        device const float* s [[buffer(projection_binding::scales)]],
                        device const float* b [[buffer(projection_binding::bias)]],
                        device const int4* x [[buffer(projection_binding::input)]],
                        device const ushort* a [[buffer(projection_binding::activation_scales)]],
                        device ushort* y [[buffer(projection_binding::output)]],
                        device const ushort* skip [[buffer(projection_binding::residual)]],
                        constant ProjectionParams& p [[buffer(projection_binding::parameters)]],
                        uint id [[thread_position_in_grid]])
{
    packed3<8, true>(w, s, b, x, a, y, p, id, skip);
}
