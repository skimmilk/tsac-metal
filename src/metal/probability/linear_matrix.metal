#include "../abi/bindings.h"
#include "numeric.h"
#include "parameters.h"

using namespace metal;
using namespace tsac_metal_abi;
using namespace tsac_probability;

// Every unscaled 256-term dot is an integer with magnitude <=2^22. FP32
// matrix accumulators therefore reproduce the int32 sum exactly. Signed int8
// operands also convert to half tiles exactly. Scales are
// applied only after each block, in ascending order, before bias/BF16 rounding.
template <bool Add>
inline void
matrix_projection(device const char* weights, device const float* scales, device const float* bias,
                  device const char* x, device const ushort* activation_scales, device ushort* y,
                  device const ushort* residual, constant ProjectionParams& p, uint tile, uint tid,
                  threadgroup half* wa, threadgroup half* xa, threadgroup float* dots)
{
    uint rows = (p.output_width + 31) / 32, out_base = tile % rows * 32,
         col_base = tile / rows * 32, blocks = p.input_width / 256;
    uint simd = tid / 32, row = simd / 2 * 16, col = simd % 2 * 16;
    float sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint block = 0; block < blocks; ++block)
    {
        simdgroup_float8x8 a00(0.0f), a01(0.0f), a10(0.0f), a11(0.0f);
        for (uint chunk = 0; chunk < 256; chunk += 32)
        {
            for (uint i = tid; i < 1024; i += 128)
            {
                uint out = out_base + i / 32, k = chunk + i % 32;
                wa[i] = out < p.output_width ? half(weights[out * p.input_width + block * 256 + k])
                                             : half(0);
                uint c = col_base + i % 32;
                k = chunk + i / 32;
                xa[i] = c < p.column_count ? half(x[c * p.input_width + block * 256 + k]) : half(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < 32; k += 8)
            {
                simdgroup_half8x8 w0, w1, x0, x1;
                simdgroup_load(w0, wa + row * 32 + k, 32);
                simdgroup_load(w1, wa + (row + 8) * 32 + k, 32);
                simdgroup_load(x0, xa + k * 32 + col, 32);
                simdgroup_load(x1, xa + k * 32 + col + 8, 32);
                simdgroup_multiply_accumulate(a00, w0, x0, a00);
                simdgroup_multiply_accumulate(a01, w0, x1, a01);
                simdgroup_multiply_accumulate(a10, w1, x0, a10);
                simdgroup_multiply_accumulate(a11, w1, x1, a11);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        simdgroup_store(a00, dots + row * 32 + col, 32);
        simdgroup_store(a01, dots + row * 32 + col + 8, 32);
        simdgroup_store(a10, dots + (row + 8) * 32 + col, 32);
        simdgroup_store(a11, dots + (row + 8) * 32 + col + 8, 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < 8; ++j)
        {
            uint element = tid + j * 128, out = out_base + element / 32,
                 c = col_base + element % 32;
            if (out < p.output_width && c < p.column_count)
            {
                float scale = scales[block * p.output_width + out] *
                              fp(activation_scales[c * blocks + block]);
                sums[j] = fma(dots[element], scale, sums[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = 0; j < 8; ++j)
    {
        uint element = tid + j * 128, out = out_base + element / 32, c = col_base + element % 32;
        if (out < p.output_width && c < p.column_count)
        {
            uint at = c * p.output_width + out;
            ushort rounded = bf(sums[j] + bias[out]);
            y[at] = Add ? bf(fp(rounded) + fp(residual[at])) : rounded;
        }
    }
}

kernel void linear_matrix(device const char* w [[buffer(projection_binding::weights)]],
                          device const float* s [[buffer(projection_binding::scales)]],
                          device const float* b [[buffer(projection_binding::bias)]],
                          device const char* x [[buffer(projection_binding::input)]],
                          device const ushort* a [[buffer(projection_binding::activation_scales)]],
                          device ushort* y [[buffer(projection_binding::output)]],
                          constant ProjectionParams& p [[buffer(projection_binding::parameters)]],
                          uint tile [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]])
{
    threadgroup half wa[1024], xa[1024];
    threadgroup float dots[1024];
    matrix_projection<false>(w, s, b, x, a, y, y, p, tile, tid, wa, xa, dots);
}

kernel void
linear_matrix_add(device const char* w [[buffer(projection_binding::weights)]],
                  device const float* s [[buffer(projection_binding::scales)]],
                  device const float* b [[buffer(projection_binding::bias)]],
                  device const char* x [[buffer(projection_binding::input)]],
                  device const ushort* a [[buffer(projection_binding::activation_scales)]],
                  device ushort* y [[buffer(projection_binding::output)]],
                  device const ushort* residual [[buffer(projection_binding::residual)]],
                  constant ProjectionParams& p [[buffer(projection_binding::parameters)]],
                  uint tile [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]])
{
    threadgroup half wa[1024], xa[1024];
    threadgroup float dots[1024];
    matrix_projection<true>(w, s, b, x, a, y, residual, p, tile, tid, wa, xa, dots);
}
