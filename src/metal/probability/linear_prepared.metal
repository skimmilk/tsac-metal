#include "../abi/bindings.h"
#include "numeric.h"
#include "parameters.h"

using namespace metal;
using namespace tsac_metal_abi;
using namespace tsac_probability;

// Full preparation/projection fusion. Eight SIMD groups share prepared
// activations; each owns R output rows and up to three independent columns.
// Integer dots may regroup, but scaled block FMAs and BF16 boundaries may not.
template <uint R, uint C, uint S>
inline void prepared_projection(device const char4* w, device const float* scales,
                                device const float* bias, device const ushort* x, device ushort* y,
                                device const ushort* gain, device const ushort* norm_bias,
                                device const ushort* residual, device atomic_uint* error,
                                constant PreparedProjectionParams& p, uint group, uint tid,
                                threadgroup char* quantized, threadgroup ushort* activation_scales,
                                threadgroup ushort* normalized, threadgroup float* sums,
                                threadgroup float* squares, threadgroup float* affine)
{
    uint rows = (p.output_width + S * R - 1) / (S * R), column = group / rows * C, simd = tid / 32,
         lane = tid % 32;
    uint blocks = p.input_width / 256;
    for (uint c = 0; c < C && column + c < p.column_count; ++c)
    {
        if (p.normalize_input)
        {
            constexpr uint N = 512 / (S * 32);
            uint at = tid * N;
            float vals[N], sq[N];
            for (uint j = 0; j < N; ++j)
            {
                vals[j] = fp(x[(column + c) * 512 + at + j]);
                sq[j] = vals[j] * vals[j];
            }
            for (uint n = N / 2; n > 0; n /= 2)
            {
                for (uint j = 0; j < n; ++j)
                {
                    vals[j] = add32(vals[j * 2], vals[j * 2 + 1]);
                    sq[j] = add32(sq[j * 2], sq[j * 2 + 1]);
                }
            }
            float sum = vals[0], square = sq[0];
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
            if (!lane)
            {
                sums[simd] = sum;
                squares[simd] = square;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!tid)
            {
                for (uint n = S / 2; n > 0; n /= 2)
                {
                    for (uint j = 0; j < n; ++j)
                    {
                        sums[j] = add32(sums[j * 2], sums[j * 2 + 1]);
                        squares[j] = add32(squares[j * 2], squares[j * 2 + 1]);
                    }
                }
                float mean = add32(0.0f, sums[0]) * div32(1.0f, 512.0f);
                float variance = fma(add32(0.0f, squares[0]), div32(1.0f, 512.0f), 0.0f);
                variance = fma(mean, -mean, variance);
                variance = variance + p.epsilon;
                affine[0] = div32(1.0f, precise::sqrt(variance));
                affine[1] = -mean * affine[0];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint j = 0; j < N; ++j)
            {
                uint i = at + j;
                ushort value = bf(fma(fma(fp(x[(column + c) * 512 + i]), affine[0], affine[1]),
                                      fp(gain[i]), fp(norm_bias[i])));
                normalized[i] = value;
                if ((value & 0x7f80u) == 0x7f80u)
                {
                    atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint activation_block = simd; activation_block < blocks; activation_block += S)
        {
            float values[8];
            float maximum = 0;
            bool invalid = false;
            for (uint j = 0; j < 8; ++j)
            {
                uint i = activation_block * 256 + lane * 8 + j;
                ushort value =
                    p.normalize_input ? normalized[i] : x[(column + c) * p.input_width + i];
                if (p.apply_gelu)
                {
                    value = gelu_value(value);
                }
                bool bad = (value & 0x7f80u) == 0x7f80u;
                invalid |= bad;
                values[j] = bad ? 0.0f : flush32(fp(value));
                maximum = max(maximum, abs(values[j]));
            }
            if (simd_any(invalid) && !lane)
            {
                atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
            }
            maximum = simd_max(maximum);
            ushort scale = bf(max(div32(maximum, 127.0f), 1e-35f));
            float inverse = div32(1.0f, fp(scale));
            if (!lane)
            {
                activation_scales[c * blocks + activation_block] = scale;
            }
            for (uint j = 0; j < 8; ++j)
            {
                quantized[c * p.input_width + activation_block * 256 + lane * 8 + j] =
                    char(clamp(int(rint(flush32(values[j] * inverse))), -128, 127));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint row = lane % R, k = lane / R, out = (group % rows * S + simd) * R + row;
    float3 result = 0;
    for (uint block = 0; block < blocks; ++block)
    {
        int3 dots = 0;
        for (uint i = k; i < 64; i += 32 / R)
        {
            int4 weight = out < p.output_width
                              ? int4(w[((out / R * blocks + block) * 64 + i) * R + row])
                              : int4(0);
            for (uint c = 0; c < C && column + c < p.column_count; ++c)
            {
                int4 a = int4(*reinterpret_cast<threadgroup const char4*>(
                    quantized + c * p.input_width + block * 256 + i * 4));
                int4 product = weight * a;
                dots[c] += product.x + product.y + product.z + product.w;
            }
        }
        for (uint delta = 16; delta >= R; delta /= 2)
        {
            dots += simd_shuffle_down(dots, delta);
        }
        if (out < p.output_width)
        {
            for (uint c = 0; c < C && column + c < p.column_count; ++c)
            {
                result[c] = fma(float(dots[c]),
                                scales[block * p.output_width + out] *
                                    fp(activation_scales[c * blocks + block]),
                                result[c]);
            }
        }
    }
    if (!k && out < p.output_width)
    {
        for (uint c = 0; c < C && column + c < p.column_count; ++c)
        {
            uint at = (column + c) * p.output_width + out;
            ushort rounded = bf(result[c] + bias[out]);
            y[at] = p.add_residual ? bf(fp(rounded) + fp(residual[at])) : rounded;
        }
    }
}

#define PREPARED_KERNEL(NAME, R, C, S)                                                             \
    kernel void NAME(                                                                              \
        device const char4* w [[buffer(prepared_projection_binding::weights)]],                    \
        device const float* scales [[buffer(prepared_projection_binding::scales)]],                \
        device const float* bias [[buffer(prepared_projection_binding::bias)]],                    \
        device const ushort* x [[buffer(prepared_projection_binding::input)]],                     \
        device ushort* y [[buffer(prepared_projection_binding::output)]],                          \
        device const ushort* gain [[buffer(prepared_projection_binding::gain)]],                   \
        device const ushort* norm_bias [[buffer(prepared_projection_binding::norm_bias)]],         \
        device const ushort* residual [[buffer(prepared_projection_binding::residual)]],           \
        device atomic_uint* error [[buffer(prepared_projection_binding::error)]],                  \
        constant PreparedProjectionParams& p [[buffer(prepared_projection_binding::parameters)]],  \
        uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])     \
    {                                                                                              \
        threadgroup char q[C * 2048];                                                              \
        threadgroup ushort scales_a[C * 8], normalized[512];                                       \
        threadgroup float sums[8], squares[8], affine[2];                                          \
        prepared_projection<R, C, S>(w, scales, bias, x, y, gain, norm_bias, residual, error, p,   \
                                     group, tid, q, scales_a, normalized, sums, squares, affine);  \
    }
PREPARED_KERNEL(prepared2, 2, 1, 8)
PREPARED_KERNEL(prepared8, 8, 1, 8)

#undef PREPARED_KERNEL
