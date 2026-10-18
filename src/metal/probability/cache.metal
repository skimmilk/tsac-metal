#include "../abi/bindings.h"
#include "numeric.h"
#include "parameters.h"

using namespace metal;
using namespace tsac_metal_abi;
using namespace tsac_probability;

// Split QKV, rotate all heads and append only the new key/value cache row.
kernel void cache_qkv(device const ushort* x [[buffer(cache_binding::qkv)]],
                      device const ushort* rotation [[buffer(cache_binding::rotation)]],
                      device ushort* query [[buffer(cache_binding::query)]],
                      device ushort* keys [[buffer(cache_binding::keys)]],
                      device ushort* values [[buffer(cache_binding::values)]],
                      device atomic_uint* error [[buffer(cache_binding::error)]],
                      constant CacheAppendParams& p [[buffer(cache_binding::parameters)]],
                      uint id [[thread_position_in_grid]])
{
    uint column = id / 256;
    id %= 256;
    if (column >= p.column_count)
    {
        return;
    }
    x += column * 1536;
    query += column * 512;
    keys += column * p.key_column_stride_elements;
    values += column * p.value_column_stride_elements;
    uint i = id * 2, at = p.write_position * 512 + i;
    values[at] = x[1024 + i];
    values[at + 1] = x[1024 + i + 1];
    bool invalid = false;
    for (uint j = 0; j < 3; ++j)
    {
        invalid |=
            (x[j * 512 + i] & 0x7f80u) == 0x7f80u || (x[j * 512 + i + 1] & 0x7f80u) == 0x7f80u;
    }
    if (p.apply_rotary)
    {
        float cosine = fp(rotation[i % 128]), sine = fp(rotation[i % 128 + 1]);
        float real = fp(x[i]), imag = fp(x[i + 1]);
        float a = imag * sine, b = real * sine;
        ushort qr = bf(fma(real, cosine, -a)), qi = bf(fma(imag, cosine, b));
        query[i] = qr;
        query[i + 1] = qi;
        invalid |= (qr & 0x7f80u) == 0x7f80u || (qi & 0x7f80u) == 0x7f80u;
        real = fp(x[512 + i]);
        imag = fp(x[512 + i + 1]);
        a = imag * sine;
        b = real * sine;
        ushort kr = bf(fma(real, cosine, -a)), ki = bf(fma(imag, cosine, b));
        keys[at] = kr;
        keys[at + 1] = ki;
        invalid |= (kr & 0x7f80u) == 0x7f80u || (ki & 0x7f80u) == 0x7f80u;
    }
    else
    {
        query[i] = x[i];
        query[i + 1] = x[i + 1];
        keys[at] = x[512 + i];
        keys[at + 1] = x[512 + i + 1];
    }
    if (simd_any(invalid) && id % 32 == 0)
    {
        atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    }
}
