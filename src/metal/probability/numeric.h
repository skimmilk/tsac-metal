#pragma once

#include <metal_stdlib>

// Preserve explicit FMA and BF16 operation boundaries in every probability source.
#pragma clang fp contract(off)

namespace tsac_probability
{

using namespace metal;

inline float fp(ushort x)
{
    return as_type<float>(uint(x) << 16);
}

inline ushort bf(float x)
{
    uint u = as_type<uint>(x);
    return ushort((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

inline float flush32(float x)
{
    uint b = as_type<uint>(x);
    return (b & 0x7f800000u) == 0 ? as_type<float>(b & 0x80000000u) : x;
}

inline float add32(float a, float b)
{
    return flush32(flush32(a) + flush32(b));
}

inline float div32(float a, float b)
{
    return precise::divide(a, b);
}

inline float exp_original(float z)
{
    z = 87.0f < z ? 87.0f : z;
    z = -88.0f > z ? -88.0f : z;
    float exponent = z * 0x1.715476p+0f;
    int n = isfinite(exponent) ? int(rint(exponent)) : int(0x80000000u);
    z = fma(float(n), -0x1.630000p-1f, z);
    z = fma(float(n), 0x1.bd0106p-13f, z);
    float p = fma(0x1.a01a02p-13f, z, 0x1.6c16c2p-10f);
    p = fma(p, z, 0x1.111112p-7f);
    p = fma(p, z, 0x1.555556p-5f);
    p = fma(p, z, 0x1.555556p-3f);
    float square = z * z;
    p = fma(p, z, 0.5f);
    z = fma(p, square, z);
    float plus = z + 1.0f;
    return as_type<float>(as_type<uint>(plus) + (uint(n) << 23));
}

inline float sum64(thread float* values, uint width)
{
    while (width > 1)
    {
        width /= 2;
        for (uint i = 0; i < width; ++i)
        {
            values[i] = add32(values[2 * i], values[2 * i + 1]);
        }
    }
    return values[0];
}

inline ushort gelu_value(ushort bits)
{
    // Preserve gradual float32 underflow at the BF16 half-multiplication.
    uint magnitude = bits & 0x7fffu;
    if (magnitude < 0x100u)
    {
        return ushort((bits & 0x8000u) | ((magnitude >> 1) + ((magnitude & 3u) == 3u)));
    }
    if (magnitude >= 0x7f80u)
    {
        return magnitude == 0x7f80u ? (bits == 0x7f80u ? bits : ushort(0x7fc0u))
                                    : ushort(bits | 0x40u);
    }
    float v = fp(bits);
    float cubic = 0x1.2444f2p-5f * v;
    float z = fma(cubic, v, 0x1.988454p-1f);
    z = z * v;
    float half_product = v * 0.5f;
    z = z * -2.0f;
    float e = exp_original(z);
    float denominator = e + 1.0f;
    float sigmoid = div32(2.0f, denominator);
    float t = sigmoid - 1.0f;
    float plus = t + 1.0f;
    return bf(half_product * plus);
}

} // namespace tsac_probability
