#include "../abi/bindings.h"
#include "numeric.h"
#include "parameters.h"

using namespace metal;
using namespace tsac_metal_abi;
using namespace tsac_probability;

kernel void attention_scores(device const ushort* query [[buffer(attention_scores_binding::query)]],
                             device const ushort* keys [[buffer(attention_scores_binding::keys)]],
                             device float* scores [[buffer(attention_scores_binding::scores)]],
                             constant AttentionParams& p
                             [[buffer(attention_scores_binding::parameters)]],
                             uint id [[thread_position_in_grid]])
{
    if (id >= p.head_count * p.key_count)
    {
        return;
    }
    uint head = id / p.key_count, t = id % p.key_count;
    float sum = 0;
    uint stride = p.head_count * p.head_width;
    for (uint i = 0; i < p.head_width; ++i)
    {
        sum = fma(fp(query[head * p.head_width + i]),
                  fp(keys[t * stride + head * p.head_width + i]), sum);
    }
    scores[id] = sum * p.score_scale;
}

kernel void attention_weights(device float* scores [[buffer(attention_weights_binding::scores)]],
                              device float* metadata
                              [[buffer(attention_weights_binding::metadata)]],
                              constant AttentionParams& p
                              [[buffer(attention_weights_binding::parameters)]],
                              uint head [[thread_position_in_grid]])
{
    if (head >= p.head_count)
    {
        return;
    }
    uint blocks = (p.key_count + 63) / 64;
    float maximum_running = -INFINITY, total = 0;
    for (uint block = 0; block < blocks; ++block)
    {
        uint start = block * 64, count = min(64u, p.key_count - start);
        float maximum = -INFINITY;
        for (uint i = 0; i < count; ++i)
        {
            maximum = max(maximum, scores[head * p.key_count + start + i]);
        }
        float vals[64];
        for (uint i = 0; i < 64; ++i)
        {
            float z = i < count ? scores[head * p.key_count + start + i] - maximum : 0;
            vals[i] = i < count ? exp_original(z) : 0;
            if (i < count)
            {
                scores[head * p.key_count + start + i] = vals[i];
            }
        }
        float difference = maximum - maximum_running;
        float factor = exp_original(-abs(difference));
        float old_scale = maximum_running > maximum ? 1.0f : factor,
              new_scale = maximum_running > maximum ? factor : 1.0f;
        float old_total = total * old_scale;
        total = fma(sum64(vals, 64), new_scale, old_total);
        metadata[(head * blocks + block) * 2] = old_scale;
        metadata[(head * blocks + block) * 2 + 1] = new_scale;
        maximum_running = max(maximum_running, maximum);
    }
    metadata[p.head_count * blocks * 2 + head] = div32(1.0f, total);
}

kernel void
attention_output(device const ushort* values [[buffer(attention_output_binding::values)]],
                 device const float* scores [[buffer(attention_output_binding::scores)]],
                 device const float* metadata [[buffer(attention_output_binding::metadata)]],
                 device ushort* output [[buffer(attention_output_binding::output)]],
                 constant AttentionParams& p [[buffer(attention_output_binding::parameters)]],
                 uint id [[thread_position_in_grid]])
{
    if (id >= p.head_count * p.head_width)
    {
        return;
    }
    uint head = id / p.head_width, blocks = (p.key_count + 63) / 64;
    float numerator = 0;
    for (uint block = 0; block < blocks; ++block)
    {
        uint start = block * 64, count = min(64u, p.key_count - start);
        float sum = 0;
        for (uint i = 0; i < count; ++i)
        {
            sum = fma(fp(values[(start + i) * p.head_count * p.head_width + id]),
                      scores[head * p.key_count + start + i], sum);
        }
        if (p.round_block_values)
        {
            sum = fp(bf(sum));
        }
        float old = numerator * metadata[(head * blocks + block) * 2];
        numerator = fma(sum, metadata[(head * blocks + block) * 2 + 1], old);
    }
    output[id] = bf(numerator * metadata[p.head_count * blocks * 2 + head]);
}

// Codebook histories fit in one SIMD group. One group computes the scores
// and the original adjacent-pair sum tree; the whole threadgroup then consumes
// the weights. Keep the value FMAs serial and the prefill rounding boundary.
kernel void attention_short(device const ushort* query [[buffer(attention_short_binding::query)]],
                            device const ushort* keys [[buffer(attention_short_binding::keys)]],
                            device const ushort* values [[buffer(attention_short_binding::values)]],
                            device ushort* output [[buffer(attention_short_binding::output)]],
                            constant AttentionParams& p
                            [[buffer(attention_short_binding::parameters)]],
                            uint head [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_threadgroup]])
{
    uint column = head / p.head_count;
    head %= p.head_count;
    query += column * p.head_count * p.head_width;
    output += column * p.head_count * p.head_width;
    keys += column * p.key_column_stride_elements;
    values += column * p.value_column_stride_elements;
    threadgroup float weights[32], reciprocal;
    uint width = p.head_count * p.head_width;
    if (lane < 32)
    {
        float score = -INFINITY;
        if (lane < p.key_count)
        {
            float sum = 0;
            for (uint i = 0; i < p.head_width; ++i)
            {
                sum = fma(fp(query[head * p.head_width + i]),
                          fp(keys[lane * width + head * p.head_width + i]), sum);
            }
            score = sum * p.score_scale;
        }
        float maximum = simd_max(score);
        float weight = lane < p.key_count ? exp_original(score - maximum) : 0.0f;
        weights[lane] = weight;
        // Shuffle down by ascending powers of two: lane zero retains the
        // adjacent-pair tree, unlike an unspecified simd_sum(float) reduction.
        for (uint stride = 1; stride < 32; stride *= 2)
        {
            weight = add32(weight, simd_shuffle_down(weight, stride));
        }
        if (lane == 0)
        {
            float sum = add32(weight, 0.0f); // the unused half of the 64-value tree
            reciprocal = div32(1.0f, fma(sum, 1.0f, 0.0f));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < p.head_width; i += 128)
    {
        float sum = 0;
        for (uint t = 0; t < p.key_count; ++t)
        {
            sum = fma(fp(values[t * width + head * p.head_width + i]), weights[t], sum);
        }
        if (p.round_block_values)
        {
            sum = fp(bf(sum));
        }
        // The first streaming block has new_scale=1 and old numerator=+0.
        output[head * p.head_width + i] = bf(fma(sum, 1.0f, 0.0f) * reciprocal);
    }
}

kernel void attention_codebook(device const ushort* qkv [[buffer(attention_codebook_binding::qkv)]],
                               device ushort* keys [[buffer(attention_codebook_binding::keys)]],
                               device ushort* values [[buffer(attention_codebook_binding::values)]],
                               device ushort* output [[buffer(attention_codebook_binding::output)]],
                               device atomic_uint* error
                               [[buffer(attention_codebook_binding::error)]],
                               constant AttentionParams& p
                               [[buffer(attention_codebook_binding::parameters)]],
                               uint group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_threadgroup]])
{
    uint column = group / p.head_count, head = group % p.head_count;
    qkv += column * 1536;
    keys += column * p.key_column_stride_elements;
    values += column * p.value_column_stride_elements;
    output += column * 512;
    device const ushort* query = qkv;
    uint coordinate = head * 128 + lane, at = (p.key_count - 1) * 512 + coordinate;
    keys[at] = qkv[512 + coordinate];
    values[at] = qkv[1024 + coordinate];
    bool invalid = (qkv[coordinate] & 0x7f80u) == 0x7f80u ||
                   (qkv[512 + coordinate] & 0x7f80u) == 0x7f80u ||
                   (qkv[1024 + coordinate] & 0x7f80u) == 0x7f80u;
    if (simd_any(invalid) && lane % 32 == 0)
    {
        atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    }
    // Each group reads only the head it just wrote; no cross-group dependency.
    threadgroup_barrier(mem_flags::mem_device);
    threadgroup float weights[32], reciprocal;
    uint width = p.head_count * p.head_width;
    if (lane < 32)
    {
        float score = -INFINITY;
        if (lane < p.key_count)
        {
            float sum = 0;
            for (uint i = 0; i < p.head_width; ++i)
            {
                sum = fma(fp(query[head * p.head_width + i]),
                          fp(keys[lane * width + head * p.head_width + i]), sum);
            }
            score = sum * p.score_scale;
        }
        float maximum = simd_max(score);
        float weight = lane < p.key_count ? exp_original(score - maximum) : 0.0f;
        weights[lane] = weight;
        // Shuffle down by ascending powers of two: lane zero retains the
        // adjacent-pair tree, unlike an unspecified simd_sum(float) reduction.
        for (uint stride = 1; stride < 32; stride *= 2)
        {
            weight = add32(weight, simd_shuffle_down(weight, stride));
        }
        if (lane == 0)
        {
            float sum = add32(weight, 0.0f); // the unused half of the 64-value tree
            reciprocal = div32(1.0f, fma(sum, 1.0f, 0.0f));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < p.head_width; i += 128)
    {
        float sum = 0;
        for (uint t = 0; t < p.key_count; ++t)
        {
            sum = fma(fp(values[t * width + head * p.head_width + i]), weights[t], sum);
        }
        if (p.round_block_values)
        {
            sum = fp(bf(sum));
        }
        // The first streaming block has new_scale=1 and old numerator=+0.
        output[head * p.head_width + i] = bf(fma(sum, 1.0f, 0.0f) * reciprocal);
    }
}

// A group owns one head and 64 history positions. Scores and exponentials
// stay in threadgroup memory through the value product, eliminating the global
// score round-trip and the four-thread serial normalization of a whole history.
// Each dot/value sum still uses its original serial FMA order. The sum tree is
// the same adjacent-pair tree as sum64, including zero-padded final blocks.
kernel void
attention_blocks(device const ushort* query [[buffer(attention_blocks_binding::query)]],
                 device const ushort* keys [[buffer(attention_blocks_binding::keys)]],
                 device const ushort* values [[buffer(attention_blocks_binding::values)]],
                 device float* partials [[buffer(attention_blocks_binding::partials)]],
                 device float* summaries [[buffer(attention_blocks_binding::summaries)]],
                 constant AttentionParams& p [[buffer(attention_blocks_binding::parameters)]],
                 uint group [[threadgroup_position_in_grid]],
                 uint lane [[thread_index_in_threadgroup]])
{
    uint blocks = (p.key_count + 63) / 64, column = group / (p.head_count * blocks);
    group %= p.head_count * blocks;
    query += column * p.head_count * p.head_width;
    keys += column * p.key_column_stride_elements;
    values += column * p.value_column_stride_elements;
    partials += column * blocks * p.head_count * p.head_width;
    summaries += column * p.head_count * blocks * 2;
    uint head = group / blocks, block = group % blocks;
    uint start = block * 64, count = min(64u, p.key_count - start),
         width = p.head_count * p.head_width;
    threadgroup float scores[64], tree[4], maximum;
    float score = -INFINITY;
    if (lane < count)
    {
        float sum = 0;
        for (uint i = 0; i < p.head_width; ++i)
        {
            sum = fma(fp(query[head * p.head_width + i]),
                      fp(keys[(start + lane) * width + head * p.head_width + i]), sum);
        }
        score = sum * p.score_scale;
    }
    float local_max = simd_max(score);
    if (lane % 32 == 0)
    {
        tree[lane / 32] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0)
    {
        maximum = max(tree[0], tree[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float weight = lane < count ? exp_original(score - maximum) : 0.0f;
    if (lane < 64)
    {
        scores[lane] = weight;
    }
    for (uint stride = 1; stride < 32; stride *= 2)
    {
        weight = add32(weight, simd_shuffle_down(weight, stride));
    }
    if (lane % 32 == 0)
    {
        tree[lane / 32] = weight;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0)
    {
        summaries[group * 2] = maximum;
        summaries[group * 2 + 1] = add32(tree[0], tree[1]);
    }
    for (uint i = lane; i < p.head_width; i += 128)
    {
        float sum = 0;
        for (uint t = 0; t < count; ++t)
        {
            sum = fma(fp(values[(start + t) * width + head * p.head_width + i]), scores[t], sum);
        }
        // Materialize the original BF16 block boundary before another kernel
        // rescales and combines these values in ascending block order.
        partials[block * width + head * p.head_width + i] = fp(bf(sum));
    }
}

// Only the small streaming recurrence remains serial per head. Do not replace
// this with a global softmax: rescaling/rounding at each block is codec-visible.
// Store (old_scale, new_scale) per head/block and a final reciprocal total;
// attention_combine applies exactly that recurrence to each output coordinate.
kernel void attention_block_metadata(device const float* summaries
                                     [[buffer(attention_metadata_binding::summaries)]],
                                     device float* metadata
                                     [[buffer(attention_metadata_binding::metadata)]],
                                     constant AttentionParams& p
                                     [[buffer(attention_metadata_binding::parameters)]],
                                     uint head [[thread_position_in_grid]])
{
    if (head >= p.head_count * p.column_count)
    {
        return;
    }
    uint column = head / p.head_count;
    head %= p.head_count;
    summaries += column * p.head_count * ((p.key_count + 63) / 64) * 2;
    metadata += column * (p.head_count * ((p.key_count + 63) / 64) * 2 + p.head_count);
    uint blocks = (p.key_count + 63) / 64;
    float maximum_running = -INFINITY, total = 0;
    for (uint block = 0; block < blocks; ++block)
    {
        uint at = (head * blocks + block) * 2;
        float maximum = summaries[at];
        float difference = maximum - maximum_running, factor = exp_original(-abs(difference));
        float old_scale = maximum_running > maximum ? 1.0f : factor,
              new_scale = maximum_running > maximum ? factor : 1.0f;
        float old_total = total * old_scale;
        total = fma(summaries[at + 1], new_scale, old_total);
        metadata[at] = old_scale;
        metadata[at + 1] = new_scale;
        maximum_running = max(maximum_running, maximum);
    }
    metadata[p.head_count * blocks * 2 + head] = div32(1.0f, total);
}

kernel void
attention_combine(device const float* partials [[buffer(attention_combine_binding::partials)]],
                  device const float* metadata [[buffer(attention_combine_binding::metadata)]],
                  device ushort* output [[buffer(attention_combine_binding::output)]],
                  constant AttentionParams& p [[buffer(attention_combine_binding::parameters)]],
                  uint id [[thread_position_in_grid]])
{
    uint width = p.head_count * p.head_width, blocks = (p.key_count + 63) / 64;
    if (id >= width * p.column_count)
    {
        return;
    }
    uint column = id / width;
    id %= width;
    partials += column * blocks * width;
    metadata += column * (p.head_count * blocks * 2 + p.head_count);
    output += column * width;
    uint head = id / p.head_width;
    float numerator = 0;
    for (uint block = 0; block < blocks; ++block)
    {
        float old = numerator * metadata[(head * blocks + block) * 2];
        numerator =
            fma(partials[block * width + id], metadata[(head * blocks + block) * 2 + 1], old);
    }
    output[id] = bf(numerator * metadata[p.head_count * blocks * 2 + head]);
}
