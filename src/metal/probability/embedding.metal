#include "../abi/bindings.h"
#include "numeric.h"
#include "parameters.h"
#include "quantization.h"

using namespace metal;
using namespace tsac_metal_abi;
using namespace tsac_probability;

kernel void embed_tokens(device const ushort* embeddings [[buffer(embedding_binding::embeddings)]],
                         device const ushort* tokens [[buffer(embedding_binding::tokens)]],
                         device ushort* output [[buffer(embedding_binding::output)]],
                         constant EmbeddingParams& p [[buffer(embedding_binding::parameters)]],
                         uint id [[thread_position_in_grid]])
{
    if (id >= p.column_count * 1024)
    {
        return;
    }
    uint channel = id % 1024;
    tokens += (id / 1024) * p.prefix_books;
    ushort sum = embeddings[uint(tokens[0]) * 1024 + channel];
    for (uint book = 1; book < p.prefix_books; ++book)
    {
        sum = bf(fp(sum) + fp(embeddings[(book * 1024 + uint(tokens[book])) * 1024 + channel]));
    }
    output[id] = sum;
}

// Tokens are addressed in the stream's frame/group/book layout. UINT_MAX
// denotes the initial all-zero feedback, including codebook entry zero.
kernel void embed_resident(device const ushort* embeddings
                           [[buffer(embedding_binding::embeddings)]],
                           device const ushort* tokens [[buffer(embedding_binding::tokens)]],
                           device ushort* output [[buffer(embedding_binding::output)]],
                           constant ResidentEmbeddingParams& p
                           [[buffer(embedding_binding::parameters)]],
                           uint id [[thread_position_in_grid]])
{
    if (id >= p.column_count * 1024)
    {
        return;
    }
    uint channel = id % 1024, sequence = p.first_sequence + id / 1024,
         lane = sequence / p.group_count, group = sequence % p.group_count;
    uint frame = lane * (p.frame_count / p.lane_count) + min(lane, p.frame_count % p.lane_count) +
                 p.step_index;
    uint at = (frame * p.group_count + group) * p.book_count;
    ushort token = p.initial_feedback ? 0 : tokens[at];
    ushort sum = embeddings[uint(token) * 1024 + channel];
    for (uint book = 1; book < p.prefix_books; ++book)
    {
        token = p.initial_feedback ? 0 : tokens[at + book];
        sum = bf(fp(sum) + fp(embeddings[(book * 1024 + uint(token)) * 1024 + channel]));
    }
    output[id] = sum;
}

template <class Q>
inline void embed_quant_resident_impl(device const ushort* embeddings, device const ushort* tokens,
                                      device Q* quantized, device ushort* scales,
                                      device atomic_uint* error,
                                      constant ResidentEmbeddingParams& p, uint block, uint tid,
                                      threadgroup QuantizationShared& shared)
{
    uint column = block / 4, channel = (block % 4) * 256 + tid,
         sequence = p.first_sequence + column;
    uint lane = sequence / p.group_count, group = sequence % p.group_count;
    uint frame = lane * (p.frame_count / p.lane_count) + min(lane, p.frame_count % p.lane_count) +
                 p.step_index;
    uint at = (frame * p.group_count + group) * p.book_count;
    ushort token = p.initial_feedback ? 0 : tokens[at];
    ushort value = embeddings[uint(token) * 1024 + channel];
    for (uint book = 1; book < p.prefix_books; ++book)
    {
        token = p.initial_feedback ? 0 : tokens[at + book];
        value = bf(fp(value) + fp(embeddings[(book * 1024 + uint(token)) * 1024 + channel]));
    }
    bool invalid = (value & 0x7f80u) == 0x7f80u;
    if (simd_any(invalid) && tid % 32 == 0)
    {
        atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
    }
    float v = invalid ? 0.0f : flush32(fp(value)), maximum = simd_max(abs(v));
    if (tid % 32 == 0)
    {
        shared.maxima[tid / 32] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32)
    {
        maximum = simd_max(tid < 8 ? shared.maxima[tid] : 0.0f);
        if (!tid)
        {
            ushort scale = bf(max(div32(maximum, 127.0f), 1e-35f));
            scales[block] = scale;
            shared.inverse = div32(1.0f, fp(scale));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    quantized[column * 1024 + channel] =
        char(clamp(int(rint(flush32(v * shared.inverse))), -128, 127));
}

kernel void embed_quant_resident(
    device const ushort* embeddings [[buffer(embedding_quantized_binding::embeddings)]],
    device const ushort* tokens [[buffer(embedding_quantized_binding::tokens)]],
    device char* quantized [[buffer(embedding_quantized_binding::output)]],
    device ushort* scales [[buffer(embedding_quantized_binding::scales)]],
    device atomic_uint* error [[buffer(embedding_quantized_binding::error)]],
    constant ResidentEmbeddingParams& p [[buffer(embedding_quantized_binding::parameters)]],
    uint block [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup QuantizationShared shared;
    embed_quant_resident_impl(embeddings, tokens, quantized, scales, error, p, block, tid, shared);
}

kernel void embed_quant_resident_wide(
    device const ushort* embeddings [[buffer(embedding_quantized_binding::embeddings)]],
    device const ushort* tokens [[buffer(embedding_quantized_binding::tokens)]],
    device int* quantized [[buffer(embedding_quantized_binding::output)]],
    device ushort* scales [[buffer(embedding_quantized_binding::scales)]],
    device atomic_uint* error [[buffer(embedding_quantized_binding::error)]],
    constant ResidentEmbeddingParams& p [[buffer(embedding_quantized_binding::parameters)]],
    uint block [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup QuantizationShared shared;
    embed_quant_resident_impl(embeddings, tokens, quantized, scales, error, p, block, tid, shared);
}
