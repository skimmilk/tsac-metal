#include "../abi/bindings.h"
#include "numeric.h"
#include "parameters.h"

using namespace metal;
using namespace tsac_metal_abi;
using namespace tsac_probability;

// One serialized range state serves every lane/group, in stream order. The
// balanced subtree masses are produced in parallel by softmax_decode. Right
// mass is still subtraction from the path mass, never a re-summed subtree.
kernel void range_decode(device const ushort* probabilities
                         [[buffer(range_binding::probabilities)]],
                         device const float* trees [[buffer(range_binding::tree)]],
                         device const uchar* payload [[buffer(range_binding::payload)]],
                         device RangeState& state [[buffer(range_binding::state)]],
                         device ushort* tokens [[buffer(range_binding::tokens)]],
                         device atomic_uint* error [[buffer(range_binding::error)]],
                         constant RangeDecodeParams& p [[buffer(range_binding::parameters)]],
                         uint id [[thread_position_in_grid]])
{
    if (id || state.status)
    {
        return;
    }
    uint range = state.range, code = state.code, offset = state.payload_offset;
    for (uint sequence = 0; sequence < p.column_count; ++sequence)
    {
        uint base = 0, node = 1, count = 1024;
        float mass = 1.0f;
        while (count > 1)
        {
            uint midpoint = count / 2;
            float left = midpoint == 1 ? fp(probabilities[sequence * 1024 + base])
                                       : trees[sequence * 1024 + node * 2];
            float ratio = flush32(div32(flush32(flush32(left) * 32768.0f), flush32(mass)));
            uint q = 1;
            if (isfinite(ratio) && ratio < 0x1p63f && ratio >= -0x1p63f)
            {
                q = uint(clamp(int(uint(long(rint(ratio)))), 1, 32767));
            }
            uint split = uint((ulong(range) * q) >> 15);
            bool bit = code >= split;
            if (bit)
            {
                code -= split;
                range -= split;
                base += midpoint;
                mass = add32(mass, -left);
            }
            else
            {
                range = split;
                mass = left;
            }
            while (range < 0xff01u)
            {
                range <<= 8;
                code = (code << 8) | (offset < p.payload_bytes ? uint(payload[offset++]) : 0u);
            }
            if (code >= range)
            {
                state.status = 1;
                atomic_fetch_or_explicit(error, 2u, memory_order_relaxed);
                return;
            }
            count = midpoint;
            node = node * 2 + uint(bit);
        }
        uint lane = sequence / p.group_count, group = sequence % p.group_count;
        uint frame = lane * (p.frame_count / p.lane_count) +
                     min(lane, p.frame_count % p.lane_count) + p.step_index;
        tokens[(frame * p.group_count + group) * p.book_count + p.book_index] = ushort(base);
    }
    state.range = range;
    state.code = code;
    state.payload_offset = offset;
}
