#pragma once

namespace tsac
{
struct DacConfig
{
    unsigned channels;
    unsigned codebooks;
};

struct TransformerConfig
{
    unsigned codebooks;
    unsigned temporal_layers = 12, codebook_layers = 3;
    unsigned width = 512, heads = 4, head_width = 128, context = 1024;
};

} // namespace tsac
