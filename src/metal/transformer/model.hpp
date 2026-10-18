#pragma once

#include "metal/linear.hpp"
#include "metal/runtime.hpp"
#include "models/transformer_model.hpp"

namespace tsac::metal::transformer
{
struct Norm
{
    Buffer gain, bias;
};

struct Layer
{
    Norm first, second;
    MetalLinear qkv, attention_out, mlp_in, mlp_out;
};

struct Stack
{
    std::vector<Layer> layers;
    Norm final;
};

// Contains only GPU execution layouts. Serialized models and preparation
// temporaries are released by the caller after construction.
struct PreparedModel
{
    unsigned channels, maximum_books;
    Limits limits;
    Stack temporal, codebook;
    MetalLinear temporal_projection, codebook_projection, output_projection;
    Buffer constants, embeddings, positions, rotations;
    std::shared_ptr<const ResidencySet> residency;

    static std::shared_ptr<const PreparedModel> load(TransformerModel transformer,
                                                     const Limits& limits);
};
} // namespace tsac::metal::transformer
