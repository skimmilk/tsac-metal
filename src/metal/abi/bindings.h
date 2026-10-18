#pragma once

#include "types.h"

namespace tsac_metal_abi
{

namespace add_binding
{
enum : u32
{
    parameters = parameter_slot,
    input = 1,
    right = 2,
    output = 3,
};
} // namespace add_binding

namespace copy_binding
{
enum : u32
{
    parameters = parameter_slot,
    input = 1,
    output = 2,
};
} // namespace copy_binding

namespace embedding_binding
{
enum : u32
{
    parameters = parameter_slot,
    embeddings = 1,
    tokens = 2,
    output = 3,
};
} // namespace embedding_binding

namespace embedding_quantized_binding
{
enum : u32
{
    parameters = parameter_slot,
    embeddings = 1,
    tokens = 2,
    output = 3,
    scales = 4,
    error = 5,
};
} // namespace embedding_quantized_binding

namespace quantization_checked_binding
{
enum : u32
{
    parameters = parameter_slot,
    input = 1,
    output = 2,
    scales = 3,
    error = 4,
};
} // namespace quantization_checked_binding

namespace projection_binding
{
enum : u32
{
    parameters = parameter_slot,
    weights = 1,
    scales = 2,
    bias = 3,
    input = 4,
    activation_scales = 5,
    output = 6,
    residual = 7,
};
} // namespace projection_binding

namespace prepared_projection_binding
{
enum : u32
{
    parameters = parameter_slot,
    weights = 1,
    scales = 2,
    bias = 3,
    input = 4,
    output = 5,
    gain = 6,
    norm_bias = 7,
    residual = 8,
    error = 9,
};
} // namespace prepared_projection_binding

namespace normalization_binding
{
enum : u32
{
    parameters = parameter_slot,
    input = 1,
    gain = 2,
    bias = 3,
    output = 4,
    error = 5,
};
} // namespace normalization_binding

namespace normalization_quantized_binding
{
enum : u32
{
    parameters = parameter_slot,
    input = 1,
    gain = 2,
    bias = 3,
    output = 4,
    scales = 5,
    error = 6,
};
} // namespace normalization_quantized_binding

namespace cache_binding
{
enum : u32
{
    parameters = parameter_slot,
    qkv = 1,
    rotation = 2,
    query = 3,
    keys = 4,
    values = 5,
    error = 6,
};
} // namespace cache_binding

namespace attention_scores_binding
{
enum : u32
{
    parameters = parameter_slot,
    query = 1,
    keys = 2,
    scores = 3,
};
} // namespace attention_scores_binding

namespace attention_weights_binding
{
enum : u32
{
    parameters = parameter_slot,
    scores = 1,
    metadata = 2,
};
} // namespace attention_weights_binding

namespace attention_output_binding
{
enum : u32
{
    parameters = parameter_slot,
    values = 1,
    scores = 2,
    metadata = 3,
    output = 4,
};
} // namespace attention_output_binding

namespace attention_short_binding
{
enum : u32
{
    parameters = parameter_slot,
    query = 1,
    keys = 2,
    values = 3,
    output = 4,
};
} // namespace attention_short_binding

namespace attention_codebook_binding
{
enum : u32
{
    parameters = parameter_slot,
    qkv = 1,
    keys = 2,
    values = 3,
    output = 4,
    error = 5,
};
} // namespace attention_codebook_binding

namespace attention_blocks_binding
{
enum : u32
{
    parameters = parameter_slot,
    query = 1,
    keys = 2,
    values = 3,
    partials = 4,
    summaries = 5,
};
} // namespace attention_blocks_binding

namespace attention_metadata_binding
{
enum : u32
{
    parameters = parameter_slot,
    summaries = 1,
    metadata = 2,
};
} // namespace attention_metadata_binding

namespace attention_combine_binding
{
enum : u32
{
    parameters = parameter_slot,
    partials = 1,
    metadata = 2,
    output = 3,
};
} // namespace attention_combine_binding

namespace softmax_binding
{
enum : u32
{
    parameters = parameter_slot,
    input = 1,
    output = 2,
    error = 3,
};
} // namespace softmax_binding

namespace softmax_decode_binding
{
enum : u32
{
    parameters = parameter_slot,
    input = 1,
    tree = 2,
    output = 3,
    error = 4,
};
} // namespace softmax_decode_binding

namespace range_binding
{
enum : u32
{
    parameters = parameter_slot,
    probabilities = 1,
    tree = 2,
    payload = 3,
    state = 4,
    tokens = 5,
    error = 6,
};
} // namespace range_binding

namespace convolution_binding
{
enum : u32
{
    parameters = parameter_slot,
    input = 1,
    weights = 2,
    bias = 3,
    output = 4,
};
} // namespace convolution_binding

namespace snake_binding
{
enum : u32
{
    parameters = parameter_slot,
    input = 1,
    alpha = 2,
    output = 3,
    error = 4,
};
} // namespace snake_binding

namespace rvq_search_binding
{
enum : u32
{
    parameters = parameter_slot,
    input = 1,
    codebook = 2,
    output = 3,
};
} // namespace rvq_search_binding

namespace residual_binding
{
enum : u32
{
    parameters = parameter_slot,
    left = 1,
    right = 2,
    output = 3,
    error = 4,
};
} // namespace residual_binding

namespace rvq_project_binding
{
enum : u32
{
    parameters = parameter_slot,
    tokens = 1,
    codebook = 2,
    weights = 3,
    bias = 4,
    output = 5,
    error = 6,
};
} // namespace rvq_project_binding

namespace rvq_accumulate_binding
{
enum : u32
{
    parameters = parameter_slot,
    tokens = 1,
    codebook = 2,
    weights = 3,
    bias = 4,
    residual = 5,
    output = 6,
    error = 7,
};
} // namespace rvq_accumulate_binding

namespace check_finite_binding
{
enum : u32
{
    parameters = parameter_slot,
    input = 1,
    error = 2,
};
} // namespace check_finite_binding

} // namespace tsac_metal_abi
