#pragma once

#include "../dac/parameters.h"
#include "../probability/parameters.h"
#include "bindings.h"

#include <cstddef>
#include <string_view>

namespace tsac_metal_abi
{
// Host-side validation of the wire interface before encoding any GPU work.
struct KernelContract
{
    std::string_view name;
    std::size_t parameter_bytes;
    u32 resource_mask;
};

inline constexpr KernelContract kernel_contracts[] = {
    {"add_kernel", sizeof(AddParams),
     (1u << add_binding::input) | (1u << add_binding::right) | (1u << add_binding::output)},
    {"attention_block_metadata", sizeof(AttentionParams),
     (1u << attention_metadata_binding::summaries) | (1u << attention_metadata_binding::metadata)},
    {"attention_blocks", sizeof(AttentionParams),
     (1u << attention_blocks_binding::query) | (1u << attention_blocks_binding::keys) |
         (1u << attention_blocks_binding::values) | (1u << attention_blocks_binding::partials) |
         (1u << attention_blocks_binding::summaries)},
    {"attention_codebook", sizeof(AttentionParams),
     (1u << attention_codebook_binding::qkv) | (1u << attention_codebook_binding::keys) |
         (1u << attention_codebook_binding::values) | (1u << attention_codebook_binding::output) |
         (1u << attention_codebook_binding::error)},
    {"attention_combine", sizeof(AttentionParams),
     (1u << attention_combine_binding::partials) | (1u << attention_combine_binding::metadata) |
         (1u << attention_combine_binding::output)},
    {"attention_output", sizeof(AttentionParams),
     (1u << attention_output_binding::values) | (1u << attention_output_binding::scores) |
         (1u << attention_output_binding::metadata) | (1u << attention_output_binding::output)},
    {"attention_scores", sizeof(AttentionParams),
     (1u << attention_scores_binding::query) | (1u << attention_scores_binding::keys) |
         (1u << attention_scores_binding::scores)},
    {"attention_short", sizeof(AttentionParams),
     (1u << attention_short_binding::query) | (1u << attention_short_binding::keys) |
         (1u << attention_short_binding::values) | (1u << attention_short_binding::output)},
    {"attention_weights", sizeof(AttentionParams),
     (1u << attention_weights_binding::scores) | (1u << attention_weights_binding::metadata)},
    {"cache_qkv", sizeof(CacheAppendParams),
     (1u << cache_binding::qkv) | (1u << cache_binding::rotation) | (1u << cache_binding::query) |
         (1u << cache_binding::keys) | (1u << cache_binding::values) |
         (1u << cache_binding::error)},
    {"check_finite", sizeof(CheckFiniteParams),
     (1u << check_finite_binding::input) | (1u << check_finite_binding::error)},
    {"conv_hot_192_1", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"conv_hot_192_3", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"conv_hot_192_9", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"conv_hot_384_1", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"conv_hot_384_3", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"conv_hot_384_9", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"conv_static_768_1", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"conv_static_768_3", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"conv_static_768_9", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"conv_static_96_1", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"conv_static_96_3", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"conv_static_96_9", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"conv_transpose_1536_768", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"conv_transpose_192_96", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"conv_transpose_384_192", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"conv_transpose_768_384", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"convolution", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"convolution4", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"convolution_tiled", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"convolution_tiled64", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"copy_bf16", sizeof(CopyParams), (1u << copy_binding::input) | (1u << copy_binding::output)},
    {"embed_quant_resident", sizeof(ResidentEmbeddingParams),
     (1u << embedding_quantized_binding::embeddings) | (1u << embedding_quantized_binding::tokens) |
         (1u << embedding_quantized_binding::output) | (1u << embedding_quantized_binding::scales) |
         (1u << embedding_quantized_binding::error)},
    {"embed_quant_resident_wide", sizeof(ResidentEmbeddingParams),
     (1u << embedding_quantized_binding::embeddings) | (1u << embedding_quantized_binding::tokens) |
         (1u << embedding_quantized_binding::output) | (1u << embedding_quantized_binding::scales) |
         (1u << embedding_quantized_binding::error)},
    {"embed_resident", sizeof(ResidentEmbeddingParams),
     (1u << embedding_binding::embeddings) | (1u << embedding_binding::tokens) |
         (1u << embedding_binding::output)},
    {"embed_tokens", sizeof(EmbeddingParams),
     (1u << embedding_binding::embeddings) | (1u << embedding_binding::tokens) |
         (1u << embedding_binding::output)},
    {"linear_kernel", sizeof(ProjectionParams),
     (1u << projection_binding::weights) | (1u << projection_binding::scales) |
         (1u << projection_binding::bias) | (1u << projection_binding::input) |
         (1u << projection_binding::activation_scales) | (1u << projection_binding::output)},
    {"linear_matrix", sizeof(ProjectionParams),
     (1u << projection_binding::weights) | (1u << projection_binding::scales) |
         (1u << projection_binding::bias) | (1u << projection_binding::input) |
         (1u << projection_binding::activation_scales) | (1u << projection_binding::output)},
    {"linear_matrix_add", sizeof(ProjectionParams),
     (1u << projection_binding::weights) | (1u << projection_binding::scales) |
         (1u << projection_binding::bias) | (1u << projection_binding::input) |
         (1u << projection_binding::activation_scales) | (1u << projection_binding::output) |
         (1u << projection_binding::residual)},
    {"linear_packed_add_wide2", sizeof(ProjectionParams),
     (1u << projection_binding::weights) | (1u << projection_binding::scales) |
         (1u << projection_binding::bias) | (1u << projection_binding::input) |
         (1u << projection_binding::activation_scales) | (1u << projection_binding::output) |
         (1u << projection_binding::residual)},
    {"linear_packed_add_wide8", sizeof(ProjectionParams),
     (1u << projection_binding::weights) | (1u << projection_binding::scales) |
         (1u << projection_binding::bias) | (1u << projection_binding::input) |
         (1u << projection_binding::activation_scales) | (1u << projection_binding::output) |
         (1u << projection_binding::residual)},
    {"linear_packed_wide2", sizeof(ProjectionParams),
     (1u << projection_binding::weights) | (1u << projection_binding::scales) |
         (1u << projection_binding::bias) | (1u << projection_binding::input) |
         (1u << projection_binding::activation_scales) | (1u << projection_binding::output)},
    {"linear_packed_wide8", sizeof(ProjectionParams),
     (1u << projection_binding::weights) | (1u << projection_binding::scales) |
         (1u << projection_binding::bias) | (1u << projection_binding::input) |
         (1u << projection_binding::activation_scales) | (1u << projection_binding::output)},
    {"linear_small2", sizeof(ProjectionParams),
     (1u << projection_binding::weights) | (1u << projection_binding::scales) |
         (1u << projection_binding::bias) | (1u << projection_binding::input) |
         (1u << projection_binding::activation_scales) | (1u << projection_binding::output)},
    {"linear_small4", sizeof(ProjectionParams),
     (1u << projection_binding::weights) | (1u << projection_binding::scales) |
         (1u << projection_binding::bias) | (1u << projection_binding::input) |
         (1u << projection_binding::activation_scales) | (1u << projection_binding::output)},
    {"matched", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"norm512_parallel", sizeof(NormalizationParams),
     (1u << normalization_binding::input) | (1u << normalization_binding::gain) |
         (1u << normalization_binding::bias) | (1u << normalization_binding::output) |
         (1u << normalization_binding::error)},
    {"norm_quant512", sizeof(NormalizationParams),
     (1u << normalization_quantized_binding::input) |
         (1u << normalization_quantized_binding::gain) |
         (1u << normalization_quantized_binding::bias) |
         (1u << normalization_quantized_binding::output) |
         (1u << normalization_quantized_binding::scales) |
         (1u << normalization_quantized_binding::error)},
    {"norm_quant512_wide", sizeof(NormalizationParams),
     (1u << normalization_quantized_binding::input) |
         (1u << normalization_quantized_binding::gain) |
         (1u << normalization_quantized_binding::bias) |
         (1u << normalization_quantized_binding::output) |
         (1u << normalization_quantized_binding::scales) |
         (1u << normalization_quantized_binding::error)},
    {"point64", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"point96", sizeof(ConvolutionParams),
     (1u << convolution_binding::input) | (1u << convolution_binding::weights) |
         (1u << convolution_binding::bias) | (1u << convolution_binding::output)},
    {"prepared2", sizeof(PreparedProjectionParams),
     (1u << prepared_projection_binding::weights) | (1u << prepared_projection_binding::scales) |
         (1u << prepared_projection_binding::bias) | (1u << prepared_projection_binding::input) |
         (1u << prepared_projection_binding::output) | (1u << prepared_projection_binding::gain) |
         (1u << prepared_projection_binding::norm_bias) |
         (1u << prepared_projection_binding::residual) |
         (1u << prepared_projection_binding::error)},
    {"prepared8", sizeof(PreparedProjectionParams),
     (1u << prepared_projection_binding::weights) | (1u << prepared_projection_binding::scales) |
         (1u << prepared_projection_binding::bias) | (1u << prepared_projection_binding::input) |
         (1u << prepared_projection_binding::output) | (1u << prepared_projection_binding::gain) |
         (1u << prepared_projection_binding::norm_bias) |
         (1u << prepared_projection_binding::residual) |
         (1u << prepared_projection_binding::error)},
    {"quant_bf16_parallel", sizeof(QuantizationParams),
     (1u << quantization_checked_binding::input) | (1u << quantization_checked_binding::output) |
         (1u << quantization_checked_binding::scales) |
         (1u << quantization_checked_binding::error)},
    {"quant_bf16_parallel_wide", sizeof(QuantizationParams),
     (1u << quantization_checked_binding::input) | (1u << quantization_checked_binding::output) |
         (1u << quantization_checked_binding::scales) |
         (1u << quantization_checked_binding::error)},
    {"range_decode", sizeof(RangeDecodeParams),
     (1u << range_binding::probabilities) | (1u << range_binding::tree) |
         (1u << range_binding::payload) | (1u << range_binding::state) |
         (1u << range_binding::tokens) | (1u << range_binding::error)},
    {"residual", sizeof(ResidualParams),
     (1u << residual_binding::left) | (1u << residual_binding::right) |
         (1u << residual_binding::output) | (1u << residual_binding::error)},
    {"rvq_accumulate", sizeof(RvqAccumulateParams),
     (1u << rvq_accumulate_binding::tokens) | (1u << rvq_accumulate_binding::codebook) |
         (1u << rvq_accumulate_binding::weights) | (1u << rvq_accumulate_binding::bias) |
         (1u << rvq_accumulate_binding::residual) | (1u << rvq_accumulate_binding::output) |
         (1u << rvq_accumulate_binding::error)},
    {"rvq_project", sizeof(RvqProjectParams),
     (1u << rvq_project_binding::tokens) | (1u << rvq_project_binding::codebook) |
         (1u << rvq_project_binding::weights) | (1u << rvq_project_binding::bias) |
         (1u << rvq_project_binding::output) | (1u << rvq_project_binding::error)},
    {"rvq_search", sizeof(RvqSearchParams),
     (1u << rvq_search_binding::input) | (1u << rvq_search_binding::codebook) |
         (1u << rvq_search_binding::output)},
    {"snake", sizeof(SnakeParams),
     (1u << snake_binding::input) | (1u << snake_binding::alpha) | (1u << snake_binding::output) |
         (1u << snake_binding::error)},
    {"softmax1024_decode", sizeof(SoftmaxParams),
     (1u << softmax_decode_binding::input) | (1u << softmax_decode_binding::tree) |
         (1u << softmax_decode_binding::output) | (1u << softmax_decode_binding::error)},
    {"softmax1024_parallel", sizeof(SoftmaxParams),
     (1u << softmax_binding::input) | (1u << softmax_binding::output) |
         (1u << softmax_binding::error)},
};

inline const KernelContract* find_kernel_contract(std::string_view name)
{
    for (const auto& contract : kernel_contracts)
    {
        if (contract.name == name)
        {
            return &contract;
        }
    }
    return nullptr;
}
} // namespace tsac_metal_abi
