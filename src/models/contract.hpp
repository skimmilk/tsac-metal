#pragma once
#include "models/types.hpp"

namespace tsac::model_contract
{
inline constexpr std::size_t tensor_alignment = 256;
// Apple Silicon's 16 KiB page size also defines the on-disk region alignment.
inline constexpr std::size_t mapping_alignment = 16384;
inline constexpr unsigned stages = 4, residual_units = 3, dac_convolutions = 30;
inline constexpr unsigned temporal_layers = 12, codebook_layers = 3;
inline constexpr unsigned transformer_width = 512, latent_width = 1024;
inline constexpr unsigned codebook_width = 8, alphabet = 1024;
inline constexpr std::size_t embedding_book_bytes = std::size_t{2} * 1024 * 1024;
inline constexpr TensorShape rvq_vectors{{codebook_width, alphabet}};
inline constexpr TensorShape rvq_output_weights{{codebook_width, latent_width}};
inline constexpr TensorShape rvq_output_bias{{latent_width}};
inline constexpr TensorShape rvq_input_bias{{codebook_width}};
inline constexpr TensorShape normalization{{transformer_width}};
inline constexpr TensorShape rotary{{128, 1024}};
unsigned channels(ModelVariant variant);
unsigned codebooks(ModelVariant variant);
unsigned layers(TransformerStack stack);
void validate_rvq_input_layout(RvqInputLayout layout);
TensorShape dac_shape(bool synthesis, ModelVariant variant, unsigned convolution);
TensorEncoding dac_encoding(bool synthesis, unsigned convolution);
TensorShape activation_shape(bool synthesis, unsigned index);
TensorShape rvq_input_shape(RvqInputLayout layout);
TensorShape positions_shape(unsigned books);
TensorShape feedback_shape(unsigned books);
TensorShape projection_shape(unsigned projection);
std::size_t encoded_bytes(TensorShape shape, TensorEncoding encoding);
unsigned residual_convolution(bool synthesis, unsigned stage, unsigned unit,
                              ResidualConvolution which);
unsigned residual_activation(bool synthesis, unsigned stage, unsigned unit,
                             ResidualActivation which);
unsigned resample_convolution(bool synthesis, unsigned stage);
unsigned resample_activation(bool synthesis, unsigned stage);
unsigned projection_index(TransformerStack stack, unsigned layer, unsigned operation);
unsigned normalization_index(TransformerStack stack, unsigned layer, bool mlp);
unsigned final_normalization_index(TransformerStack stack);
} // namespace tsac::model_contract
