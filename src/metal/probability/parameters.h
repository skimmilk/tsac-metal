#pragma once

#include "../abi/types.h"

// Wire fields are 32-bit scalars. Boolean options are encoded as 0 or 1.
namespace tsac_metal_abi
{

struct AddParams
{
    u32 element_count;
    u32 right_period_elements;
};

struct CopyParams
{
    u32 element_count;
};

struct EmbeddingParams
{
    u32 column_count;
    u32 prefix_books;
};

struct ResidentEmbeddingParams
{
    u32 frame_count;
    u32 lane_count;
    u32 group_count;
    u32 book_count;
    u32 step_index;
    u32 prefix_books;
    u32 first_sequence;
    u32 column_count;
    u32 initial_feedback;
};

struct QuantizationParams
{
    u32 input_width;
    u32 column_count;
    u32 apply_gelu;
};

struct ProjectionParams
{
    u32 input_width;
    u32 output_width;
    u32 column_count;
};

struct PreparedProjectionParams
{
    u32 input_width;
    u32 output_width;
    u32 column_count;
    u32 normalize_input;
    u32 apply_gelu;
    u32 add_residual;
    float epsilon;
};

struct NormalizationParams
{
    u32 column_count;
    float epsilon;
};

struct CacheAppendParams
{
    u32 column_count;
    u32 write_position;
    u32 key_column_stride_elements;
    u32 value_column_stride_elements;
    u32 apply_rotary;
};

struct AttentionParams
{
    u32 column_count;
    u32 head_count;
    u32 head_width;
    u32 key_count;
    u32 key_column_stride_elements;
    u32 value_column_stride_elements;
    float score_scale;
    u32 round_block_values;
};

struct SoftmaxParams
{
    u32 column_count;
};

struct RangeDecodeParams
{
    u32 frame_count;
    u32 lane_count;
    u32 group_count;
    u32 book_count;
    u32 step_index;
    u32 book_index;
    u32 column_count;
    u32 payload_bytes;
};

struct RangeState
{
    u32 range;
    u32 code;
    u32 payload_offset;
    u32 status;
};

static_assert(sizeof(AddParams) == 8, "Metal ABI layout mismatch");
static_assert(alignof(AddParams) == 4, "Metal ABI layout mismatch");
static_assert(sizeof(CopyParams) == 4, "Metal ABI layout mismatch");
static_assert(alignof(CopyParams) == 4, "Metal ABI layout mismatch");
static_assert(sizeof(EmbeddingParams) == 8, "Metal ABI layout mismatch");
static_assert(alignof(EmbeddingParams) == 4, "Metal ABI layout mismatch");
static_assert(sizeof(ResidentEmbeddingParams) == 36, "Metal ABI layout mismatch");
static_assert(alignof(ResidentEmbeddingParams) == 4, "Metal ABI layout mismatch");
static_assert(sizeof(QuantizationParams) == 12, "Metal ABI layout mismatch");
static_assert(alignof(QuantizationParams) == 4, "Metal ABI layout mismatch");
static_assert(sizeof(ProjectionParams) == 12, "Metal ABI layout mismatch");
static_assert(alignof(ProjectionParams) == 4, "Metal ABI layout mismatch");
static_assert(sizeof(PreparedProjectionParams) == 28, "Metal ABI layout mismatch");
static_assert(alignof(PreparedProjectionParams) == 4, "Metal ABI layout mismatch");
static_assert(sizeof(NormalizationParams) == 8, "Metal ABI layout mismatch");
static_assert(alignof(NormalizationParams) == 4, "Metal ABI layout mismatch");
static_assert(sizeof(CacheAppendParams) == 20, "Metal ABI layout mismatch");
static_assert(alignof(CacheAppendParams) == 4, "Metal ABI layout mismatch");
static_assert(sizeof(AttentionParams) == 32, "Metal ABI layout mismatch");
static_assert(alignof(AttentionParams) == 4, "Metal ABI layout mismatch");
static_assert(sizeof(SoftmaxParams) == 4, "Metal ABI layout mismatch");
static_assert(alignof(SoftmaxParams) == 4, "Metal ABI layout mismatch");
static_assert(sizeof(RangeDecodeParams) == 32, "Metal ABI layout mismatch");
static_assert(alignof(RangeDecodeParams) == 4, "Metal ABI layout mismatch");
static_assert(sizeof(RangeState) == 16, "Metal ABI layout mismatch");
static_assert(alignof(RangeState) == 4, "Metal ABI layout mismatch");

#ifndef __METAL_VERSION__

static_assert(std::is_standard_layout_v<AddParams> && std::is_trivially_copyable_v<AddParams>);
static_assert(offsetof(AddParams, element_count) == 0);
static_assert(offsetof(AddParams, right_period_elements) == 4);
template <> inline constexpr bool is_parameter_block<AddParams> = true;

static_assert(std::is_standard_layout_v<CopyParams> && std::is_trivially_copyable_v<CopyParams>);
static_assert(offsetof(CopyParams, element_count) == 0);
template <> inline constexpr bool is_parameter_block<CopyParams> = true;

static_assert(std::is_standard_layout_v<EmbeddingParams> &&
              std::is_trivially_copyable_v<EmbeddingParams>);
static_assert(offsetof(EmbeddingParams, column_count) == 0);
static_assert(offsetof(EmbeddingParams, prefix_books) == 4);
template <> inline constexpr bool is_parameter_block<EmbeddingParams> = true;

static_assert(std::is_standard_layout_v<ResidentEmbeddingParams> &&
              std::is_trivially_copyable_v<ResidentEmbeddingParams>);
static_assert(offsetof(ResidentEmbeddingParams, frame_count) == 0);
static_assert(offsetof(ResidentEmbeddingParams, lane_count) == 4);
static_assert(offsetof(ResidentEmbeddingParams, group_count) == 8);
static_assert(offsetof(ResidentEmbeddingParams, book_count) == 12);
static_assert(offsetof(ResidentEmbeddingParams, step_index) == 16);
static_assert(offsetof(ResidentEmbeddingParams, prefix_books) == 20);
static_assert(offsetof(ResidentEmbeddingParams, first_sequence) == 24);
static_assert(offsetof(ResidentEmbeddingParams, column_count) == 28);
static_assert(offsetof(ResidentEmbeddingParams, initial_feedback) == 32);
template <> inline constexpr bool is_parameter_block<ResidentEmbeddingParams> = true;

static_assert(std::is_standard_layout_v<QuantizationParams> &&
              std::is_trivially_copyable_v<QuantizationParams>);
static_assert(offsetof(QuantizationParams, input_width) == 0);
static_assert(offsetof(QuantizationParams, column_count) == 4);
static_assert(offsetof(QuantizationParams, apply_gelu) == 8);
template <> inline constexpr bool is_parameter_block<QuantizationParams> = true;

static_assert(std::is_standard_layout_v<ProjectionParams> &&
              std::is_trivially_copyable_v<ProjectionParams>);
static_assert(offsetof(ProjectionParams, input_width) == 0);
static_assert(offsetof(ProjectionParams, output_width) == 4);
static_assert(offsetof(ProjectionParams, column_count) == 8);
template <> inline constexpr bool is_parameter_block<ProjectionParams> = true;

static_assert(std::is_standard_layout_v<PreparedProjectionParams> &&
              std::is_trivially_copyable_v<PreparedProjectionParams>);
static_assert(offsetof(PreparedProjectionParams, input_width) == 0);
static_assert(offsetof(PreparedProjectionParams, output_width) == 4);
static_assert(offsetof(PreparedProjectionParams, column_count) == 8);
static_assert(offsetof(PreparedProjectionParams, normalize_input) == 12);
static_assert(offsetof(PreparedProjectionParams, apply_gelu) == 16);
static_assert(offsetof(PreparedProjectionParams, add_residual) == 20);
static_assert(offsetof(PreparedProjectionParams, epsilon) == 24);
template <> inline constexpr bool is_parameter_block<PreparedProjectionParams> = true;

static_assert(std::is_standard_layout_v<NormalizationParams> &&
              std::is_trivially_copyable_v<NormalizationParams>);
static_assert(offsetof(NormalizationParams, column_count) == 0);
static_assert(offsetof(NormalizationParams, epsilon) == 4);
template <> inline constexpr bool is_parameter_block<NormalizationParams> = true;

static_assert(std::is_standard_layout_v<CacheAppendParams> &&
              std::is_trivially_copyable_v<CacheAppendParams>);
static_assert(offsetof(CacheAppendParams, column_count) == 0);
static_assert(offsetof(CacheAppendParams, write_position) == 4);
static_assert(offsetof(CacheAppendParams, key_column_stride_elements) == 8);
static_assert(offsetof(CacheAppendParams, value_column_stride_elements) == 12);
static_assert(offsetof(CacheAppendParams, apply_rotary) == 16);
template <> inline constexpr bool is_parameter_block<CacheAppendParams> = true;

static_assert(std::is_standard_layout_v<AttentionParams> &&
              std::is_trivially_copyable_v<AttentionParams>);
static_assert(offsetof(AttentionParams, column_count) == 0);
static_assert(offsetof(AttentionParams, head_count) == 4);
static_assert(offsetof(AttentionParams, head_width) == 8);
static_assert(offsetof(AttentionParams, key_count) == 12);
static_assert(offsetof(AttentionParams, key_column_stride_elements) == 16);
static_assert(offsetof(AttentionParams, value_column_stride_elements) == 20);
static_assert(offsetof(AttentionParams, score_scale) == 24);
static_assert(offsetof(AttentionParams, round_block_values) == 28);
template <> inline constexpr bool is_parameter_block<AttentionParams> = true;

static_assert(std::is_standard_layout_v<SoftmaxParams> &&
              std::is_trivially_copyable_v<SoftmaxParams>);
static_assert(offsetof(SoftmaxParams, column_count) == 0);
template <> inline constexpr bool is_parameter_block<SoftmaxParams> = true;

static_assert(std::is_standard_layout_v<RangeDecodeParams> &&
              std::is_trivially_copyable_v<RangeDecodeParams>);
static_assert(offsetof(RangeDecodeParams, frame_count) == 0);
static_assert(offsetof(RangeDecodeParams, lane_count) == 4);
static_assert(offsetof(RangeDecodeParams, group_count) == 8);
static_assert(offsetof(RangeDecodeParams, book_count) == 12);
static_assert(offsetof(RangeDecodeParams, step_index) == 16);
static_assert(offsetof(RangeDecodeParams, book_index) == 20);
static_assert(offsetof(RangeDecodeParams, column_count) == 24);
static_assert(offsetof(RangeDecodeParams, payload_bytes) == 28);
template <> inline constexpr bool is_parameter_block<RangeDecodeParams> = true;

static_assert(std::is_standard_layout_v<RangeState> && std::is_trivially_copyable_v<RangeState>);
static_assert(offsetof(RangeState, range) == 0);
static_assert(offsetof(RangeState, code) == 4);
static_assert(offsetof(RangeState, payload_offset) == 8);
static_assert(offsetof(RangeState, status) == 12);

#endif
} // namespace tsac_metal_abi
