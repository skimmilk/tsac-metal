#pragma once

#include "../abi/types.h"

// Wire fields are 32-bit scalars. Boolean options are encoded as 0 or 1.
namespace tsac_metal_abi
{

struct ConvolutionParams
{
    u32 input_channels;
    u32 output_channels;
    u32 kernel_size;
    u32 stride;
    u32 padding;
    u32 dilation;
    u32 transposed;
    u32 input_frames;
    u32 output_frames;
    u32 group_count;
};

struct SnakeParams
{
    u32 element_count;
    u32 channel_count;
};

struct RvqSearchParams
{
    u32 vector_count;
    u32 normalize_input;
};

struct ResidualParams
{
    u32 element_count;
    u32 subtract;
};

struct RvqProjectParams
{
    u32 token_count;
};

struct RvqAccumulateParams
{
    u32 element_count;
    u32 subtract;
};

struct CheckFiniteParams
{
    u32 element_count;
};

static_assert(sizeof(ConvolutionParams) == 40, "Metal ABI layout mismatch");
static_assert(alignof(ConvolutionParams) == 4, "Metal ABI layout mismatch");
static_assert(sizeof(SnakeParams) == 8, "Metal ABI layout mismatch");
static_assert(alignof(SnakeParams) == 4, "Metal ABI layout mismatch");
static_assert(sizeof(RvqSearchParams) == 8, "Metal ABI layout mismatch");
static_assert(alignof(RvqSearchParams) == 4, "Metal ABI layout mismatch");
static_assert(sizeof(ResidualParams) == 8, "Metal ABI layout mismatch");
static_assert(alignof(ResidualParams) == 4, "Metal ABI layout mismatch");
static_assert(sizeof(RvqProjectParams) == 4, "Metal ABI layout mismatch");
static_assert(alignof(RvqProjectParams) == 4, "Metal ABI layout mismatch");
static_assert(sizeof(RvqAccumulateParams) == 8, "Metal ABI layout mismatch");
static_assert(alignof(RvqAccumulateParams) == 4, "Metal ABI layout mismatch");
static_assert(sizeof(CheckFiniteParams) == 4, "Metal ABI layout mismatch");
static_assert(alignof(CheckFiniteParams) == 4, "Metal ABI layout mismatch");

#ifndef __METAL_VERSION__

static_assert(std::is_standard_layout_v<ConvolutionParams> &&
              std::is_trivially_copyable_v<ConvolutionParams>);
static_assert(offsetof(ConvolutionParams, input_channels) == 0);
static_assert(offsetof(ConvolutionParams, output_channels) == 4);
static_assert(offsetof(ConvolutionParams, kernel_size) == 8);
static_assert(offsetof(ConvolutionParams, stride) == 12);
static_assert(offsetof(ConvolutionParams, padding) == 16);
static_assert(offsetof(ConvolutionParams, dilation) == 20);
static_assert(offsetof(ConvolutionParams, transposed) == 24);
static_assert(offsetof(ConvolutionParams, input_frames) == 28);
static_assert(offsetof(ConvolutionParams, output_frames) == 32);
static_assert(offsetof(ConvolutionParams, group_count) == 36);
template <> inline constexpr bool is_parameter_block<ConvolutionParams> = true;

static_assert(std::is_standard_layout_v<SnakeParams> && std::is_trivially_copyable_v<SnakeParams>);
static_assert(offsetof(SnakeParams, element_count) == 0);
static_assert(offsetof(SnakeParams, channel_count) == 4);
template <> inline constexpr bool is_parameter_block<SnakeParams> = true;

static_assert(std::is_standard_layout_v<RvqSearchParams> &&
              std::is_trivially_copyable_v<RvqSearchParams>);
static_assert(offsetof(RvqSearchParams, vector_count) == 0);
static_assert(offsetof(RvqSearchParams, normalize_input) == 4);
template <> inline constexpr bool is_parameter_block<RvqSearchParams> = true;

static_assert(std::is_standard_layout_v<ResidualParams> &&
              std::is_trivially_copyable_v<ResidualParams>);
static_assert(offsetof(ResidualParams, element_count) == 0);
static_assert(offsetof(ResidualParams, subtract) == 4);
template <> inline constexpr bool is_parameter_block<ResidualParams> = true;

static_assert(std::is_standard_layout_v<RvqProjectParams> &&
              std::is_trivially_copyable_v<RvqProjectParams>);
static_assert(offsetof(RvqProjectParams, token_count) == 0);
template <> inline constexpr bool is_parameter_block<RvqProjectParams> = true;

static_assert(std::is_standard_layout_v<RvqAccumulateParams> &&
              std::is_trivially_copyable_v<RvqAccumulateParams>);
static_assert(offsetof(RvqAccumulateParams, element_count) == 0);
static_assert(offsetof(RvqAccumulateParams, subtract) == 4);
template <> inline constexpr bool is_parameter_block<RvqAccumulateParams> = true;

static_assert(std::is_standard_layout_v<CheckFiniteParams> &&
              std::is_trivially_copyable_v<CheckFiniteParams>);
static_assert(offsetof(CheckFiniteParams, element_count) == 0);
template <> inline constexpr bool is_parameter_block<CheckFiniteParams> = true;

#endif
} // namespace tsac_metal_abi
