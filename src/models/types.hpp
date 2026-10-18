#pragma once
#include "models/tensor_view.hpp"

namespace tsac
{
enum class ModelVariant
{
    mono,
    stereo
};
// Serialized RVQ projection axes, independent of the execution implementation.
enum class RvqInputLayout
{
    input_major,
    output_major
};
enum class ResidualConvolution
{
    dilated,
    pointwise
};
enum class ResidualActivation
{
    before_dilated,
    before_pointwise
};
enum class TransformerStack
{
    temporal,
    codebook
};

struct ConvolutionWeights
{
    EncodedTensorView values;
    TensorView<float> gain, bias;
};

struct FloatProjectionWeights
{
    TensorView<float> values, bias;
};

struct LinearWeights
{
    TensorView<std::int8_t> values;
    TensorView<float> scales, bias;
};

struct NormalizationWeights
{
    TensorView<std::uint16_t> gain, bias;
};
} // namespace tsac
