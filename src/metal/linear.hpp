#pragma once

#include "common/common.hpp"
#include "models/types.hpp"
#include "weights/tensor_weights.hpp"

#include <memory>

namespace tsac::metal
{
class Prediction;
class Buffer;
} // namespace tsac::metal

namespace tsac
{

// Integer weights, scales and bias remain in unified Metal buffers. Each apply
// quantizes the activation and performs every projection operation on the GPU.
class MetalLinear
{
public:
    static MetalLinear prepare(std::size_t input_width, std::size_t output_width,
                               std::span<const std::int8_t> weights,
                               std::span<const float> block_scales, std::span<const float> bias);
    static MetalLinear from_weights(const LinearWeights& linear);
    std::vector<std::uint16_t> apply(std::span<const std::uint16_t> values) const;
    // Owning handles for registering the immutable model residency set.
    std::vector<metal::Buffer> resources() const;

private:
    friend class metal::Prediction;
    struct Impl;
    explicit MetalLinear(std::shared_ptr<Impl> impl);
    std::shared_ptr<Impl> impl_;
};
} // namespace tsac
