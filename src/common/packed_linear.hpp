#pragma once
#include "common/probability.hpp"
#include "models/types.hpp"
#include "weights/tensor_weights.hpp"

#include <memory>

namespace tsac
{

// Packed BF8L linear projection. Axis 0 is input width and axis 1 output width.
// Preserves integer 256-value dot blocks and serialized BF16 scale planes.
template <class Kernels> class PackedLinear
{
public:
    // Owned by the evaluating lane, never by immutable model weights.
    struct Workspace
    {
        std::vector<float> expanded, scales;
        ProbabilityBf8l quantized;
    };

    static PackedLinear from_weights(const LinearWeights& linear, const Limits& limits = {});

    std::size_t input_width() const
    {
        return input_;
    }

    std::size_t output_width() const
    {
        return output_;
    }

    // Consecutive input columns, each input_width() BF16 values. Returns BF16
    // output columns. Numerical compatibility is checked against original calls.
    std::vector<std::uint16_t> apply(std::span<const std::uint16_t> input) const;
    void apply_into(std::span<const std::uint16_t> input, std::vector<std::uint16_t>& output,
                    Workspace& workspace) const;

private:
    std::size_t input_ = 0, output_ = 0;
    Limits limits_;
    std::vector<std::int8_t> weights_;
    weights::Array<float> scales_, bias_;
};
} // namespace tsac
