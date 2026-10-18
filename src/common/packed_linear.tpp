#include "common/internal.hpp"
#include "common/packed_linear.hpp"
#include "common/probability.hpp"

#include <cfenv>
#include <cmath>

namespace tsac
{
namespace
{
class LinearRounding
{
public:
    LinearRounding() : previous_(std::fegetround())
    {
        if (previous_ != FE_TONEAREST)
        {
            std::fesetround(FE_TONEAREST);
        }
    }

    ~LinearRounding()
    {
        if (previous_ != FE_TONEAREST)
        {
            std::fesetround(previous_);
        }
    }

private:
    int previous_;
};

} // namespace

template <class Kernels>
PackedLinear<Kernels> PackedLinear<Kernels>::from_weights(const LinearWeights& ref,
                                                          const Limits& limits)
{
    PackedLinear result;
    const auto input = ref.values.shape().axes[0], output = ref.values.shape().axes[1];
    result.input_ = input;
    result.output_ = output;
    result.limits_ = limits;
    detail::require(input && input % 256 == 0 && output % 4 == 0 &&
                        checked_mul(input, output) <= limits.tensor_elements,
                    "invalid generated linear shape");
    result.scales_ = weights::Array<float>(ref.scales);
    result.bias_ = weights::Array<float>(ref.bias);
    detail::require(result.scales_.size() == input / 256 * output && result.bias_.size() == output,
                    "invalid generated linear constants");
    {
        const auto values = ref.values.values();
        detail::require(values.size() == input * output, "invalid generated linear weights");
        result.weights_.resize(values.size());
        for (std::size_t out = 0; out < output; out += 4)
        {
            for (std::size_t b = 0; b < input / 256; ++b)
            {
                for (std::size_t q = 0; q < 64; ++q)
                {
                    for (std::size_t row = 0; row < 4; ++row)
                    {
                        std::copy_n(values.data() + (out + row) * input + b * 256 + q * 4, 4,
                                    result.weights_.data() + out * input + b * 1024 + q * 16 +
                                        row * 4);
                    }
                }
            }
        }
    }
    return result;
}

template <class Kernels>
std::vector<std::uint16_t> PackedLinear<Kernels>::apply(std::span<const std::uint16_t> input) const
{
    Workspace workspace;
    std::vector<std::uint16_t> result;
    apply_into(input, result, workspace);
    return result;
}

template <class Kernels>
void PackedLinear<Kernels>::apply_into(std::span<const std::uint16_t> input,
                                       std::vector<std::uint16_t>& result,
                                       Workspace& workspace) const
{
    const LinearRounding rounding;
    detail::require(input_ && input.size() % input_ == 0 && input.size() <= limits_.tensor_elements,
                    "packed linear input shape mismatch");
    const auto columns = input.size() / input_;
    const auto count = checked_mul(columns, output_);
    detail::require(count <= limits_.tensor_elements, "packed linear output exceeds element limit");
    auto& expanded = workspace.expanded;
    expanded.resize(input.size());
    // Expand BF16 activations before block quantization.
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        expanded[i] = probability_float(input[i]);
    }
    auto& quantized = workspace.quantized;
    Kernels::bf8l_into(expanded, quantized);
    const auto blocks = input_ / 256;
    // Activation scales are shared by every output row. Expand them once,
    // keeping conversion calls (and their register spills) out of the dot loop.
    auto& activation_scales = workspace.scales;
    activation_scales.resize(quantized.scales.size());
    for (std::size_t i = 0; i < activation_scales.size(); ++i)
    {
        activation_scales[i] = probability_float(quantized.scales[i]);
    }
    // Expansion finishes before resizing output, so input/output may alias.
    result.resize(count);
    for (std::size_t column = 0; column < columns; ++column)
    {
        Kernels::linear_column(weights_, scales_, bias_, input_, output_,
                               std::span(quantized.values).subspan(column * input_, input_),
                               std::span(activation_scales).subspan(column * blocks, blocks),
                               std::span(result).subspan(column * output_, output_));
    }
}

} // namespace tsac
