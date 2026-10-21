#include "weights/preparation.hpp"

#include "common/numeric.hpp"

#include <array>
#include <cstring>

namespace tsac::weights
{
std::vector<float> expand(const EncodedTensorView& tensor)
{
    std::vector<float> values(tensor.shape().elements());
    expand_into(tensor, 0, values);
    return values;
}

void expand_into(const EncodedTensorView& tensor, std::size_t first, std::span<float> values)
{
    const auto payload = tensor.payload();
    const auto count = tensor.shape().elements();
    detail::require(first <= count && values.size() <= count - first,
                    "source weight range exceeds tensor");
    if (tensor.encoding() == TensorEncoding::float32)
    {
        detail::require(payload.size() == count * 4, "invalid FP32 source weights");
        if (!values.empty())
        {
            std::memcpy(values.data(), payload.data() + first * 4, values.size_bytes());
        }
    }
    else if (tensor.encoding() == TensorEncoding::bf16)
    {
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            values[i] = detail::bf16(payload, (first + i) * 2);
        }
    }
    else
    {
        detail::require(tensor.encoding() == TensorEncoding::bf8, "unsupported source weights");
        const auto rows = tensor.shape().axes[0], columns = tensor.shape().axes[1];
        const std::size_t block = 32;
        const auto padded = (columns + 7) / 8 * 8;
        const auto plane = tensor.data_plane_bytes();
        detail::require(rows && columns && rows % block == 0, "invalid quantized source shape");
        detail::require(first % rows == 0 && values.size() % rows == 0,
                        "unaligned quantized source range");
        for (std::size_t c = 0; c < values.size() / rows; ++c)
        {
            const auto column = first / rows + c;
            for (std::size_t b = 0; b < rows / block; ++b)
            {
                const auto scale = detail::bf16(
                    payload,
                    plane +
                        ((column / columns * (rows / block) + b) * padded + column % columns) * 2);
                const auto begin = column * rows + b * block;
                for (std::size_t i = 0; i < block; ++i)
                {
                    values[c * rows + b * block + i] =
                        static_cast<float>(static_cast<std::int8_t>(payload[begin + i])) * scale;
                }
            }
        }
    }
    detail::require_finite(values, "nonfinite source weights");
}

void normalize_search(std::span<const float, 8> input, std::span<float, 8> output)
{
    constexpr float epsilon = std::bit_cast<float>(std::uint32_t{0x2b8cbccc});
    std::array<float, 8> squares;
    for (std::size_t i = 0; i < 8; ++i)
    {
        squares[i] = input[i] * input[i];
    }
    for (std::size_t width = 8; width > 1; width /= 2)
    {
        for (std::size_t i = 0; i < width / 2; ++i)
        {
            squares[i] = squares[i * 2] + squares[i * 2 + 1];
        }
    }
    detail::require(std::isfinite(squares[0]), "invalid RVQ normalization norm");
    const float mean = squares[0] * (1.0f / 8.0f);
    const float inverse_rms = 1.0f / std::sqrt(mean + epsilon);
    const float inverse_width = 1.0f / std::sqrt(8.0f);
    for (std::size_t i = 0; i < 8; ++i)
    {
        const float normalized = input[i] * inverse_rms;
        output[i] = normalized * inverse_width;
    }
}

void project_codebook(std::uint16_t index, std::span<const float> codebook,
                      std::span<const float> weights, std::span<const float> bias,
                      std::span<float> output, bool fused)
{
    detail::require(index < 1024 && codebook.size() == 8192 && weights.size() == 8192 &&
                        bias.size() == 1024 && output.size() == 1024,
                    "invalid codebook projection");
    const auto offset = std::size_t(index) * 8;
    for (std::size_t j = 0; j < 1024; ++j)
    {
        float sum = 0;
        for (std::size_t i = 0; i < 8; ++i)
        {
            const float x = codebook[offset + i], w = weights[j * 8 + i];
            sum = fused ? std::fma(w, x, sum) : sum + w * x;
        }
        output[j] = sum + bias[j];
    }
    detail::require_finite(output, "nonfinite RVQ value");
}
} // namespace tsac::weights
