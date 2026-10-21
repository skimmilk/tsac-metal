#include "weights/normalization.hpp"

#include "common/numeric.hpp"

namespace tsac
{
std::vector<float> fold_weight_norm(std::span<const float> v, std::span<const float> g,
                                    std::size_t fan_in, const Limits& limits)
{
    detail::require(fan_in > 0 && !g.empty(), "empty weight normalization shape");
    const auto count = checked_mul(fan_in, g.size());
    detail::require(v.size() == count, "weight normalization shape mismatch");
    detail::require(count <= limits.tensor_elements, "normalized weight exceeds element limit");
    detail::require_finite(v, "nonfinite weight normalization value");
    detail::require_finite(g, "nonfinite weight normalization value");

    std::vector<float> weights(count);
    for (std::size_t column_index = 0; column_index < g.size(); ++column_index)
    {
        const auto column = v.subspan(column_index * fan_in, fan_in);
        float squares = 0;
        for (float value : column)
        {
            squares += value * value;
        }
        detail::require(squares > 0 && std::isfinite(squares), "invalid weight normalization norm");
        const float scale = g[column_index] / std::sqrt(squares);
        detail::require(std::isfinite(scale), "nonfinite weight normalization scale");
        for (std::size_t i = 0; i < fan_in; ++i)
        {
            weights[column_index * fan_in + i] = column[i] * scale;
        }
    }
    detail::require_finite(weights, "nonfinite weight normalization value");
    return weights;
}

std::vector<float> fold_dac_weight_norm(std::span<const float> v, std::span<const float> g,
                                        std::size_t fan_in, const Limits& limits)
{
    detail::require(fan_in > 0 && !g.empty(), "empty weight normalization shape");
    detail::require(v.size() == checked_mul(fan_in, g.size()),
                    "weight normalization shape mismatch");
    detail::require(v.size() <= limits.tensor_elements, "normalized weight exceeds element limit");
    std::vector<float> weights(v.size());
    fold_dac_weight_norm_into(v, g, fan_in, weights, limits);
    return weights;
}

void fold_dac_weight_norm_into(std::span<const float> v, std::span<const float> g,
                               std::size_t fan_in, std::span<float> weights, const Limits& limits)
{
    detail::require(fan_in > 0 && !g.empty(), "empty weight normalization shape");
    const auto count = checked_mul(fan_in, g.size());
    detail::require(v.size() == count && weights.size() == count,
                    "weight normalization shape mismatch");
    detail::require(count <= limits.tensor_elements, "normalized weight exceeds element limit");
    detail::require_finite(v, "nonfinite weight normalization value");
    detail::require_finite(g, "nonfinite weight normalization value");
    const float inverse_width = 1.0f / std::sqrt(static_cast<float>(fan_in));
    const float reciprocal_width = 1.0f / static_cast<float>(fan_in);
    std::vector<float> squares(fan_in);
    for (std::size_t column_index = 0; column_index < g.size(); ++column_index)
    {
        const auto column = v.subspan(column_index * fan_in, fan_in);
        for (std::size_t i = 0; i < fan_in; ++i)
        {
            squares[i] = column[i] * column[i];
        }
        // Balanced adjacent pairs, equivalent to padding to the next power of
        // two with zeros. Carrying the last nonnegative term saves padding.
        for (std::size_t width = fan_in; width > 1; width = (width + 1) / 2)
        {
            for (std::size_t i = 0; i < width / 2; ++i)
            {
                squares[i] = squares[2 * i] + squares[2 * i + 1];
            }
            if (width % 2)
            {
                squares[width / 2] = squares[width - 1];
            }
        }
        const float mean = squares[0] * reciprocal_width;
        detail::require(mean > 0 && std::isfinite(mean), "invalid weight normalization norm");
        const float inverse_rms = 1.0f / std::sqrt(mean);
        for (std::size_t i = 0; i < fan_in; ++i)
        {
            const float normalized = column[i] * inverse_rms;
            const float unit = normalized * inverse_width;
            weights[column_index * fan_in + i] = unit * g[column_index];
        }
    }
    detail::require_finite(weights, "nonfinite weight normalization value");
}
} // namespace tsac
