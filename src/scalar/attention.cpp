#include "scalar/kernels.hpp"
#include "scalar/probability_numeric.hpp"

namespace tsac::scalar
{
namespace
{
using namespace scalar;

// Lanes represent independent outputs/positions. Keep coordinate and position
// FMAs in ascending order, and retain every captured BF16 rounding boundary.
struct Attention
{
    std::span<const std::uint16_t> query, keys, values;
    std::size_t heads, head_width, width, time;
    float scale;
    bool round_values, tiled_keys;

    float score_block(std::span<float> scores, std::size_t head, std::size_t start) const
    {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t t = 0; t < scores.size(); ++t)
        {
            const auto position = start + t;
            float dot = 0;
            for (std::size_t j = 0; j < head_width; ++j)
            {
                const auto index =
                    tiled_keys && position < time / 4 * 4
                        ? (position / 4 * 4) * width + head * head_width * 4 + j * 4 + position % 4
                        : position * width + head * head_width + j;
                dot = std::fma(probability_float(query[head * head_width + j]),
                               probability_float(keys[index]), dot);
            }
            scores[t] = dot * scale;
            maximum = std::max(maximum, scores[t]);
        }
        return maximum;
    }

    void value_block(std::span<const float> scores, std::span<float> numerator, std::size_t head,
                     std::size_t start, float old_scale, float new_scale) const
    {
        for (std::size_t j = 0; j < head_width; ++j)
        {
            float sum = 0;
            for (std::size_t t = 0; t < scores.size(); ++t)
            {
                sum =
                    std::fma(probability_float(values[(start + t) * width + head * head_width + j]),
                             scores[t], sum);
            }
            if (round_values)
            {
                sum = probability_float(probability_bf16(sum));
            }
            volatile float old = numerator[j] * old_scale;
            numerator[j] = std::fma(sum, new_scale, static_cast<float>(old));
        }
    }

    void run(AttentionWorkspace& workspace) const
    {
        auto& result = workspace.output;
        auto& numerator = workspace.numerator;
        result.resize(width);
        numerator.resize(head_width);
        std::array<float, 64> score_storage;
        for (std::size_t head = 0; head < heads; ++head)
        {
            float running_maximum = -std::numeric_limits<float>::infinity(), total = 0;
            std::fill(numerator.begin(), numerator.end(), 0.0f);
            // Merge each 64-position softmax block in order, including the BF16
            // value-sum boundary for long histories and rollover prefill.
            for (std::size_t start = 0; start < time; start += 64)
            {
                auto scores =
                    std::span(score_storage).first(std::min<std::size_t>(64, time - start));
                const float maximum = score_block(scores, head, start);
                for (auto& score : scores)
                {
                    volatile float shifted = score - maximum;
                    score = original_exp(shifted);
                }
                volatile float difference = maximum - running_maximum;
                const float factor = original_exp(-std::abs(static_cast<float>(difference)));
                const float old_scale = running_maximum > maximum ? 1.0f : factor;
                const float new_scale = running_maximum > maximum ? factor : 1.0f;
                volatile float old_total = total * old_scale;
                total = std::fma(probability_sum(scores), new_scale, static_cast<float>(old_total));
                value_block(scores, numerator, head, start, old_scale, new_scale);
                running_maximum = std::max(running_maximum, maximum);
            }
            volatile float reciprocal = 1.0f / total;
            for (std::size_t j = 0; j < head_width; ++j)
            {
                volatile float output = numerator[j] * reciprocal;
                result[head * head_width + j] = probability_bf16(output);
            }
        }
    }
};
} // namespace

std::vector<std::uint16_t> attention_bf16(std::span<const std::uint16_t> query,
                                          std::span<const std::uint16_t> keys,
                                          std::span<const std::uint16_t> values, std::size_t heads,
                                          std::size_t head_width, float scale, bool prefill,
                                          bool tiled_keys)
{
    AttentionWorkspace workspace;
    attention_bf16_into(query, keys, values, workspace, heads, head_width, scale, prefill,
                        tiled_keys);
    return std::move(workspace.output);
}

void attention_bf16_into(std::span<const std::uint16_t> query, std::span<const std::uint16_t> keys,
                         std::span<const std::uint16_t> values, AttentionWorkspace& workspace,
                         std::size_t heads, std::size_t head_width, float scale, bool prefill,
                         bool tiled_keys)
{
    const scalar::NearestRounding rounding;
    const auto width = checked_mul(heads, head_width);
    detail::require(width && query.size() == width && keys.size() == values.size() &&
                        !keys.empty() && keys.size() % width == 0,
                    "attention shape mismatch");
    const auto time = keys.size() / width;
    Attention{query,     keys, values, heads, head_width, width, time, scale, time > 64 || prefill,
              tiled_keys}
        .run(workspace);
}
} // namespace tsac::scalar
