#include "apple-arm/kernels.hpp"
#include "probability_numeric.hpp"

namespace tsac::apple_arm
{
namespace
{
using namespace apple_arm;

// Lanes represent independent outputs/positions. Keep coordinate and position
// FMAs in ascending order, and retain every captured BF16 rounding boundary.
struct Attention
{
    std::span<const std::uint16_t> query, keys, values;
    std::size_t heads, head_width, width, time;
    float scale;
    bool round_values, tiled_keys;

    void scores16(std::span<float> scores, std::size_t head, std::size_t start) const
    {
        auto d0 = vdupq_n_f32(0), d1 = d0, d2 = d0, d3 = d0;
        const auto* key = keys.data() + start * width + head * head_width * 4;
        const auto* q = query.data() + head * head_width;
        for (std::size_t j = 0; j < head_width; ++j)
        {
            const auto x = probability_float(q[j]);
            d0 = vfmaq_n_f32(d0, expand4(key + j * 4), x);
            d1 = vfmaq_n_f32(d1, expand4(key + width * 4 + j * 4), x);
            d2 = vfmaq_n_f32(d2, expand4(key + width * 8 + j * 4), x);
            d3 = vfmaq_n_f32(d3, expand4(key + width * 12 + j * 4), x);
        }
        vst1q_f32(scores.data(), vmulq_n_f32(d0, scale));
        vst1q_f32(scores.data() + 4, vmulq_n_f32(d1, scale));
        vst1q_f32(scores.data() + 8, vmulq_n_f32(d2, scale));
        vst1q_f32(scores.data() + 12, vmulq_n_f32(d3, scale));
    }

    float32x4_t tiled_dot4(const std::uint16_t* q, const std::uint16_t* key) const
    {
        auto dot = vdupq_n_f32(0);
        std::size_t j = 0;
        for (; j + 4 <= head_width; j += 4)
        {
            const auto x = expand4(q + j);
            dot = vfmaq_laneq_f32(dot, expand4(key + j * 4), x, 0);
            dot = vfmaq_laneq_f32(dot, expand4(key + j * 4 + 4), x, 1);
            dot = vfmaq_laneq_f32(dot, expand4(key + j * 4 + 8), x, 2);
            dot = vfmaq_laneq_f32(dot, expand4(key + j * 4 + 12), x, 3);
        }
        for (; j < head_width; ++j)
        {
            dot = vfmaq_n_f32(dot, expand4(key + j * 4), probability_float(q[j]));
        }
        return dot;
    }

    float32x4_t row_dot4(const std::uint16_t* q, const std::uint16_t* key) const
    {
        auto dot = vdupq_n_f32(0);
        std::size_t j = 0;
        for (; j + 4 <= head_width; j += 4)
        {
            const auto a = expand4(key + j), b = expand4(key + width + j);
            const auto c = expand4(key + 2 * width + j), d = expand4(key + 3 * width + j);
            const auto ab0 = vtrn1q_f32(a, b), ab1 = vtrn2q_f32(a, b);
            const auto cd0 = vtrn1q_f32(c, d), cd1 = vtrn2q_f32(c, d);
            const auto x = expand4(q + j);
            dot = vfmaq_laneq_f32(dot, vcombine_f32(vget_low_f32(ab0), vget_low_f32(cd0)), x, 0);
            dot = vfmaq_laneq_f32(dot, vcombine_f32(vget_low_f32(ab1), vget_low_f32(cd1)), x, 1);
            dot = vfmaq_laneq_f32(dot, vcombine_f32(vget_high_f32(ab0), vget_high_f32(cd0)), x, 2);
            dot = vfmaq_laneq_f32(dot, vcombine_f32(vget_high_f32(ab1), vget_high_f32(cd1)), x, 3);
        }
        for (; j < head_width; ++j)
        {
            const float32x4_t k{probability_float(key[j]), probability_float(key[width + j]),
                                probability_float(key[2 * width + j]),
                                probability_float(key[3 * width + j])};
            dot = vfmaq_n_f32(dot, k, probability_float(q[j]));
        }
        return dot;
    }

    float score_block(std::span<float> scores, std::size_t head, std::size_t start) const
    {
        std::size_t t = 0;
        if (tiled_keys)
        {
            for (; t + 16 <= scores.size(); t += 16)
            {
                scores16(scores.subspan(t, 16), head, start + t);
            }
        }
        const auto* q = query.data() + head * head_width;
        for (; t + 4 <= scores.size(); t += 4)
        {
            const auto* key =
                keys.data() + (start + t) * width + head * head_width * (tiled_keys ? 4 : 1);
            const auto dot = tiled_keys ? tiled_dot4(q, key) : row_dot4(q, key);
            vst1q_f32(scores.data() + t, vmulq_n_f32(dot, scale));
        }
        // The final 1–3 history positions are row-major in both cache layouts.
        for (; t < scores.size(); ++t)
        {
            float dot = 0;
            for (std::size_t j = 0; j < head_width; ++j)
            {
                dot = std::fma(probability_float(q[j]),
                               probability_float(keys[(start + t) * width + head * head_width + j]),
                               dot);
            }
            scores[t] = dot * scale;
        }
        float maximum = -std::numeric_limits<float>::infinity();
        for (const auto score : scores)
        {
            maximum = std::max(maximum, score);
        }
        return maximum;
    }

    void merge4(std::span<float> numerator, std::size_t j, float32x4_t sum, float old_scale,
                float new_scale) const
    {
        if (round_values)
        {
            sum = vreinterpretq_f32_u32(vshll_n_u16(round_bf16(sum), 16));
        }
        const auto old = vmulq_n_f32(vld1q_f32(numerator.data() + j), old_scale);
        vst1q_f32(numerator.data() + j, vfmaq_n_f32(old, sum, new_scale));
    }

    void values16(std::span<const float> scores, std::span<float> numerator, std::size_t head,
                  std::size_t start, std::size_t j, float old_scale, float new_scale) const
    {
        auto s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
        for (std::size_t position = 0; position < scores.size(); ++position)
        {
            const auto* value = values.data() + (start + position) * width + head * head_width + j;
            const auto score = scores[position];
            s0 = vfmaq_n_f32(s0, expand4(value), score);
            s1 = vfmaq_n_f32(s1, expand4(value + 4), score);
            s2 = vfmaq_n_f32(s2, expand4(value + 8), score);
            s3 = vfmaq_n_f32(s3, expand4(value + 12), score);
        }
        merge4(numerator, j, s0, old_scale, new_scale);
        merge4(numerator, j + 4, s1, old_scale, new_scale);
        merge4(numerator, j + 8, s2, old_scale, new_scale);
        merge4(numerator, j + 12, s3, old_scale, new_scale);
    }

    void value_block(std::span<const float> scores, std::span<float> numerator, std::size_t head,
                     std::size_t start, float old_scale, float new_scale) const
    {
        std::size_t j = 0;
        for (; j + 16 <= head_width; j += 16)
        {
            values16(scores, numerator, head, start, j, old_scale, new_scale);
        }
        for (; j + 4 <= head_width; j += 4)
        {
            auto sum = vdupq_n_f32(0);
            for (std::size_t position = 0; position < scores.size(); ++position)
            {
                sum = vfmaq_n_f32(
                    sum,
                    expand4(values.data() + (start + position) * width + head * head_width + j),
                    scores[position]);
            }
            merge4(numerator, j, sum, old_scale, new_scale);
        }
        for (; j < head_width; ++j)
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
            volatile float old_numerator = numerator[j] * old_scale;
            numerator[j] = std::fma(sum, new_scale, static_cast<float>(old_numerator));
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
    const apple_arm::NearestRounding rounding;
    const auto width = checked_mul(heads, head_width);
    detail::require(width && query.size() == width && keys.size() == values.size() &&
                        !keys.empty() && keys.size() % width == 0,
                    "attention shape mismatch");
    const auto time = keys.size() / width;
    Attention{query,     keys, values, heads, head_width, width, time, scale, time > 64 || prefill,
              tiled_keys}
        .run(workspace);
}
} // namespace tsac::apple_arm
