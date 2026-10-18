#include "apple-arm/dsp.hpp"

#include "apple-arm/execution.hpp"
#include "common/internal.hpp"
#include "common/snake.hpp"
#include "weights/normalization.hpp"

#include <arm_neon.h>
#include <limits>

namespace tsac::apple_arm
{
namespace
{
void parallel_for(const Execution* execution, std::size_t count,
                  const std::function<void(std::size_t)>& task)
{
    if (execution)
    {
        execution->parallel_for(count, task);
        return;
    }
    for (std::size_t i = 0; i < count; ++i)
    {
        task(i);
    }
}
} // namespace

Conv1d::Conv1d(Conv1dConfig config, std::span<const float> weights, std::span<const float> bias,
               const Limits& limits)
    : Convolution(config, weights, bias, limits)
{
    if (config.kind == ConvolutionKind::transposed)
    {
        weights_.clear();
        prepare_transposed(weights);
    }
}

Conv1d Conv1d::from_prepared(Conv1dConfig config, const weights::View& weights,
                             const weights::View& bias, const Limits& limits)
{
    detail::require(weights.as<float>().size() == detail::weight_count(config, limits) &&
                        bias.as<float>().size() == config.output_channels &&
                        config.kind == ConvolutionKind::ordinary,
                    "invalid prepared convolution");
    Conv1d result;
    result.config_ = config;
    result.bias_ = weights::Array<float>(bias);
    result.prepared_ = weights;
    return result;
}

void Conv1d::prepare_transposed(std::span<const float> weights)
{
    const auto& config = config_;
    // Each phase contains only contributing taps; avoid multiplying the
    // stride-1 zeros that a conventional transposed im2col would introduce.
    for (std::size_t tap = 0; tap < config.kernel; ++tap)
    {
        const auto residue = (tap * config.dilation) % config.stride;
        auto phase = std::find_if(phases_.begin(), phases_.end(),
                                  [&](const Phase& p)
                                  {
                                      return p.residue == residue;
                                  });
        if (phase == phases_.end())
        {
            phases_.push_back({residue, 0, {tap}});
        }
        else
        {
            phase->taps.push_back(tap);
        }
    }
    weights_.reserve(weights.size());
    for (auto& phase : phases_)
    {
        phase.weight_offset = weights_.size();
        for (auto tap : phase.taps)
        {
            for (std::size_t c = 0; c < config.input_channels; ++c)
            {
                const auto offset = (c * config.kernel + tap) * config.output_channels;
                weights_.insert(weights_.end(), weights.data() + offset,
                                weights.data() + offset + config.output_channels);
            }
        }
    }
}

std::size_t Conv1d::tile_size(std::size_t frames, const Limits& limits) const
{
    std::size_t taps = config_.kind == ConvolutionKind::ordinary ? config_.kernel : 0;
    for (const auto& phase : phases_)
    {
        taps = std::max(taps, phase.taps.size());
    }
    const auto k = checked_mul(taps, config_.input_channels);
    detail::require(k > 0 && config_.output_channels > 0, "invalid convolution tile geometry");
    // Keep lowering storage near 4 MiB, with at least one column when permitted.
    const auto cache = std::bit_floor(std::max(std::size_t{1}, (std::size_t{1024} * 1024) / k));
    const auto size = std::min({std::size_t{512}, cache, frames, limits.tensor_elements / k,
                                limits.tensor_elements / config_.output_channels});
    detail::require(size > 0, "convolution workspace exceeds element limit");
    return size;
}

void Conv1d::lower_tile(std::span<const float> input, SignalShape shape, const Tile& tile,
                        std::span<float> columns) const
{
    const auto cin = config_.input_channels;
    const auto taps = tile.phase ? tile.phase->taps.size() : config_.kernel;
    if (tile.phase)
    {
        // Phase construction guarantees equal stride residues. Divide once per
        // tap, then advance source frames by one; no inner-loop remainder test.
        const auto first = (tile.first + config_.padding) / config_.stride;
        for (std::size_t j = 0; j < taps; ++j)
        {
            const auto offset = tile.phase->taps[j] * config_.dilation / config_.stride;
            for (std::size_t t = 0; t < tile.frames; ++t)
            {
                auto* column = columns.data() + (t * taps + j) * cin;
                const auto position = first + t;
                if (position >= offset && position - offset < shape.frames)
                {
                    std::copy_n(input.data() +
                                    (tile.group * shape.frames + position - offset) * cin,
                                cin, column);
                }
                else
                {
                    std::fill_n(column, cin, 0.0f);
                }
            }
        }
        return;
    }
    for (std::size_t t = 0; t < tile.frames; ++t)
    {
        for (std::size_t j = 0; j < taps; ++j)
        {
            const auto position = (tile.first + t) * config_.stride + j * config_.dilation;
            auto* column = columns.data() + (t * taps + j) * cin;
            if (position >= config_.padding && position - config_.padding < shape.frames)
            {
                std::copy_n(input.data() +
                                (tile.group * shape.frames + position - config_.padding) * cin,
                            cin, column);
            }
            else
            {
                std::fill_n(column, cin, 0.0f);
            }
        }
    }
}

void Conv1d::apply_tile(std::span<const float> input, SignalShape shape, Signal& output,
                        const Tile& tile) const
{
    const auto& c = config_;
    const auto cin = c.input_channels, cout = c.output_channels;
    const auto k = (tile.phase ? tile.phase->taps.size() : c.kernel) * cin;
    const bool ordinary = !tile.phase;
    const bool direct = ordinary && c.kernel == 1 && c.stride == 1 && c.padding == 0;
    // Per-worker high-water storage avoids repeated allocation and zero fills.
    thread_local std::vector<float> columns, product;
    if (!direct)
    {
        if (columns.size() < tile.frames * k)
        {
            columns.resize(tile.frames * k);
        }
        lower_tile(input, shape, tile, columns);
    }
    if (!ordinary && product.size() < tile.frames * cout)
    {
        product.resize(tile.frames * cout);
    }
    const float* b =
        direct ? input.data() + (tile.group * shape.frames + tile.first) * cin : columns.data();
    float* destination =
        ordinary ? output.values.data() + (tile.group * output.shape.frames + tile.first) * cout
                 : product.data();
    const float* a = (prepared_ ? prepared_.as<float>().data() : weights_.data()) +
                     (ordinary ? 0 : tile.phase->weight_offset);
    matrix_product(ordinary, cout, tile.frames, k, a, b, destination, execution_);
    for (std::size_t t = 0; t < tile.frames; ++t)
    {
        float* out = output.values.data() +
                     (tile.group * output.shape.frames + tile.first + t * tile.step) * cout;
        for (std::size_t channel = 0; channel < cout; ++channel)
        {
            out[channel] = destination[t * cout + channel] + bias_[channel];
        }
    }
}

void Conv1d::evaluate(std::span<const float> input, SignalShape shape, Signal& output,
                      const Limits& limits) const
{
    const auto size = tile_size(output.shape.frames, limits);

    struct Jobs
    {
        std::size_t first, step, frames, end;
        const Phase* phase;
    };

    // Store one descriptor per phase, rather than one per tile/group, so even
    // a large number of tiny independent groups needs bounded scheduling space.
    std::vector<Jobs> phases;
    phases.reserve(std::max(std::size_t{1}, phases_.size()));
    std::size_t jobs_per_group = 0;
    const auto append =
        [&](std::size_t first, std::size_t step, std::size_t frames, const Phase* phase)
    {
        jobs_per_group = checked_add(jobs_per_group, (frames - 1) / size + 1);
        phases.push_back({first, step, frames, jobs_per_group, phase});
    };
    if (config_.kind == ConvolutionKind::ordinary)
    {
        append(0, 1, output.shape.frames, nullptr);
    }
    else
    {
        if (phases_.size() < config_.stride)
        {
            // Dilation/stride combinations may leave residues without taps.
            for (std::size_t t = 0; t < output.values.size(); t += config_.output_channels)
            {
                std::copy(bias_.begin(), bias_.end(), output.values.data() + t);
            }
        }
        for (const auto& phase : phases_)
        {
            const auto padding = config_.padding % config_.stride;
            const auto first = phase.residue >= padding
                                   ? phase.residue - padding
                                   : config_.stride - (padding - phase.residue);
            if (first < output.shape.frames)
            {
                append(first, config_.stride,
                       (output.shape.frames - 1 - first) / config_.stride + 1, &phase);
            }
        }
    }
    // Distinct groups, phases and tiles write disjoint output frames.
    parallel_for(execution_, checked_mul(shape.groups, jobs_per_group),
                 [&](std::size_t job)
                 {
                     const auto group = job / jobs_per_group;
                     const auto local = job % jobs_per_group;
                     std::size_t previous = 0;
                     for (const auto& phase : phases)
                     {
                         if (local < phase.end)
                         {
                             const auto begin = (local - previous) * size;
                             apply_tile(input, shape, output,
                                        {group, phase.first + begin * phase.step, phase.step,
                                         std::min(size, phase.frames - begin), phase.phase});
                             break;
                         }
                         previous = phase.end;
                     }
                 });
}

namespace
{
float32x4_t dac_sine_squared(float32x4_t phase)
{
    const auto constant = [](std::uint32_t bits)
    {
        return vdupq_n_f32(std::bit_cast<float>(bits));
    };
    const auto quadrant = vrndnq_f32(vmulq_f32(phase, constant(0x3f22f983)));
    // Match x86's indefinite integer outside the signed conversion range.
    const auto in_range = vandq_u32(vcgeq_f32(quadrant, vdupq_n_f32(-2147483648.0f)),
                                    vcltq_f32(quadrant, vdupq_n_f32(2147483648.0f)));
    const auto q = vbslq_s32(in_range, vcvtq_s32_f32(quadrant),
                             vdupq_n_s32(std::numeric_limits<std::int32_t>::min()));
    auto x = vfmaq_f32(phase, quadrant, constant(0xbfc90000));
    x = vfmaq_f32(x, quadrant, constant(0xb9fdaa22));
    const auto x2 = vmulq_f32(x, x);
    auto sine = vfmaq_f32(constant(0x3c08839e), constant(0xb94ca1f9), x2);
    sine = vfmaq_f32(constant(0xbe2aaaa3), sine, x2);
    sine = vfmaq_f32(x, vmulq_f32(x, x2), sine);
    auto cosine = vfmaq_f32(constant(0xbab6061a), constant(0x37ccf5ce), x2);
    cosine = vfmaq_f32(constant(0x3d2aaaa5), cosine, x2);
    cosine = vfmaq_f32(constant(0xbf000000), cosine, x2);
    cosine = vfmaq_f32(vdupq_n_f32(1.0f), cosine, x2);
    const auto even = vceqq_s32(vandq_s32(q, vdupq_n_s32(1)), vdupq_n_s32(0));
    const auto value = vbslq_f32(even, sine, cosine);
    return vmulq_f32(value, value);
}
} // namespace

void apple_snake_frames(std::span<const float> input, std::span<const float> alpha,
                        std::span<float> output)
{
    for (std::size_t base = 0; base < input.size(); base += alpha.size())
    {
        std::size_t channel = 0;
        {
            for (; channel + 4 <= alpha.size(); channel += 4)
            {
                const auto a = vld1q_f32(alpha.data() + channel);
                const auto x = vld1q_f32(input.data() + base + channel);
                const auto phase = vmulq_f32(a, x);
                detail::require(vmaxvq_f32(vabsq_f32(phase)) <= std::numeric_limits<float>::max(),
                                "nonfinite Snake phase");
                const auto squared = dac_sine_squared(phase);
                const auto denominator = vaddq_f32(a, vdupq_n_f32(snake_epsilon));
                vst1q_f32(output.data() + base + channel,
                          vaddq_f32(x, vdivq_f32(squared, denominator)));
            }
        }
        if (channel < alpha.size())
        {
            scalar::scalar_snake_frames(
                input.subspan(base + channel, alpha.size() - channel), alpha.subspan(channel),
                output.subspan(base + channel, alpha.size() - channel), true);
        }
    }
}

void snake_dac_into(std::span<const float> input, SignalShape shape, std::span<const float> alpha,
                    Signal& output, const Limits& limits, const Execution* execution)
{
    detail::activate_into(
        input, shape, alpha, limits, output,
        [&](auto source, auto a, auto output)
        {
            const auto frames_per_job =
                std::max(std::size_t{1}, std::size_t{32768} / shape.channels);
            const auto frames = source.size() / shape.channels;
            const auto jobs =
                source.size() >= 65536 ? (frames + frames_per_job - 1) / frames_per_job : 1;
            parallel_for(execution, jobs,
                         [&](std::size_t job)
                         {
                             const auto begin =
                                 jobs == 1 ? 0 : job * frames_per_job * shape.channels;
                             const auto count = jobs == 1
                                                    ? source.size()
                                                    : std::min(frames_per_job * shape.channels,
                                                               source.size() - begin);
                             apple_snake_frames(source.subspan(begin, count), a,
                                                output.subspan(begin, count));
                         });
        });
}

Signal snake_dac(std::span<const float> input, SignalShape shape, std::span<const float> alpha,
                 const Limits& limits, const Execution* execution)
{
    Signal output;
    snake_dac_into(input, shape, alpha, output, limits, execution);
    return output;
}
} // namespace tsac::apple_arm
