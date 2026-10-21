#include "metal/weight_preparation.hpp"

#include "common/numeric.hpp"
#include "weights/normalization.hpp"
#include "weights/preparation.hpp"

#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <exception>
#include <functional>
#include <mutex>

namespace tsac::metal
{
namespace
{
// A small number of contiguous ranges bounds scratch and avoids one queued job
// per column/tile. GCD shares workers between concurrent model preparations.
void prepare_ranges(std::size_t count, bool parallel,
                    const std::function<void(std::size_t, std::size_t)>& task)
{
    const auto workers = parallel ? std::min<std::size_t>(4, count) : 1;
    if (workers <= 1)
    {
        task(0, count);
        return;
    }

    struct Context
    {
        const std::function<void(std::size_t, std::size_t)>& task;
        std::size_t count, workers;
        std::mutex mutex;
        std::exception_ptr error;
    } context{task, count, workers, {}, {}};

    dispatch_apply_f(workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &context,
                     [](void* opaque, std::size_t worker)
                     {
                         auto& c = *static_cast<Context*>(opaque);
                         try
                         {
                             c.task(c.count * worker / c.workers,
                                    c.count * (worker + 1) / c.workers);
                         }
                         catch (...)
                         {
                             std::lock_guard lock(c.mutex);
                             if (!c.error)
                             {
                                 c.error = std::current_exception();
                             }
                         }
                     });
    if (context.error)
    {
        std::rethrow_exception(context.error);
    }
}
} // namespace

void prepare_dac_weights(const ConvolutionWeights& weights, bool transposed,
                         std::span<float> output, const Limits& limits)
{
    const auto axes = weights.values.shape().axes;
    const auto fan_in = checked_mul(axes[0], axes[1]);
    const auto count = weights.values.shape().elements();
    detail::require(count <= limits.tensor_elements, "Metal convolution exceeds element limit");
    detail::require(output.size() == count && weights.gain.size() == axes[2],
                    "Metal convolution normalization shape mismatch");
    const auto gain = weights.gain.values();
    constexpr std::size_t columns = 32;
    prepare_ranges((axes[2] + columns - 1) / columns, count >= 65536,
                   [&](std::size_t begin, std::size_t end)
                   {
                       std::vector<float> tile(fan_in * std::min(columns, axes[2]));
                       for (std::size_t first = begin * columns;
                            first < std::min(end * columns, axes[2]); first += columns)
                       {
                           const auto width = std::min(columns, axes[2] - first);
                           auto values = std::span(tile).first(fan_in * width);
                           weights::expand_into(weights.values, first * fan_in, values);
                           fold_dac_weight_norm_into(values, gain.subspan(first, width), fan_in,
                                                     values, limits);
                           if (!transposed)
                           {
                               // Transpose a bounded output-channel tile into [out, in, tap].
                               // The stride is the full output width, even for the final tile.
                               for (std::size_t i = 0; i < fan_in; ++i)
                               {
                                   for (std::size_t c = 0; c < width; ++c)
                                   {
                                       output[first + c + axes[2] * i] = values[c * fan_in + i];
                                   }
                               }
                           }
                           else
                           {
                               // The source normalization columns are input channels for a
                               // transposed convolution; output channels stay contiguous.
                               for (std::size_t tap = 0; tap < axes[1]; ++tap)
                               {
                                   for (std::size_t c = 0; c < width; ++c)
                                   {
                                       std::copy_n(
                                           values.data() + c * fan_in + tap * axes[0], axes[0],
                                           output.data() + axes[0] * (first + c + axes[2] * tap));
                                   }
                               }
                           }
                       }
                   });
}

void pack_linear_weights(std::span<const std::int8_t> source, std::size_t input_width,
                         std::size_t output_width, std::size_t rows, std::span<std::int8_t> output)
{
    detail::require(
        input_width && input_width % 16 == 0 && (rows == 2 || rows == 8) &&
            source.size() == checked_mul(input_width, output_width) &&
            output.size() ==
                checked_mul(input_width, checked_add(output_width, rows - 1) / rows * rows),
        "invalid Metal linear packing shape");
    prepare_ranges((output_width + rows - 1) / rows, source.size() >= 262144,
                   [&](std::size_t begin, std::size_t end)
                   {
                       for (std::size_t first = begin * rows;
                            first < std::min(end * rows, output_width); first += rows)
                       {
                           auto* dest = output.data() + first * input_width;
                           if (output_width - first < rows)
                           {
                               // Only a partial final tile needs zero padding.
                               std::fill_n(dest, input_width * rows, std::int8_t{0});
                               for (std::size_t q = 0; q < input_width / 4; ++q)
                               {
                                   for (std::size_t r = 0; r < output_width - first; ++r)
                                   {
                                       std::copy_n(source.data() + (first + r) * input_width +
                                                       q * 4,
                                                   4, dest + q * 4 * rows + r * 4);
                                   }
                               }
                               continue;
                           }
                           for (std::size_t i = 0; i < input_width; i += 16)
                           {
                               const auto load = [&](std::size_t r)
                               {
                                   return vreinterpretq_u32_s8(
                                       vld1q_s8(source.data() + (first + r) * input_width + i));
                               };
                               const auto store = [&](std::size_t offset, uint32x4_t value)
                               {
                                   vst1q_s8(dest + i * rows + offset, vreinterpretq_s8_u32(value));
                               };
                               if (rows == 2)
                               {
                                   const auto a = load(0), b = load(1);
                                   store(0, vzip1q_u32(a, b));
                                   store(16, vzip2q_u32(a, b));
                               }
                               else
                               {
                                   for (std::size_t r = 0; r < 8; r += 4)
                                   {
                                       const auto a = load(r), b = load(r + 1), c = load(r + 2),
                                                  d = load(r + 3);
                                       const auto ac0 = vzip1q_u32(a, c), ac1 = vzip2q_u32(a, c);
                                       const auto bd0 = vzip1q_u32(b, d), bd1 = vzip2q_u32(b, d);
                                       store(r * 4, vzip1q_u32(ac0, bd0));
                                       store(32 + r * 4, vzip2q_u32(ac0, bd0));
                                       store(64 + r * 4, vzip1q_u32(ac1, bd1));
                                       store(96 + r * 4, vzip2q_u32(ac1, bd1));
                                   }
                               }
                           }
                       }
                   });
}
} // namespace tsac::metal
