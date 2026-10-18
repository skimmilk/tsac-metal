#include "../probability/parameters.h"
#include "common/internal.hpp"
#include "format/range.hpp"
#include "session.hpp"

#include <algorithm>

namespace tsac::metal::transformer
{
std::vector<std::uint16_t> Session::decode(ByteView stream, const EntropyObserver& observer)
{
    const auto h = parse_header(stream, model.limits);
    detail::require(h.method == 1, "resident decode requires normal framing");
    const auto payload = stream.subspan(h.payload_offset, h.payload_bytes);
    // Preserve initial-interval validation, including empty token streams.
    const RangeDecoder initial(payload);
    const auto count = std::min<std::size_t>(h.batch, h.frames) * h.groups;
    reset(h, count);
    std::vector<std::uint16_t> output(h.symbol_count());
    if (!count)
    {
        verify_token_crc(h, output);
        return output;
    }
    detail::require(count * 1024 <= model.limits.tensor_elements,
                    "decode distributions exceed element limit");
    auto tokens = Buffer::allocate(checked_mul(output.size(), sizeof(std::uint16_t)));
    auto compressed = Buffer::copy(payload.data(), payload.size());
    std::uint32_t code = 0;
    for (std::size_t i = 0; i < 3; ++i)
    {
        code = (code << 8) | (i < payload.size() ? payload[i] : 0);
    }
    const tsac_metal_abi::RangeState start{
        0xff0000, code, static_cast<std::uint32_t>(std::min<std::size_t>(3, payload.size())), 0};
    auto range = Buffer::copy(&start, sizeof start);
    auto probabilities = Buffer::allocate(count * 2048, BufferStorage::gpu_only),
         trees = Buffer::allocate(count * 4096, BufferStorage::gpu_only);
    auto& graph = recorder;
    const auto steps = h.frames / h.batch + (h.frames % h.batch != 0);
    std::size_t context = 0, chunk_start = 0, chunk_steps = 0;
    graph.begin();
    const auto publish = [&](std::size_t first, std::size_t end)
    {
        if (!observer.before_symbol && !observer.after_symbol)
        {
            return;
        }
        for (std::size_t step = first; step < end; ++step)
        {
            for (std::size_t book = 0; book < h.codebooks; ++book)
            {
                for (std::size_t lane = 0; lane < std::min<std::size_t>(h.batch, h.frames); ++lane)
                {
                    if (step >= h.frames / h.batch + (lane < h.frames % h.batch))
                    {
                        continue;
                    }
                    const auto frame = lane * (h.frames / h.batch) +
                                       std::min<std::size_t>(lane, h.frames % h.batch) + step;
                    for (std::size_t group = 0; group < h.groups; ++group)
                    {
                        const EntropyPosition position{
                            frame, group, book,
                            lane,  step,  (frame * h.groups + group) * h.codebooks + book};
                        if (observer.before_symbol)
                        {
                            observer.before_symbol(position);
                        }
                        if (observer.after_symbol)
                        {
                            observer.after_symbol(position, 0);
                        }
                    }
                }
            }
        }
    };
    const auto flush = [&](std::size_t end)
    {
        if (chunk_steps)
        {
            graph.finish();
            publish(chunk_start, end);
            chunk_steps = 0;
        }
        graph.begin();
        chunk_start = end;
    };
    for (std::size_t step = 0; step < steps; ++step)
    {
        const auto active = (step < h.frames / h.batch ? std::min<std::size_t>(h.batch, h.frames)
                                                       : h.frames % h.batch) *
                            h.groups;
        const bool replay = context == context_frames;
        if (replay)
        {
            flush(step);
            // Rollover feeds exactly the last 256 complete token frames with
            // the original replay rounding. Bound recording/scratch even here.
            for (std::size_t lane = 0; lane < active; ++lane)
            {
                reset_temporal(lane);
                for (std::size_t t = 0; t < replay_frames; ++t)
                {
                    graph.recycle();
                    auto embedded =
                        graph.embed_resident(model.embeddings, tokens, h,
                                             Prediction::ResidentPosition{step - replay_frames + t},
                                             h.codebooks, lane, 1);
                    auto input = graph.linear(model.temporal_projection, embedded);
                    auto result =
                        evaluate_stack(graph, model.temporal, lanes[lane].temporal, input, t,
                                       model.rotations.slice(t * 256, 256), true, true);
                    graph.copy(result, lanes[lane].conditioning);
                    if ((t + 1) % submission_steps == 0 || t + 1 == replay_frames)
                    {
                        graph.finish();
                        graph.begin();
                    }
                }
            }
            context = replay_frames;
        }
        for (std::size_t book = 0; book < h.codebooks; ++book)
        {
            for (std::size_t first = 0; first < active; first += batch_columns)
            {
                const auto columns = std::min(batch_columns, active - first);
                graph.recycle();
                std::vector<StackCache*> caches;
                caches.reserve(columns);
                std::vector<std::size_t> positions(columns, book ? book : context);
                for (std::size_t c = 0; c < columns; ++c)
                {
                    caches.push_back(book ? &lanes[first + c].codebook
                                          : &lanes[first + c].temporal);
                }
                Buffer input = graph.temporary(columns * 1024);
                if (!book && replay)
                {
                    for (std::size_t c = 0; c < columns; ++c)
                    {
                        graph.copy(lanes[first + c].conditioning, input.slice(c * 1024, 1024));
                    }
                }
                else
                {
                    input = graph.embed_linear_resident(
                        (book ? model.codebook_projection : model.temporal_projection),
                        model.embeddings, tokens, h,
                        Prediction::ResidentPosition{book ? step : (step ? step - 1 : 0),
                                                     !book && !step},
                        book ? book : h.codebooks, first, columns);
                    if (!book)
                    {
                        std::vector<Buffer> keys, values;
                        if (!temporal_keys.empty() && (context < 32 || context >= 64))
                        {
                            for (std::size_t layer = 0; layer < model.temporal.layers.size();
                                 ++layer)
                            {
                                const auto stride = temporal_extent * 1024;
                                keys.push_back(
                                    temporal_keys[layer].slice(first * stride, columns * stride));
                                values.push_back(
                                    temporal_values[layer].slice(first * stride, columns * stride));
                            }
                        }
                        input =
                            evaluate_batch(graph, model.temporal, caches, input, positions,
                                           model.rotations, true, keys, values, temporal_extent);
                    }
                }
                if (!book)
                {
                    for (std::size_t c = 0; c < columns; ++c)
                    {
                        reset_codebook(first + c);
                        caches[c] = &lanes[first + c].codebook;
                    }
                }
                std::fill(positions.begin(), positions.end(), book);
                input = graph.add(input, model.positions.slice(book * 1024, 1024), columns * 512);
                std::vector<Buffer> keys, values;
                for (std::size_t layer = 0; layer < model.codebook.layers.size(); ++layer)
                {
                    const auto stride = std::size_t(h.codebooks) * 1024;
                    keys.push_back(codebook_keys[layer].slice(first * stride, columns * stride));
                    values.push_back(
                        codebook_values[layer].slice(first * stride, columns * stride));
                }
                auto hidden =
                    evaluate_batch(graph, model.codebook, caches, input, positions, model.rotations,
                                   false, keys, values, h.codebooks, false);
                auto logits =
                    graph.norm_linear(model.output_projection, hidden, model.codebook.final.gain,
                                      model.codebook.final.bias);
                graph.softmax_decode(logits, probabilities.slice(first * 2048, columns * 2048),
                                     trees.slice(first * 4096, columns * 4096));
            }
            graph.range_decode(probabilities, trees, compressed, range, tokens, h, step, book,
                               active);
        }
        if (!replay)
        {
            ++context;
        }
        if (++chunk_steps == submission_steps || step + 1 == steps)
        {
            flush(step + 1);
        }
    }
    tokens.read(output.data(), output.size() * 2);
    verify_token_crc(h, output);
    return output;
}

} // namespace tsac::metal::transformer
