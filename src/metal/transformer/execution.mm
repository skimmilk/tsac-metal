#include "common/internal.hpp"
#include "common/numeric/bf16.hpp"
#include "session.hpp"

#include <algorithm>
#include <array>

namespace tsac::metal::transformer
{
Buffer evaluate_stack(Prediction& graph, const Stack& stack, StackCache& cache, Buffer input,
                      std::size_t position, const Buffer& rotation, bool rotated, bool prefill)
{
    for (std::size_t i = 0; i < stack.layers.size(); ++i)
    {
        const auto& layer = stack.layers[i];
        auto& state = cache.layers[i];
        detail::require(state.position == position, "Metal Transformer cache position mismatch");
        const auto qkv = graph.norm_linear(layer.qkv, input, layer.first.gain, layer.first.bias);
        const auto attended =
            graph.attention(qkv, state.key, state.value, rotation, position, rotated, prefill);
        input = graph.linear(layer.attention_out, attended, false, &input);
        const auto hidden =
            graph.norm_linear(layer.mlp_in, input, layer.second.gain, layer.second.bias);
        input = graph.linear(layer.mlp_out, hidden, true, &input);
        ++state.position;
    }
    return graph.norm(input, stack.final.gain, stack.final.bias);
}

Buffer evaluate_batch(Prediction& graph, const Stack& stack, std::span<StackCache*> caches,
                      Buffer input, std::span<const std::size_t> positions, const Buffer& rotations,
                      bool rotated, std::span<const Buffer> bulk_keys,
                      std::span<const Buffer> bulk_values, std::size_t cache_frames,
                      bool final_norm, bool replay)
{
    const auto columns = caches.size();
    detail::require(!replay || bulk_keys.empty(), "Replay requires explicit rounded attention");
    for (std::size_t i = 0; i < stack.layers.size(); ++i)
    {
        const auto& layer = stack.layers[i];
        const auto qkv = graph.norm_linear(layer.qkv, input, layer.first.gain, layer.first.bias);
        auto attended = bulk_keys.empty() ? graph.temporary(columns * 512 * 2)
                        : rotated
                            ? graph.attention_temporal(qkv, bulk_keys[i], bulk_values[i],
                                                       rotations.slice(positions[0] * 256, 256),
                                                       positions[0], columns, cache_frames)
                            : graph.attention_columns(qkv, bulk_keys[i], bulk_values[i],
                                                      positions[0], columns, cache_frames);
        if (!bulk_keys.empty())
        {
            for (std::size_t c = 0; c < columns; ++c)
            {
                if (caches[c])
                {
                    auto& state = caches[c]->layers[i];
                    detail::require(state.position == positions[c],
                                    "Metal batch cache position mismatch");
                    detail::require(positions[c] < state.key.size() / 1024,
                                    "Metal batch exceeds logical lane cache");
                    ++state.position;
                }
            }
        }
        else
        {
            for (std::size_t c = 0; c < columns; ++c)
            {
                auto& state = caches[c]->layers[i];
                detail::require(state.position == positions[c],
                                "Metal batch cache position mismatch");
                const auto destination = attended.slice(c * 1024, 1024);
                graph.attention(qkv.slice(c * 1536 * 2, std::size_t{1536} * 2), state.key,
                                state.value,
                                rotated ? rotations.slice(positions[c] * 256, 256) : rotations,
                                positions[c], rotated, replay, &destination);
                ++state.position;
            }
        }
        input = graph.linear(layer.attention_out, attended, false, &input);
        const auto hidden =
            graph.norm_linear(layer.mlp_in, input, layer.second.gain, layer.second.bias);
        input = graph.linear(layer.mlp_out, hidden, true, &input);
    }
    return final_norm ? graph.norm(input, stack.final.gain, stack.final.bias) : input;
}

std::vector<std::span<const float>> Session::predict_batch(std::span<const Request> requests)
{
    auto& graph = recorder;
    detail::require(!requests.empty(), "invalid Metal prediction batch");
    const auto columns = requests.size(), book = requests.front().book;
    detail::require(book < header.codebooks, "invalid Metal batch book");
    graph.begin();
    std::vector<std::uint16_t> tokens;
    std::vector<StackCache*> caches;
    std::vector<std::size_t> positions;
    for (const auto& r : requests)
    {
        detail::require(r.book == book && r.prefix.size() == book && r.position <= 1024,
                        "inconsistent Metal prediction batch");
        auto& lane = lanes.at(r.lane);
        const auto source = book == 0 ? r.feedback : r.prefix;
        detail::require(book != 0 || (r.position < 1024 && source.size() == header.codebooks),
                        "invalid Metal temporal batch");
        tokens.insert(tokens.end(), source.begin(), source.end());
        caches.push_back(book == 0 ? &lane.temporal : &lane.codebook);
        positions.push_back(book == 0 ? r.position : book);
    }
    const bool replayed = book == 0 && requests.front().use_conditioning;
    for (const auto& r : requests)
    {
        detail::require(r.use_conditioning == replayed, "Inconsistent replay prediction batch");
    }
    auto input =
        replayed ? graph.temporary(columns * 1024)
                 : graph.linear(book == 0 ? model.temporal_projection : model.codebook_projection,
                                graph.embed(model.embeddings, tokens, columns));
    if (book == 0 && !replayed)
    {
        bool contiguous = !temporal_keys.empty() && (positions[0] < 32 || positions[0] >= 64);
        for (std::size_t c = 0; c < columns; ++c)
        {
            contiguous &= requests[c].lane == requests[0].lane + c && positions[c] == positions[0];
        }
        std::vector<Buffer> keys, values;
        if (contiguous)
        {
            for (std::size_t layer = 0; layer < 12; ++layer)
            {
                const auto stride = temporal_extent * 1024;
                keys.push_back(
                    temporal_keys[layer].slice(requests[0].lane * stride, columns * stride));
                values.push_back(
                    temporal_values[layer].slice(requests[0].lane * stride, columns * stride));
            }
        }
        input = evaluate_batch(graph, model.temporal, caches, input, positions, model.rotations,
                               true, keys, values, temporal_extent);
    }
    if (book == 0)
    {
        for (std::size_t c = 0; c < columns; ++c)
        {
            auto& lane = lanes.at(requests[c].lane);
            if (replayed)
            {
                graph.copy(lane.conditioning, input.slice(c * 1024, 1024));
            }
            else
            {
                graph.copy(input.slice(c * 1024, 1024), lane.conditioning);
            }
            reset_codebook(requests[c].lane);
            caches[c] = &lane.codebook;
            positions[c] = 0;
        }
    }
    input = graph.add(input, model.positions.slice(book * 1024, 1024), columns * 512);
    std::vector<Buffer> keys, values;
    bool contiguous = true;
    for (std::size_t c = 0; c < columns; ++c)
    {
        contiguous &= requests[c].lane == requests[0].lane + c;
    }
    if (contiguous)
    {
        for (std::size_t layer = 0; layer < 3; ++layer)
        {
            const auto stride = std::size_t(header.codebooks) * 1024;
            keys.push_back(codebook_keys[layer].slice(requests[0].lane * stride, columns * stride));
            values.push_back(
                codebook_values[layer].slice(requests[0].lane * stride, columns * stride));
        }
    }
    const auto hidden = evaluate_batch(graph, model.codebook, caches, input, positions,
                                       model.rotations, false, keys, values, header.codebooks);
    const auto readback = graph.shared_temporary(columns * 2048);
    graph.softmax(graph.linear(model.output_projection, hidden), &readback);
    graph.finish();
    const auto bits = readback.mapped<std::uint16_t>();
    std::vector<std::span<const float>> results;
    for (std::size_t c = 0; c < columns; ++c)
    {
        auto& output = lanes.at(requests[c].lane).probabilities;
        output.resize(1024);
        std::transform(bits.begin() + static_cast<std::ptrdiff_t>(c * 1024),
                       bits.begin() + static_cast<std::ptrdiff_t>((c + 1) * 1024), output.begin(),
                       probability_float);
        results.push_back(output);
    }
    return results;
}

Buffer Session::predict_known(std::span<const Request> requests, std::size_t lane_count)
{
    auto& graph = recorder;
    detail::require(lane_count && requests.size() % lane_count == 0, "invalid known-token batch");
    const auto columns = requests.size();
    const std::size_t books = header.codebooks;
    graph.begin();
    std::vector<StackCache*> caches;
    std::vector<std::size_t> positions;
    std::vector<std::uint16_t> tokens;
    for (const auto& r : requests)
    {
        detail::require(r.position < 1024 && r.feedback.size() == books && r.prefix.size() == books,
                        "invalid known-token request");
        caches.push_back(&lanes.at(r.lane).temporal);
        positions.push_back(r.position);
        tokens.insert(tokens.end(), r.feedback.begin(), r.feedback.end());
    }
    // Layer-wise causal prefill: the projections share weights across time as
    // well as lanes. Each query still sees only its original history, with its
    // original rotary position and ordinary (not replay) attention rounding.
    auto embedded =
        graph.linear(model.temporal_projection, graph.embed(model.embeddings, tokens, columns));
    auto conditioning =
        evaluate_batch(graph, model.temporal, caches, embedded, positions, model.rotations, true);
    // Codebook sequences are independent across frames. All books for this
    // bounded chunk use true prefixes; only temporal caches survive the chunk.
    std::vector<Buffer> keys, values;
    for (std::size_t i = 0; i < model.codebook.layers.size(); ++i)
    {
        keys.push_back(graph.temporary(columns * books * 512 * 2));
        values.push_back(graph.temporary(columns * books * 512 * 2));
    }
    caches.assign(columns, nullptr);
    positions.resize(columns);
    auto distributions = graph.shared_temporary(columns * books * 1024 * 2);
    for (std::size_t book = 0; book < books; ++book)
    {
        auto input = conditioning;
        if (book)
        {
            tokens.clear();
            for (const auto& r : requests)
            {
                tokens.insert(tokens.end(), r.prefix.begin(),
                              r.prefix.begin() + static_cast<std::ptrdiff_t>(book));
            }
            input = graph.linear(model.codebook_projection,
                                 graph.embed(model.embeddings, tokens, columns));
        }
        std::fill(positions.begin(), positions.end(), book);
        input = graph.add(input, model.positions.slice(book * 1024, 1024), columns * 512);
        auto hidden = evaluate_batch(graph, model.codebook, caches, input, positions,
                                     model.rotations, false, keys, values, books);
        const auto destination = distributions.slice(book * columns * 2048, columns * 2048);
        graph.softmax(graph.linear(model.output_projection, hidden), &destination);
    }
    graph.finish();
    return distributions;
}

} // namespace tsac::metal::transformer
