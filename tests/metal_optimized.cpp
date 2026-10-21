#include "apple-arm/pipeline.hpp"
#include "common/analysis.hpp"
#include "common/packed_linear.hpp"
#include "common/probability.hpp"
#include "common/synthesis.hpp"
#include "metal/audio_dac.hpp"
#include "metal/audio_rvq.hpp"
#include "metal/dac_graph.hpp"
#include "metal/prediction.hpp"
#include "metal/transformer.hpp"
#include "scalar/dsp.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iostream>
#include <limits>

using namespace tsac;

void check(bool ok, const char* text)
{
    if (!ok)
    {
        throw Error(text);
    }
}

void prediction_slots(const TransformerModel& model)
{
    metal::Prediction graph;
    auto weights = MetalLinear::from_weights(model.temporal_projection());
    // Invalid input in slot A must survive clearing/using slot B's error word.
    graph.begin();
    std::vector<std::uint16_t> invalid(
        apple_arm::PackedLinear::from_weights(model.temporal_projection()).input_width(), 0x7f80);
    graph.linear(weights, graph.upload(invalid));
    const auto bad = graph.submit();
    graph.begin();
    const std::array values{probability_bf16(3), probability_bf16(7)};
    const auto output = metal::Buffer::allocate(sizeof values);
    graph.copy(graph.upload(values), output);
    const auto good = graph.submit();
    bool rejected = false;
    try
    {
        graph.wait(good);
    }
    catch (const Error&)
    {
        rejected = true;
    }
    check(rejected, "Second slot cleared the first slot's GPU error");
    graph.wait(bad);
    graph.wait(good);
    check(std::equal(values.begin(), values.end(), output.mapped<std::uint16_t>().begin()),
          "Queued prediction staging was overwritten");
    // Reuse both slots after an error, with different CPU-populated input.
    for (unsigned pair = 0; pair < 3; ++pair)
    {
        std::array<metal::Buffer, 2> outputs{metal::Buffer::allocate(sizeof values),
                                             metal::Buffer::allocate(sizeof values)};
        for (unsigned slot = 0; slot < 2; ++slot)
        {
            graph.begin();
            const std::array input{probability_bf16(float(pair * 2 + slot)),
                                   probability_bf16(float(pair * 2 + slot + 1))};
            graph.copy(graph.upload(input), outputs[slot]);
            graph.submit();
        }
        if (!pair)
        {
            bool invalid_ticket = false;
            try
            {
                graph.wait(std::numeric_limits<metal::Prediction::Ticket>::max());
            }
            catch (const Error&)
            {
                invalid_ticket = true;
            }
            check(invalid_ticket, "Prediction accepted an unknown submission ticket");
        }
        graph.drain();
        for (unsigned slot = 0; slot < 2; ++slot)
        {
            const auto actual = outputs[slot].mapped<std::uint16_t>();
            check(actual[0] == probability_bf16(float(pair * 2 + slot)) &&
                      actual[1] == probability_bf16(float(pair * 2 + slot + 1)),
                  "Prediction slot reuse changed staged CPU input");
        }
    }
}

void projections(const TransformerModel& model)
{
    metal::Prediction graph;
    for (bool fused_preparation : {false, true})
    {
        graph.fuse_projections(fused_preparation);

        for (const auto& ref : {model.attention_qkv(TransformerStack::temporal, 0),
                                model.mlp_input(TransformerStack::temporal, 0),
                                model.mlp_output(TransformerStack::temporal, 0),
                                model.temporal_projection(), model.output_projection()})
        {
            auto cpu = apple_arm::PackedLinear::from_weights(ref);
            auto gpu = MetalLinear::from_weights(ref);
            for (std::size_t columns : {1u, 2u, 3u, 6u, 8u, 14u, 16u, 24u, 32u, 48u, 64u, 96u})
            {
                std::vector<std::uint16_t> input(cpu.input_width() * columns);
                for (std::size_t i = 0; i < input.size(); ++i)
                {
                    input[i] =
                        probability_bf16(float(int((i * 73 + i / 17) % 257) - 128) * 0.03125f);
                }
                const auto expected = cpu.apply(input);
                graph.begin();
                auto output = graph.linear(gpu, graph.upload(input));
                graph.finish();
                std::vector<std::uint16_t> actual(expected.size());
                output.read(actual.data(), actual.size() * 2);
                check(actual == expected, "batched projection differs from CPU");
                {
                    auto activated = input;
                    probability_gelu_bf16(activated);
                    auto fused_expected = cpu.apply(activated);
                    std::vector<std::uint16_t> skip(fused_expected.size());
                    for (std::size_t i = 0; i < skip.size(); ++i)
                    {
                        skip[i] = probability_bf16(float(int(i % 31) - 15) / 32);
                    }
                    apple_arm::add_bf16(fused_expected, skip);
                    graph.begin();
                    const auto residual = graph.upload(skip);
                    auto fused = graph.linear(gpu, graph.upload(input), true, &residual);
                    graph.finish();
                    fused.read(actual.data(), actual.size() * 2);
                    check(actual == fused_expected,
                          "packed GELU/residual projection differs from CPU");
                }
            }
        }
    }
}

void projection_format_edges(const TransformerModel& model)
{
    metal::Prediction graph;
    constexpr std::array<std::uint16_t, 12> bits{0,      0x8000, 0x0001, 0x8001, 0x3f00, 0xbf00,
                                                 0x3fc0, 0xbfc0, 0x42fe, 0xc300, 0x2f80, 0xaf80};

    for (const auto& ref :
         {model.attention_qkv(TransformerStack::temporal, 0),
          model.mlp_input(TransformerStack::temporal, 0),
          model.mlp_output(TransformerStack::temporal, 0), model.temporal_projection()})
    {
        auto cpu = apple_arm::PackedLinear::from_weights(ref);
        auto gpu = MetalLinear::from_weights(ref);
        // Reuse the same scratch slots while moving into and out of wide storage.
        for (std::size_t columns : {3u, 1u, 3u, 2u, 6u, 3u})
        {
            std::vector<std::uint16_t> input(cpu.input_width() * columns);
            for (std::size_t i = 0; i < input.size(); ++i)
            {
                input[i] = (i / 256) % 3 == 0 ? 0 : bits[i % bits.size()];
            }
            for (bool gelu : {false, true})
            {
                auto prepared = input;
                if (gelu)
                {
                    probability_gelu_bf16(prepared);
                }
                const auto expected = cpu.apply(prepared);
                graph.begin();
                auto output = graph.linear(gpu, graph.upload(input), gelu);
                graph.finish();
                std::vector<std::uint16_t> actual(expected.size());
                output.read(actual.data(), actual.size() * 2);
                check(actual == expected, "quantized storage transition changed projection bits");
            }
        }
        for (std::uint16_t invalid : {std::uint16_t{0x7f80}, std::uint16_t{0x7fc1}})
        {
            std::vector<std::uint16_t> input(cpu.input_width() * 3, 0);
            input[0] = invalid;
            bool rejected = false;
            try
            {
                graph.begin();
                graph.linear(gpu, graph.upload(input));
                graph.finish();
            }
            catch (const Error&)
            {
                rejected = true;
            }
            check(rejected, "wide quantization accepted nonfinite input");
        }
    }
}

void normalized_projections(const TransformerModel& model)
{
    metal::Prediction graph;
    graph.fuse_projections(true);
    std::vector<std::uint16_t> gain(512), bias(512);
    for (std::size_t i = 0; i < 512; ++i)
    {
        gain[i] = probability_bf16(1.0f + float(int(i % 7) - 3) / 32);
        bias[i] = probability_bf16(float(int(i % 13) - 6) / 64);
    }
    auto g = metal::Buffer::copy(gain.data(), 1024), b = metal::Buffer::copy(bias.data(), 1024);

    for (const auto& ref :
         {model.attention_qkv(TransformerStack::temporal, 0),
          model.attention_output(TransformerStack::temporal, 0),
          model.mlp_input(TransformerStack::temporal, 0), model.output_projection()})
    {
        auto cpu = apple_arm::PackedLinear::from_weights(ref);
        auto gpu = MetalLinear::from_weights(ref);
        for (std::size_t columns : {1u, 2u, 3u, 6u, 14u, 24u, 32u, 96u})
        {
            std::vector<std::uint16_t> input(columns * 512), normalized;
            for (std::size_t i = 0; i < input.size(); ++i)
            {
                input[i] = probability_bf16(float(int((i * 73 + i / 17) % 257) - 128) / 32);
            }
            for (std::size_t c = 0; c < columns; ++c)
            {
                auto n =
                    probability_layer_norm_bf16(std::span(input).subspan(c * 512, 512), gain, bias);
                normalized.insert(normalized.end(), n.begin(), n.end());
            }
            auto expected = cpu.apply(normalized);
            graph.begin();
            auto result = graph.norm_linear(gpu, graph.upload(input), g, b);
            graph.finish();
            std::vector<std::uint16_t> actual(expected.size());
            result.read(actual.data(), actual.size() * 2);
            check(actual == expected, "prepared normalization/projection differs from CPU");
        }
    }
}

void known(const TransformerModel& model)
{
    auto cpu = apple_arm::CpuTransformer::from_model(model);
    auto gpu = MetalTransformer::from_model(model);
    TransformerPredictor predictor(cpu);
    for (const auto shape : {std::array<unsigned, 3>{17, 8, 16},
                             {101, 3, 1},
                             {51, 3, 1},
                             {35, 3, 2},
                             {17, 1, 1},
                             {3, 8, 2},
                             {1026, 1, 1}})
    {
        Header h;
        h.method = 1;
        h.frames = shape[0];
        h.batch = static_cast<std::uint16_t>(shape[1]);
        h.groups = static_cast<std::uint8_t>(shape[2]);
        h.model_channels = 2;
        h.codebooks = shape[0] > 1000 ? 1 : 12;
        std::vector<std::uint16_t> tokens(h.symbol_count());
        for (std::size_t i = 0; i < tokens.size(); ++i)
        {
            tokens[i] = static_cast<std::uint16_t>((i * 73 + 19) % 1024);
        }
        check(gpu.encode(h, tokens) == pack_normal(h, tokens, predictor),
              "known-token Metal 4 encoding differs from CPU stream");
        std::cout << "known tokens frames=" << h.frames << " passed\n";
    }
    Limits limited;
    limited.cache_bytes = 2300000;
    auto bounded = MetalTransformer::from_model(model, limited);
    Header h;
    h.method = 1;
    h.frames = 90;
    h.batch = 3;
    h.model_channels = 2;
    h.codebooks = 1;
    std::vector<std::uint16_t> tokens(h.symbol_count());
    for (std::size_t i = 0; i < tokens.size(); ++i)
    {
        tokens[i] = static_cast<std::uint16_t>((i * 73 + 19) % 1024);
    }
    check(pack_normal(h, tokens, predictor) == bounded.encode(h, tokens),
          "bounded lookahead changed stream");
}

void rvq_search_edges()
{
    std::vector<float> codebook(8192, 0), query(8, 1);
    for (std::size_t i = 0; i < 8; ++i)
    {
        codebook[std::size_t{7} * 8 + i] = codebook[std::size_t{93} * 8 + i] = 1;
    }
    check(metal_rvq_search(query, codebook) == std::vector<std::uint16_t>{7},
          "parallel RVQ tie did not select lowest index");
    std::fill(query.begin(), query.end(), 0);
    check(metal_rvq_search_raw_projected(query, codebook) == std::vector<std::uint16_t>{0},
          "zero RVQ tie changed");
    std::fill(query.begin(), query.end(), std::numeric_limits<float>::max());
    bool rejected = false;
    try
    {
        metal_rvq_search_raw_projected(query, codebook);
    }
    catch (const Error&)
    {
        rejected = true;
    }
    check(rejected, "parallel RVQ accepted overflowing normalization");
}

void hot_convolutions()
{
    for (std::size_t channels : {96u, 192u, 384u, 768u})
    {
        for (std::size_t dilation : {1u, 3u, 9u})
        {
            const Conv1dConfig config{channels, channels, 7, 1, 3 * dilation, dilation};
            std::vector<float> weights(channels * channels * 7), bias(channels);
            for (std::size_t i = 0; i < weights.size(); ++i)
            {
                weights[i] = float(int((i * 17 + i / 13) % 127) - 63) / 1024;
            }
            for (std::size_t i = 0; i < channels; ++i)
            {
                bias[i] = float(int(i % 17) - 8) / 128;
            }
            scalar::Conv1d cpu(config, weights, bias);
            auto gpu = MetalConvolution::prepare(config, weights, bias);
            // Exercise empty interior regions, padding thresholds, complete/partial
            // tiles, and the first interior tile even at the largest dilation.
            std::vector<std::size_t> lengths{
                1,   3 * dilation, 3 * dilation + 1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127,
                128, 129};
            std::sort(lengths.begin(), lengths.end());
            lengths.erase(std::unique(lengths.begin(), lengths.end()), lengths.end());
            for (std::size_t frames : lengths)
            {
                for (std::size_t groups : {1u, 2u})
                {
                    SignalShape shape{channels, frames, groups};
                    std::vector<float> input(shape.elements());
                    for (std::size_t i = 0; i < input.size(); ++i)
                    {
                        input[i] =
                            i % 19 == 0 ? -0.0f : float(int((i * 73 + i / 7) % 257) - 128) / 128;
                    }
                    auto expected = cpu.apply(input, shape), actual = gpu.apply(input, shape);
                    check(std::equal(expected.values.begin(), expected.values.end(),
                                     actual.values.begin(),
                                     [](float a, float b)
                                     {
                                         return std::bit_cast<std::uint32_t>(a) ==
                                                std::bit_cast<std::uint32_t>(b);
                                     }),
                          "specialized convolution changed float bits");
                }
            }
            // Signed zero must also agree when every accumulation starts at zero.
            const SignalShape shape{channels, 97, 2};
            std::vector<float> zeros(shape.elements(), -0.0f);
            auto expected = cpu.apply(zeros, shape), actual = gpu.apply(zeros, shape);
            check(std::equal(expected.values.begin(), expected.values.end(), actual.values.begin(),
                             [](float a, float b)
                             {
                                 return std::bit_cast<std::uint32_t>(a) ==
                                        std::bit_cast<std::uint32_t>(b);
                             }),
                  "specialized convolution changed signed-zero bits");
            // Mixed magnitudes make rounding and accumulation order observable.
            for (std::size_t i = 0; i < zeros.size(); ++i)
            {
                const float scale = i % 3 == 0 ? 1.0e8f : i % 3 == 1 ? 1.0f : 1.0e-8f;
                zeros[i] = float(int((i * 73 + i / 7) % 257) - 128) * scale;
            }
            // The scalar Conv1d uses separate multiply/add. Use an explicit FMA
            // reference here to match the DAC shader's required rounding behavior.
            for (std::size_t group = 0; group < shape.groups; ++group)
            {
                for (std::size_t frame = 0; frame < shape.frames; ++frame)
                {
                    for (std::size_t out = 0; out < channels; ++out)
                    {
                        float sum = 0;
                        for (std::size_t tap = 0; tap < 7; ++tap)
                        {
                            const auto position = frame + tap * dilation;
                            if (position < 3 * dilation || position - 3 * dilation >= shape.frames)
                            {
                                continue;
                            }
                            const auto source =
                                (group * shape.frames + position - 3 * dilation) * channels;
                            for (std::size_t in = 0; in < channels; ++in)
                            {
                                sum = std::fma(zeros[source + in],
                                               weights[out * channels * 7 + tap * channels + in],
                                               sum);
                            }
                        }
                        expected.values[(group * shape.frames + frame) * channels + out] =
                            sum + bias[out];
                    }
                }
            }
            actual = gpu.apply(zeros, shape);
            check(std::equal(expected.values.begin(), expected.values.end(), actual.values.begin(),
                             [](float a, float b)
                             {
                                 return std::bit_cast<std::uint32_t>(a) ==
                                        std::bit_cast<std::uint32_t>(b);
                             }),
                  "specialized convolution changed accumulation order");
        }
    }
}

// Explicit ordered-FMA reference. Transposed weights store output channels
// contiguously, so independent outputs can vectorize without regrouping sums.
void transposed_convolutions()
{
    for (std::size_t channels : {192u, 384u, 768u, 1536u})
    {
        const std::size_t out = channels / 2, stride = channels == 192   ? 2
                                                       : channels == 384 ? 4
                                                                         : 8;
        for (unsigned fallback : {0u, 1u, 2u})
        {
            const std::size_t padding = stride / 2 + (fallback == 1 ? 1 : 0),
                              dilation = fallback == 2 ? 3 : 1;
            const Conv1dConfig config{
                channels, out, 2 * stride, stride, padding, dilation, ConvolutionKind::transposed};
            std::vector<float> weights(channels * out * config.kernel), bias(out);
            for (std::size_t i = 0; i < weights.size(); ++i)
            {
                weights[i] = float(int((i * 17 + i / 13) % 127) - 63) / 1024;
            }
            for (std::size_t i = 0; i < out; ++i)
            {
                bias[i] = float(int(i % 17) - 8) / 128;
            }
            auto gpu = MetalConvolution::prepare(config, weights, bias);
            const std::size_t tile = channels == 384 ? 64 : 32;
            const std::vector<std::size_t> lengths =
                fallback ? std::vector<std::size_t>{3}
                         : std::vector<std::size_t>{1, tile - 1, tile, tile + 1, 2 * tile + 3};
            for (std::size_t frames : lengths)
            {
                for (std::size_t groups : {1u, 2u})
                {
                    const SignalShape shape{channels, frames, groups};
                    std::vector<float> input(shape.elements());
                    for (std::size_t i = 0; i < input.size(); ++i)
                    {
                        const float scale = i % 3 == 0 ? 1.0e8f : i % 3 == 1 ? 1.0f : 1.0e-8f;
                        input[i] =
                            i % 19 == 0 ? -0.0f : float(int((i * 73 + i / 7) % 257) - 128) * scale;
                    }
                    auto actual = gpu.apply(input, shape);
                    std::vector<float> expected(actual.values.size(), 0.0f);
                    for (std::size_t group = 0; group < groups; ++group)
                    {
                        for (std::size_t frame = 0; frame < actual.shape.frames; ++frame)
                        {
                            float* sums =
                                expected.data() + (group * actual.shape.frames + frame) * out;
                            for (std::size_t tap = 0; tap < config.kernel; ++tap)
                            {
                                const auto position = frame + padding, offset = tap * dilation;
                                if (position < offset || (position - offset) % stride)
                                {
                                    continue;
                                }
                                const auto source = (position - offset) / stride;
                                if (source >= frames)
                                {
                                    continue;
                                }
                                for (std::size_t in = 0; in < channels; ++in)
                                {
                                    const float value =
                                        input[(group * frames + source) * channels + in];
                                    for (std::size_t channel = 0; channel < out; ++channel)
                                    {
                                        sums[channel] = std::fma(
                                            value,
                                            weights[(in * config.kernel + tap) * out + channel],
                                            sums[channel]);
                                    }
                                }
                            }
                            for (std::size_t channel = 0; channel < out; ++channel)
                            {
                                sums[channel] += bias[channel];
                            }
                        }
                    }
                    check(std::equal(expected.begin(), expected.end(), actual.values.begin(),
                                     [](float x, float y)
                                     {
                                         return std::bit_cast<std::uint32_t>(x) ==
                                                std::bit_cast<std::uint32_t>(y);
                                     }),
                          "transposed convolution changed ordered-FMA bits");
                }
            }
        }
    }
}

void decode_streams(const TransformerModel& model)
{
    auto cpu = apple_arm::CpuTransformer::from_model(model);
    auto gpu = MetalTransformer::from_model(model);
    TransformerPredictor encoder(cpu);
    for (const auto shape : {std::array<unsigned, 3>{197, 3, 1}, {197, 3, 2}, {3074, 3, 1}})
    {
        Header h;
        h.method = 1;
        h.frames = shape[0];
        h.batch = static_cast<std::uint16_t>(shape[1]);
        h.groups = static_cast<std::uint8_t>(shape[2]);
        h.model_channels = 2;
        h.codebooks = 1;
        std::vector<std::uint16_t> tokens(h.symbol_count());
        for (std::size_t i = 0; i < tokens.size(); ++i)
        {
            tokens[i] = static_cast<std::uint16_t>((i * 73 + 19) % 1024);
        }
        const auto bytes = pack_normal(h, tokens, encoder);
        check(gpu.decode(bytes) == tokens, "resident range decode changed tokens");
        if (h.frames == 197 && h.groups == 1)
        {
            Limits limits;
            // The old unpadded estimate plus 96 lookahead columns fits exactly;
            // the padded temporal allocation requires falling back instead.
            limits.cache_bytes = 197 * 12 * 512 * 4 + 3 * 3 * 512 * 4 + 96 * 1024 * 8;
            auto bounded_backend = MetalTransformer::from_model(model, limits);
            check(bounded_backend.encode(h, tokens) == bytes,
                  "padded cache-budget fallback changed stream");
            check(bounded_backend.decode(bytes) == tokens,
                  "resident decode failed tight cache budget");
        }
        std::cout << "decoded frames=" << h.frames << " groups=" << unsigned(h.groups)
                  << " passed\n";
    }
}

void resident_streams(const TransformerModel& model, unsigned channels, unsigned maximum_books)
{
    auto cpu = apple_arm::CpuTransformer::from_model(model);
    auto gpu = MetalTransformer::from_model(model);
    TransformerPredictor encoder(cpu);
    for (const auto shape : {std::array<unsigned, 4>{0, 3, 1, 1},
                             {1, 24, 1, 1},
                             {197, 24, 1, maximum_books},
                             {71, 3, 2, 3},
                             {193, 97, 1, 1}})
    {
        Header h;
        h.method = 1;
        h.frames = shape[0];
        h.batch = static_cast<std::uint16_t>(shape[1]);
        h.groups = static_cast<std::uint8_t>(shape[2]);
        h.model_channels = static_cast<std::uint8_t>(channels);
        h.codebooks = static_cast<std::uint8_t>(shape[3]);
        std::vector<std::uint16_t> tokens(h.symbol_count());
        for (std::size_t i = 0; i < tokens.size(); ++i)
        {
            tokens[i] = static_cast<std::uint16_t>((i * 73 + 19) % 1024);
        }
        const auto bytes = pack_normal(h, tokens, encoder);
        std::vector<EntropyPosition> observed, expected;
        visit_entropy_positions(h,
                                [&](const auto& p)
                                {
                                    expected.push_back(p);
                                });
        auto result = gpu.decode(bytes, {[&](const auto& p)
                                         {
                                             observed.push_back(p);
                                         },
                                         {}});
        check(result == tokens, "resident decode differs from CPU stream");
        check(observed.size() == expected.size(), "resident progress lost symbols");
        for (std::size_t i = 0; i < observed.size(); ++i)
        {
            check(observed[i].token_offset == expected[i].token_offset &&
                      observed[i].step == expected[i].step &&
                      observed[i].codebook == expected[i].codebook,
                  "resident progress reordered symbols");
        }
        auto corrupted = bytes;
        corrupted.back() ^= 1;
        bool rejected = false;
        try
        {
            gpu.decode(corrupted);
        }
        catch (const Error&)
        {
            rejected = true;
        }
        check(rejected, "resident decode accepted invalid CRC");
        corrupted = bytes;
        corrupted[14] = corrupted[15] = corrupted[16] = 255;
        rejected = false;
        try
        {
            gpu.decode(corrupted);
        }
        catch (const Error&)
        {
            rejected = true;
        }
        check(rejected, "resident decode accepted invalid initial interval");
        if (h.frames == 71)
        {
            bool interrupted = false;
            try
            {
                gpu.decode(bytes, {[&](const auto&)
                                   {
                                       throw Error("observer stopped decode");
                                   },
                                   {}});
            }
            catch (const Error&)
            {
                interrupted = true;
            }
            check(interrupted, "Observer did not interrupt queued decode");
            check(gpu.decode(bytes) == tokens,
                  "Session could not recover after observer exception");
        }
        std::cout << "resident channels=" << channels << " frames=" << h.frames
                  << " batch=" << h.batch << " passed\n";
    }
}

void dac_shared_chunks()
{
    metal::DacGraph graph;
    for (std::size_t frames : {4u, 9u, 3u, 9u})
    {
        graph.begin();
        const SignalShape shape{1, frames, 1};
        std::vector<float> first(frames), second(frames);
        for (std::size_t i = 0; i < frames; ++i)
        {
            first[i] = float(frames + i);
            second[i] = float(frames * 2 + i);
        }
        const auto a = graph.upload(first, shape, {});
        const auto doubled = graph.residual(a, a, false, {});
        graph.release(a);
        // A later CPU upload must not overwrite an earlier GPU input, even
        // when that input has been released by the recorder.
        const auto b = graph.upload(second, shape, {});
        const auto result = graph.residual(doubled, b, false, {});
        graph.release(doubled);
        graph.release(b);
        graph.finish();
        const auto actual = result.buffer.mapped<float>();
        for (std::size_t i = 0; i < frames; ++i)
        {
            check(actual[i] == 2 * first[i] + second[i],
                  "DAC chunk reuse aliased CPU uploads with pending GPU inputs");
        }
        graph.release(result);
    }
}

void rvq_lookup_validation()
{
    metal::DacGraph graph;
    const std::vector<float> vectors(8192), bias(1024);
    const auto v = metal::Buffer::copy(vectors.data(), vectors.size() * sizeof(float));
    const auto b = metal::Buffer::copy(bias.data(), bias.size() * sizeof(float));
    const metal::RvqBuffers weights{v, v, b, v};
    const std::uint16_t invalid = 1024;
    const auto tokens = metal::Buffer::copy(&invalid, sizeof invalid);
    (void)graph.lookup(tokens, weights, {});
    bool rejected = false;
    try
    {
        graph.finish();
    }
    catch (const Error&)
    {
        rejected = true;
    }
    check(rejected, "RVQ lookup accepted an invalid GPU token");
    graph.begin();
    const std::uint16_t valid = 0;
    tokens.write(&valid, sizeof valid);
    const auto output = graph.lookup(tokens, weights, {});
    graph.finish();
    const auto actual = output.buffer.mapped<float>().first(1024);
    check(std::all_of(actual.begin(), actual.end(),
                      [](float x)
                      {
                          return x == 0;
                      }),
          "RVQ lookup failed to recover after a rejected chunk");
}

void dac_residency(const ModelDirectory& directory)
{
    auto analysis =
        metal::AudioDac::from_model(DacAnalysisModel(directory, ModelVariant::stereo), {});
    auto synthesis =
        metal::AudioDac::from_model(DacSynthesisModel(directory, ModelVariant::stereo), {});
    SignalShape shape{2, std::size_t{512} * 3, 1};
    std::vector<float> pcm(shape.elements());
    for (std::size_t i = 0; i < pcm.size(); ++i)
    {
        pcm[i] = float(int(i % 31) - 15) * 0.01f;
    }
    const auto cpu_analysis =
        apple_arm::DacAnalysis::from_model(DacAnalysisModel(directory, ModelVariant::stereo));
    const auto cpu_synthesis =
        apple_arm::DacSynthesis::from_model(DacSynthesisModel(directory, ModelVariant::stereo));
    const auto observed = cpu_analysis.apply(pcm, shape);
    const auto resident = analysis.apply(pcm, shape, {});
    // Accelerate and GPU convolution use distinct reductions. Full pipelines
    // agree within FP32 tolerance; individual shader tests retain bitwise checks.
    const auto close = [](float a, float b)
    {
        return std::abs(a - b) <= 1e-4f * (1.0f + std::max(std::abs(a), std::abs(b)));
    };
    check(std::equal(observed.values.begin(), observed.values.end(), resident.values.begin(),
                     resident.values.end(), close),
          "resident analysis differs from CPU pipeline");
    const auto expected_pcm = cpu_synthesis.apply(resident.values, resident.shape);
    const auto actual_pcm = synthesis.apply(resident.values, resident.shape, {});
    check(std::equal(expected_pcm.values.begin(), expected_pcm.values.end(),
                     actual_pcm.values.begin(), actual_pcm.values.end(), close),
          "resident synthesis differs from CPU pipeline");
    // Compare shared graph handoffs against the standalone operator boundary.
    Header header;
    header.model_channels = 2;
    header.frames = 3;
    header.codebooks = 2;
    const auto encoder = metal::AudioRvqEncoder::from_model(
        RvqEncoderModel(directory, ModelVariant::stereo, RvqInputLayout::output_major, 2), {});
    const auto decoder =
        metal::AudioRvqDecoder::from_model(RvqDecoderModel(directory, ModelVariant::stereo, 2), {});
    const auto expected_tokens = encoder.encode(header, resident.values, {});
    metal::DacGraph combined;
    const auto latent = analysis.record(combined, combined.upload(pcm, shape, {}), {});
    const auto selected = encoder.record(combined, header, latent, {});
    combined.finish();
    for (std::size_t book = 0; book < header.codebooks; ++book)
    {
        const auto values = selected[book].mapped<std::uint16_t>();
        for (std::size_t frame = 0; frame < header.frames; ++frame)
        {
            check(values[frame] == expected_tokens[frame * header.codebooks + book],
                  "Shared analysis/RVQ handoff changed tokens");
        }
    }
    const auto decoded_latent = decoder.decode(header, expected_tokens, {});
    const auto expected_decoded = synthesis.apply(decoded_latent, {1024, 3, 1}, {});
    combined.begin();
    const auto shared_latent = decoder.record(combined, header, expected_tokens, {});
    const auto shared_pcm = synthesis.record(combined, shared_latent, {});
    combined.finish();
    const auto actual_decoded = shared_pcm.buffer.mapped<float>();
    check(std::equal(expected_decoded.values.begin(), expected_decoded.values.end(),
                     actual_decoded.begin()),
          "Shared RVQ/synthesis handoff changed PCM");
    // An intermediate overflow must remain observable even if later reuse
    // overwrites that buffer or an activation would otherwise hide the value.
    metal::DacGraph graph;
    Limits limits;
    const std::array<float, 1> huge{std::numeric_limits<float>::max()}, two{2}, zero{0};
    auto conv = MetalConvolution::prepare({1, 1, 1, 1, 0, 1}, two, zero);
    auto input = graph.upload(huge, {1, 1, 1}, limits);
    auto output = graph.convolution(conv, input, limits);
    bool rejected = false;
    try
    {
        graph.download(output, limits);
    }
    catch (const Error&)
    {
        rejected = true;
    }
    check(rejected, "resident DAC accepted nonfinite intermediate");
    for (bool snake_overflow : {false, true})
    {
        metal::DacGraph checked;
        auto source = checked.upload(huge, {1, 1, 1}, limits);
        auto invalid =
            snake_overflow
                ? checked.snake(source, metal::Buffer::copy(huge.data(), sizeof(float)), limits)
                : checked.residual(source, source, false, limits);
        checked.release(invalid);
        // Download an unrelated finite tensor: the producer must remember the
        // invalid intermediate even when its result is never read by the CPU.
        auto finite = checked.upload(zero, {1, 1, 1}, limits);
        rejected = false;
        try
        {
            checked.download(finite, limits);
        }
        catch (const Error&)
        {
            rejected = true;
        }
        check(rejected, "fused DAC check lost a nonfinite intermediate");
    }
}

int main(int argc, char** argv)
{
    try
    {
        check(argc == 2 || (argc == 3 && std::string_view(argv[2]) == "--kernels-only"),
              "expected model directory and optional --kernels-only");
        const std::filesystem::path root = argv[1];
        const ModelDirectory directory(root);
        const TransformerModel model(directory, ModelVariant::stereo, 12);
        prediction_slots(model);
        projections(model);
        projection_format_edges(model);
        normalized_projections(model);
        if (argc == 3)
        {
            std::cout << "Metal projection checks passed\n";
            return 0;
        }
        rvq_search_edges();
        hot_convolutions();
        transposed_convolutions();
        dac_shared_chunks();
        rvq_lookup_validation();
        dac_residency(directory);
        decode_streams(model);
        resident_streams(model, 2, 12);
        const TransformerModel mono(directory, ModelVariant::mono, 9);
        resident_streams(mono, 1, 9);
        known(model);
        std::cout << "optimized Metal exact checks passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
