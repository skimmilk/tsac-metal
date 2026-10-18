#include "common/probability.hpp"
#include "format/categorical.hpp"
#include "metal/prediction.hpp"
#include "metal/probability/parameters.h"

#include <algorithm>
#include <iostream>

using namespace tsac;

namespace
{
void check(bool value, const char* message)
{
    if (!value)
    {
        throw Error(message);
    }
}

void invalid_preparation()
{
    for (std::size_t width : {512u, 2048u})
    {
        std::vector<std::int8_t> weights(width * 33, 1);
        std::vector<float> scales(width / 256 * 33, 1), bias(33, 0);
        auto linear = MetalLinear::prepare(width, 33, weights, scales, bias);
        std::vector<std::uint16_t> input(width, 0);
        input.back() = 0x7f80; // +infinity must survive fusion as an error.
        std::vector<std::uint16_t> gain(512, probability_bf16(1)), offset(512, 0);
        auto g = metal::Buffer::copy(gain.data(), 1024),
             b = metal::Buffer::copy(offset.data(), 1024);
        for (int mode = 0; mode < (width == 512 ? 3 : 2); ++mode)
        {
            metal::Prediction graph;
            graph.fuse_projections(true);
            graph.begin();
            auto source = graph.upload(input);
            bool rejected = false;
            try
            {
                if (mode == 2)
                {
                    graph.norm_linear(linear, source, g, b);
                }
                else
                {
                    graph.linear(linear, source, mode == 1);
                }
                graph.finish();
            }
            catch (const Error& error)
            {
                rejected =
                    std::string(error.what()).find("nonfinite Metal Transformer activation") !=
                    std::string::npos;
            }
            check(rejected, "fused preparation lost a nonfinite activation");
        }
    }
}

// Capture CPU categorical decisions independently of Transformer inference.
// Exercise all tree depths, zero/underflowed masses, uneven lanes, groups,
// byte renormalization and state carried across command-buffer boundaries.
void decisions(float amplitude)
{
    Header h;
    h.method = 1;
    h.frames = 5;
    h.batch = 3;
    h.groups = 2;
    h.codebooks = 3;

    struct Round
    {
        std::size_t step, book, columns;
        std::vector<std::uint16_t> logits;
    };

    std::vector<Round> rounds;
    std::vector<std::uint16_t> expected(h.symbol_count()), actual(h.symbol_count());
    RangeEncoder encoder;
    for (std::size_t step = 0; step < 2; ++step)
    {
        const std::size_t lanes = step == 0 ? 3 : 2;
        for (std::size_t book = 0; book < h.codebooks; ++book)
        {
            Round round{step, book, lanes * h.groups, {}};
            round.logits.resize(round.columns * 1024);
            for (std::size_t c = 0; c < round.columns; ++c)
            {
                auto logits = std::span(round.logits).subspan(c * 1024, 1024);
                for (std::size_t i = 0; i < logits.size(); ++i)
                {
                    logits[i] = probability_bf16(
                        float(int((i * 73 + c * 19 + step + book) % 257) - 128) * amplitude);
                }
                auto bits = probability_softmax_bf16(logits);
                std::vector<float> probabilities(bits.size());
                std::transform(bits.begin(), bits.end(), probabilities.begin(), probability_float);
                const auto symbol =
                    static_cast<std::uint16_t>((c * 137 + book * 73 + step * 19) % 1024);
                encode_categorical(encoder, probabilities, symbol);
                const auto lane = c / h.groups, group = c % h.groups;
                const auto frame = lane + std::min<std::size_t>(lane, 2) + step;
                expected[(frame * h.groups + group) * h.codebooks + book] = symbol;
            }
            rounds.push_back(std::move(round));
        }
    }
    auto payload = encoder.finish();
    auto compressed = metal::Buffer::copy(payload.data(), payload.size());
    std::uint32_t code = 0;
    for (std::size_t i = 0; i < 3; ++i)
    {
        code = (code << 8) | (i < payload.size() ? payload[i] : 0);
    }
    const tsac_metal_abi::RangeState initial{
        0xff0000, code, static_cast<std::uint32_t>(std::min<std::size_t>(3, payload.size())), 0};
    auto state = metal::Buffer::copy(&initial, sizeof initial);
    auto tokens = metal::Buffer::copy(actual.data(), actual.size() * 2);
    metal::Prediction graph;
    for (const auto& round : rounds)
    {
        graph.begin();
        auto logits = graph.upload(round.logits);
        auto probabilities = graph.temporary(round.columns * 2048);
        auto tree = graph.temporary(round.columns * 4096);
        graph.softmax_decode(logits, probabilities, tree);
        graph.range_decode(probabilities, tree, compressed, state, tokens, h, round.step,
                           round.book, round.columns);
        graph.finish();
    }
    tokens.read(actual.data(), actual.size() * 2);
    check(actual == expected, "GPU categorical/range decisions differ from CPU");
}
} // namespace

int main()
{
    try
    {
        for (float amplitude : {0.0f, 0.03125f, 1.0f, 40.0f})
        {
            decisions(amplitude);
        }
        invalid_preparation();
        std::cout << "Metal categorical/range decisions passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
