#include "tsac/metal_analyzer.hpp"

#include "dac_preparation_fixture.hpp"
#include "metal/audio_dac.hpp"
#include "metal/transformer.hpp"
#include "metal/weight_preparation.hpp"
#include "models/contract.hpp"
#include "weights/normalization.hpp"
#include "weights/preparation.hpp"

#include <bit>
#include <chrono>
#include <iostream>
#include <sys/resource.h>

using namespace tsac;

namespace
{
void check(bool condition, const char* message)
{
    if (!condition)
    {
        throw Error(message);
    }
}

bool same(float a, float b)
{
    return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}

void dac(const ConvolutionWeights& weights, bool transposed)
{
    const auto axes = weights.values.shape().axes;
    const auto fan_in = axes[0] * axes[1];
    const auto expanded = weights::expand(weights.values);
    const auto folded = fold_dac_weight_norm(expanded, weights.gain.values(), fan_in);
    std::vector<float> actual(folded.size());
    metal::prepare_dac_weights(weights, transposed, actual, {});
    // Compare every prepared weight with the original full-tensor path, including
    // signed zeros, ordinary transposes, and transposed-convolution source axes.
    for (std::size_t c = 0; c < axes[2]; ++c)
    {
        for (std::size_t tap = 0; tap < axes[1]; ++tap)
        {
            for (std::size_t i = 0; i < axes[0]; ++i)
            {
                const auto source = i + axes[0] * (tap + axes[1] * c);
                const auto dest = transposed ? i + axes[0] * (c + axes[2] * tap)
                                             : c + axes[2] * (i + axes[0] * tap);
                check(same(actual[dest], folded[source]), "DAC prepared weight bits changed");
            }
        }
    }
    // A range crossing both a BF8 block and an outer scale plane must expand
    // identically. Kernel widths such as seven exercise padded scale planes.
    const auto first = axes[0];
    const auto size = std::min(expanded.size() - first, axes[0] * (axes[1] + 3));
    std::vector<float> range(size);
    weights::expand_into(weights.values, first, range);
    for (std::size_t i = 0; i < size; ++i)
    {
        check(same(range[i], expanded[first + i]), "partial source expansion changed bits");
    }
    Limits bounded;
    bounded.tensor_elements = actual.size() - 1;
    bool rejected = false;
    try
    {
        metal::prepare_dac_weights(weights, transposed, actual, bounded);
    }
    catch (const Error&)
    {
        rejected = true;
    }
    check(rejected, "bounded preparation ignored tensor limit");
}

template <class Model> void dac_model(const Model& model, bool synthesis)
{
    dac(model.entry_convolution(), false);
    for (unsigned stage = 0; stage < 4; ++stage)
    {
        dac(model.resample_convolution(stage), synthesis);
        for (unsigned unit = 0; unit < 3; ++unit)
        {
            dac(model.residual_convolution(stage, unit, ResidualConvolution::dilated), false);
            dac(model.residual_convolution(stage, unit, ResidualConvolution::pointwise), false);
        }
    }
    dac(model.exit_convolution(), false);
}

void linear(std::span<const std::int8_t> source, std::size_t input, std::size_t output,
            std::size_t rows)
{
    const auto padded = (output + rows - 1) / rows * rows;
    // Nonzero initialization exposes missed writes as well as missing tail padding.
    std::vector<std::int8_t> actual(input * padded, 93);
    metal::pack_linear_weights(source, input, output, rows, actual);
    for (std::size_t tile = 0; tile < padded / rows; ++tile)
    {
        for (std::size_t i = 0; i < input; ++i)
        {
            for (std::size_t row = 0; row < rows; ++row)
            {
                const auto out = tile * rows + row;
                const auto expected = out < output ? source[out * input + i] : std::int8_t{0};
                check(actual[(tile * input + i / 4 * 4) * rows + row * 4 + i % 4] == expected,
                      "Metal linear packing changed weight or padding bytes");
            }
        }
    }
}

void linear(const LinearWeights& weights)
{
    const auto axes = weights.values.shape().axes;
    linear(weights.values.values(), axes[0], axes[1], axes[0] == 512 && axes[1] >= 1024 ? 8 : 2);
}

void projections(const TransformerModel& model)
{
    for (auto stack : {TransformerStack::temporal, TransformerStack::codebook})
    {
        for (unsigned layer = 0; layer < model_contract::layers(stack); ++layer)
        {
            linear(model.attention_qkv(stack, layer));
            linear(model.attention_output(stack, layer));
            linear(model.mlp_input(stack, layer));
            linear(model.mlp_output(stack, layer));
        }
    }
    linear(model.temporal_projection());
    linear(model.codebook_projection());
    linear(model.output_projection());
}

void preparation_failures(const ModelDirectory& models)
{
    test::dac_preparation_fixtures(
        models,
        [&](const ConvolutionWeights& weights, bool valid)
        {
            if (valid)
            {
                dac(weights, false);
                dac(weights, true);
                return;
            }
            bool rejected = false;
            try
            {
                std::vector<float> output(weights.values.shape().elements());
                metal::prepare_dac_weights(weights, false, output, {});
            }
            catch (const Error&)
            {
                rejected = true;
            }
            check(rejected, "parallel preparation lost a source or normalization error");
        });
}

template <class F> void measured(std::string_view role, int trial, F action)
{
    const auto start = std::chrono::steady_clock::now();
    action();
    const auto seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    std::cout << role << " trial=" << trial + 1 << " seconds=" << seconds
              << " peak_bytes=" << usage.ru_maxrss << std::endl;
}

void benchmark(const ModelDirectory& directory, std::string_view role, int trials)
{
    check(trials > 0, "trials must be positive");
    check(role == "analysis" || role == "synthesis" || role == "probability" || role == "repeated",
          "unknown role");
    if (role == "repeated")
    {
        MetalAnalyzer analyzer(directory);
        const std::vector<float> pcm(4410 * 2, 0.125f);
        EncodeOptions options;
        options.codebooks = 3;
        options.entropy_batch = 3;
        Bytes stream;
        std::vector<float> samples;
        for (int trial = 0; trial < trials; ++trial)
        {
            measured("encode", trial,
                     [&]
                     {
                         auto next = analyzer.encode(interleaved_pcm(pcm, 2), options);
                         check(!trial || next == stream, "repeated encode changed bytes");
                         stream = std::move(next);
                     });
        }
        for (int trial = 0; trial < trials; ++trial)
        {
            measured("decode", trial,
                     [&]
                     {
                         auto next = analyzer.decode(stream);
                         check(!trial || next.samples == samples, "repeated decode changed PCM");
                         samples = std::move(next.samples);
                     });
        }
        return;
    }
    for (int trial = 0; trial < trials; ++trial)
    {
        measured(role, trial,
                 [&]
                 {
                     if (role == "analysis")
                     {
                         auto model = metal::AudioDac::from_model(
                             DacAnalysisModel(directory, ModelVariant::stereo), {});
                     }
                     else if (role == "synthesis")
                     {
                         auto model = metal::AudioDac::from_model(
                             DacSynthesisModel(directory, ModelVariant::stereo), {});
                     }
                     else
                     {
                         auto model = MetalTransformer::from_model(
                             TransformerModel(directory, ModelVariant::stereo, 12));
                     }
                 });
    }
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        check(argc == 2 || argc == 4, "usage: metal-preparation MODEL_DIRECTORY [ROLE TRIALS]");
        const ModelDirectory directory(argv[1]);
        if (argc == 4)
        {
            benchmark(directory, argv[2], std::stoi(argv[3]));
            return 0;
        }
        for (auto variant : {ModelVariant::mono, ModelVariant::stereo})
        {
            dac_model(DacAnalysisModel(directory, variant), false);
            dac_model(DacSynthesisModel(directory, variant), true);
            projections(TransformerModel(directory, variant, model_contract::codebooks(variant)));
        }
        preparation_failures(directory);
        for (std::size_t rows : {2u, 8u})
        {
            for (std::size_t outputs : {1u, 2u, 7u, 8u, 9u, 17u, 1025u})
            {
                std::vector<std::int8_t> source(256 * outputs);
                for (std::size_t i = 0; i < source.size(); ++i)
                {
                    source[i] =
                        std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(i * 73 + i / 17));
                }
                linear(source, 256, outputs, rows);
            }
        }
        std::cout << "Metal weight preparation exact checks passed\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
