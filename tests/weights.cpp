#include "apple-arm/pipeline.hpp"
#include "common/analysis.hpp"
#include "common/packed_linear.hpp"
#include "common/probability.hpp"
#include "common/rvq.hpp"
#include "common/synthesis.hpp"
#include "models/contract.hpp"
#include "scalar/pipeline.hpp"
#include "weights/source_model.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

namespace
{
using namespace tsac;
using tsac::packing::SourceModel;

void check(bool condition, const char* message)
{
    if (!condition)
    {
        throw Error(message);
    }
}

template <class T> void same(std::span<const T> a, std::span<const T> b, const char* message)
{
    check(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0, message);
}

void linear(const LinearWeights& ref, const SourceModel& original, const std::string& name)
{
    auto actual = apple_arm::PackedLinear::from_weights(ref);
    const auto input_width = ref.values.shape().axes[0], output_width = ref.values.shape().axes[1];
    std::vector<std::uint16_t> input(input_width);
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] = probability_bf16(static_cast<float>(static_cast<int>(i % 37) - 18) * 0.03125f);
    }
    // Independent row-major integer-dot reference from the original source.
    std::vector<float> expanded(input.size());
    std::transform(input.begin(), input.end(), expanded.begin(), probability_float);
    const auto x = apple_arm::bf8l(expanded);
    const auto payload = original.tensor_payload(name);
    const auto& descriptor = original.tensor(name);
    const auto blocks = input_width / 256;
    const auto padded = (output_width + 7) / 8 * 8;
    std::vector<float> bias(output_width, 0);
    if (name.ends_with("/w"))
    {
        bias = original.dequantize(name.substr(0, name.size() - 1) + "b");
    }
    std::vector<std::uint16_t> expected(output_width);
    for (std::size_t out = 0; out < output_width; ++out)
    {
        float sum = 0;
        for (std::size_t block = 0; block < blocks; ++block)
        {
            std::int32_t dot = 0;
            for (std::size_t i = 0; i < 256; ++i)
            {
                dot += static_cast<std::int8_t>(payload[out * input_width + block * 256 + i]) *
                       x.values[block * 256 + i];
            }
            const auto offset = descriptor.data_plane_bytes + (block * padded + out) * 2;
            const auto w_scale = probability_float(static_cast<std::uint16_t>(
                unsigned(payload[offset]) | (unsigned(payload[offset + 1]) << 8)));
            const auto scale = w_scale * probability_float(x.scales[block]);
            sum = std::fma(static_cast<float>(dot), scale, sum);
        }
        expected[out] = probability_bf16(sum + bias[out]);
    }
    const auto a = actual.apply(input);
    const auto& b = expected;
    same<std::uint16_t>(a, b, "packed NEON projection differs");
    apple_arm::PackedLinear::Workspace workspace;
    auto aliased = input;
    actual.apply_into(aliased, aliased, workspace);
    same<std::uint16_t>(aliased, b, "aliased packed projection differs");
    // Reuse the same scratch for multiple columns and then a smaller input.
    auto columns = input;
    columns.insert(columns.end(), input.begin(), input.end());
    actual.apply_into(columns, aliased, workspace);
    same<std::uint16_t>(std::span(aliased).first(output_width), b, "first packed column differs");
    same<std::uint16_t>(std::span(aliased).subspan(output_width), b,
                        "second packed column differs");
    actual.apply_into(input, aliased, workspace);
    same<std::uint16_t>(aliased, b, "reused packed scratch differs");
    const auto reference = scalar::PackedLinear::from_weights(ref).apply(input);
    same<std::uint16_t>(reference, b, "packed scalar projection differs");
}

void transformer(const TransformerModel& packed, const SourceModel& original)
{
    for (auto stack : {TransformerStack::temporal, TransformerStack::codebook})
    {
        const std::string prefix = stack == TransformerStack::codebook ? "codebook_decoder." : "";
        for (unsigned layer = 0; layer < model_contract::layers(stack); ++layer)
        {
            const auto stem = prefix + "h" + std::to_string(layer) + "/";
            linear(packed.attention_qkv(stack, layer), original, stem + "attn/c_attn/w");
            linear(packed.attention_output(stack, layer), original, stem + "attn/c_proj/w");
            linear(packed.mlp_input(stack, layer), original, stem + "mlp/c_fc/w");
            linear(packed.mlp_output(stack, layer), original, stem + "mlp/c_proj/w");
        }
    }
    linear(packed.temporal_projection(), original, "cb_proj");
    linear(packed.codebook_projection(), original, "codebook_decoder.cb_proj");
    linear(packed.output_projection(), original, "proj_f/w");
}

// Test-only source-format oracle. It executes the source topology directly,
// without any runtime Model adapter or legacy inference constructor.
Signal source_dac(const SourceModel& source, Signal x, bool synthesis)
{
    const auto conv = [&](const std::string& name, const Signal& input, std::size_t stride = 1,
                          std::size_t dilation = 1, bool transposed = false)
    {
        const auto& dimensions = source.tensor(name + ".weight_v").dimensions;
        const auto first = dimensions[0], kernel = dimensions[1], last = dimensions[2];
        const Conv1dConfig config{transposed ? last : first,
                                  transposed ? first : last,
                                  kernel,
                                  stride,
                                  stride == 1 ? (kernel - 1) * dilation / 2 : (stride + 1) / 2,
                                  dilation,
                                  transposed ? ConvolutionKind::transposed
                                             : ConvolutionKind::ordinary};
        const auto weights =
            fold_dac_weight_norm(source.dequantize(name + ".weight_v"),
                                 source.dequantize(name + ".weight_g"), first * kernel);
        return apple_arm::Conv1d(config, weights, source.dequantize(name + ".bias"), {})
            .apply(input.values, input.shape);
    };
    const auto snake = [&](const std::string& name, const Signal& input)
    {
        return apple_arm::snake_dac(input.values, input.shape, source.dequantize(name + ".alpha"),
                                    {});
    };
    const std::string prefix = synthesis ? "decoder.model." : "encoder.block.";
    x = conv(prefix + "0", x);
    for (unsigned block = 0; block < 4; ++block)
    {
        const auto stem = prefix + std::to_string(block + 1) + ".block.";
        const auto stride = synthesis ? std::array<unsigned, 4>{8, 8, 4, 2}[block]
                                      : std::array<unsigned, 4>{2, 4, 8, 8}[block];
        if (synthesis)
        {
            x = snake(stem + "0", x);
            x = conv(stem + "1", x, stride, 1, true);
        }
        for (unsigned unit = 0; unit < 3; ++unit)
        {
            const auto name = stem + std::to_string(unit + (synthesis ? 2 : 0)) + ".block.";
            auto residual = snake(name + "0", x);
            residual = conv(name + "1", residual, 1, std::array<unsigned, 3>{1, 3, 9}[unit]);
            residual = snake(name + "2", residual);
            residual = conv(name + "3", residual);
            check(residual.values.size() == x.values.size(), "source residual shape mismatch");
            for (std::size_t i = 0; i < x.values.size(); ++i)
            {
                x.values[i] = x.values[i] + residual.values[i];
            }
        }
        if (!synthesis)
        {
            x = snake(stem + "3", x);
            x = conv(stem + "4", x, stride);
        }
    }
    x = snake(prefix + "5", x);
    return conv(prefix + "6", x);
}

void lifetime_and_audio(const SourceModel& dac, const ModelDirectory& directory,
                        ModelVariant variant)
{
    const auto channels = model_contract::channels(variant);
    std::vector<float> pcm(std::size_t{512} * channels);
    for (std::size_t i = 0; i < pcm.size(); ++i)
    {
        pcm[i] = static_cast<float>(static_cast<int>(i % 43) - 21) * 0.002f;
    }
    const SignalShape shape{channels, 512, 1};
    auto expected = source_dac(dac, {shape, pcm}, false);
    {
        DacAnalysisModel adapter(directory, variant);
        std::weak_ptr<const MappedRegion> source = adapter.entry_convolution().values.region();
        auto model = apple_arm::DacAnalysis::from_model(std::move(adapter));
        check(source.expired(), "DAC source mapping survived preparation");
        const auto actual = model.apply(pcm, shape);
        same<float>(actual.values, expected.values, "Accelerate analysis differs");
        std::vector<std::string> labels;
        const auto traced = model.apply(pcm, shape, {},
                                        [&](std::string_view label, const Signal&)
                                        {
                                            labels.emplace_back(label);
                                        });
        same<float>(traced.values, actual.values, "analysis observer changed output");
        check(labels.size() == 71 && labels.front() == "encoder.block.0" &&
                  labels[5] == "encoder.block.1.block.0" && labels.back() == "encoder.block.6",
              "analysis diagnostic labels changed");
    }
    std::vector<float> latent(1024);
    for (std::size_t i = 0; i < latent.size(); ++i)
    {
        latent[i] = static_cast<float>(i % 19) * 0.001f;
    }
    expected = source_dac(dac, {{1024, 1, 1}, latent}, true);
    {
        DacSynthesisModel adapter(directory, variant);
        std::weak_ptr<const MappedRegion> source = adapter.entry_convolution().values.region();
        auto model = apple_arm::DacSynthesis::from_model(std::move(adapter));
        check(source.expired(), "synthesis source mapping survived preparation");
        const auto actual = model.apply(latent, {1024, 1, 1});
        same<float>(actual.values, expected.values, "Accelerate synthesis differs");
        std::vector<std::string> labels;
        const auto traced = model.apply(latent, {1024, 1, 1}, {},
                                        [&](std::string_view label, const Signal&)
                                        {
                                            labels.emplace_back(label);
                                        });
        same<float>(traced.values, actual.values, "synthesis observer changed output");
        check(labels.size() == 71 && labels.front() == "decoder.model.0" &&
                  labels[7] == "decoder.model.1.block.2" && labels.back() == "decoder.model.6",
              "synthesis diagnostic labels changed");
    }
    {
        TransformerModel adapter(directory, variant, 2);
        std::weak_ptr<const MappedRegion> source =
            adapter.attention_qkv(TransformerStack::temporal, 0).values.region();
        auto model = apple_arm::CpuTransformer::from_model(std::move(adapter));
        check(source.expired(), "Transformer source mapping survived NEON preparation");
    }
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        check(argc == 3, "usage: tsac-weights-test PACKED_DIRECTORY ORIGINALS");
        for (unsigned channels : {1u, 2u})
        {
            const auto variant = channels == 1 ? "mono" : "stereo";
            auto dac = SourceModel::load(std::filesystem::path(argv[2]) /
                                         ("dac_" + std::string(variant) + "_q8.bin"));
            auto tf = SourceModel::load(std::filesystem::path(argv[2]) /
                                        ("tsac_" + std::string(variant) + "_q8.bin"));
            const ModelDirectory directory(argv[1]);
            const auto role = channels == 1 ? ModelVariant::mono : ModelVariant::stereo;
            const unsigned books = model_contract::codebooks(role);
            transformer(TransformerModel(directory, role, books), tf);
            {
                const TransformerModel prefix_model(directory, role, 2);
                const auto prefix = prefix_model.feedback_embeddings();
                check(prefix.size() * 2 == std::size_t{4} * 1024 * 1024,
                      "inactive embedding books were mapped");
            }
            {
                const TransformerModel transformer_model(directory, role, books);
                const auto embeddings = transformer_model.feedback_embeddings();
                const RvqDecoderModel quantizer(directory, role, books);
                for (unsigned b = 0; b < books; ++b)
                {
                    const auto prefix = "quantizer.quantizers." + std::to_string(b);
                    auto original =
                        RvqCodebook::from_dac_weights(dac.dequantize(prefix + ".codebook.weight"),
                                                      dac.dequantize(prefix + ".out_proj.weight_v"),
                                                      dac.dequantize(prefix + ".out_proj.weight_g"),
                                                      dac.dequantize(prefix + ".out_proj.bias"));
                    auto mapped = RvqCodebook::from_prepared(quantizer.codebook(b),
                                                             quantizer.output_projection(b));
                    std::array<float, 1024> a{}, expected{};
                    for (unsigned symbol = 0; symbol < 1024; ++symbol)
                    {
                        original.project(static_cast<std::uint16_t>(symbol), expected);
                        mapped.project(static_cast<std::uint16_t>(symbol), a);
                        same<float>(a, expected, "prepared RVQ projection differs");
                        for (std::size_t c = 0; c < 1024; ++c)
                        {
                            check(
                                embeddings.values()[(std::size_t{b} * 1024 + symbol) * 1024 + c] ==
                                    probability_bf16(expected[c]),
                                "feedback embedding differs");
                        }
                    }
                }
            }
            lifetime_and_audio(dac, directory, role);
        }
        std::cout << "Original/packed CPU parity and mapping lifetimes passed\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
