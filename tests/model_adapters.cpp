#include "models/dac_analysis_model.hpp"
#include "models/dac_synthesis_model.hpp"
#include "models/rvq_decoder_model.hpp"
#include "models/rvq_encoder_model.hpp"
#include "models/transformer_model.hpp"
#include "weights/normalization.hpp"
#include "weights/preparation.hpp"
#include "weights/source_model.hpp"

#include <bit>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <source_location>
#include <thread>
#include <unistd.h>

namespace
{
using namespace tsac;
using packing::SourceModel;

void check(bool ok, const char* message)
{
    if (!ok)
    {
        throw Error(message);
    }
}

template <class F> void rejects(F call)
{
    try
    {
        call();
    }
    catch (const Error&)
    {
        return;
    }
    throw Error("invalid adapter request was accepted");
}

template <class T>
void same(std::span<const T> a, std::span<const T> b,
          std::source_location location = std::source_location::current())
{
    if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size_bytes()) != 0)
    {
        throw Error("adapter payload differs at test line " + std::to_string(location.line()) +
                    " (sizes " + std::to_string(a.size()) + "," + std::to_string(b.size()) + ")");
    }
}

void floats(const TensorView<float>& tensor, const SourceModel& source, const std::string& name)
{
    const auto expected = source.dequantize(name);
    same<float>(tensor.values(), expected);
    const auto& dimensions = source.tensor(name).dimensions;
    check(tensor.shape().rank() == dimensions.size(), "tensor rank differs from contract");
    for (std::size_t i = 0; i < dimensions.size(); ++i)
    {
        check(tensor.shape().axes[i] == dimensions[i], "tensor shape differs from source");
    }
}

void convolution(const ConvolutionWeights& c, const SourceModel& source, const std::string& stem)
{
    same<std::uint8_t>(c.values.payload(), source.tensor_payload(stem + ".weight_v"));
    floats(c.gain, source, stem + ".weight_g");
    floats(c.bias, source, stem + ".bias");
    const auto& dimensions = source.tensor(stem + ".weight_v").dimensions;
    check(c.values.shape().axes ==
              std::array<std::size_t, 3>{dimensions[0], dimensions[1], dimensions[2]},
          "encoded convolution shape differs");
}

template <class Model> void dac(Model& model, const SourceModel& source, bool synthesis)
{
    const std::string prefix = synthesis ? "decoder.model." : "encoder.block.";
    convolution(model.entry_convolution(), source, prefix + "0");
    convolution(model.exit_convolution(), source, prefix + "6");
    floats(model.final_activation(), source, prefix + "5.alpha");
    for (unsigned stage = 0; stage < 4; ++stage)
    {
        const auto stem = prefix + std::to_string(stage + 1) + ".block.";
        convolution(model.resample_convolution(stage), source, stem + (synthesis ? "1" : "4"));
        floats(model.resample_activation(stage), source,
               stem + (synthesis ? "0.alpha" : "3.alpha"));
        for (unsigned unit = 0; unit < 3; ++unit)
        {
            const auto name = stem + std::to_string(unit + (synthesis ? 2 : 0)) + ".block.";
            convolution(model.residual_convolution(stage, unit, ResidualConvolution::dilated),
                        source, name + "1");
            convolution(model.residual_convolution(stage, unit, ResidualConvolution::pointwise),
                        source, name + "3");
            floats(model.residual_activation(stage, unit, ResidualActivation::before_dilated),
                   source, name + "0.alpha");
            floats(model.residual_activation(stage, unit, ResidualActivation::before_pointwise),
                   source, name + "2.alpha");
        }
    }
    rejects(
        [&]
        {
            model.resample_convolution(4);
        });
    rejects(
        [&]
        {
            model.residual_activation(0, 3, ResidualActivation::before_dilated);
        });
}

void rvq(const ModelDirectory& directory, ModelVariant variant, const SourceModel& source)
{
    RvqEncoderModel apple(directory, variant, RvqInputLayout::input_major, 2),
        metal(directory, variant, RvqInputLayout::output_major, 2);
    RvqDecoderModel decoder(directory, variant, 2);
    for (unsigned book = 0; book < 2; ++book)
    {
        const auto stem = "quantizer.quantizers." + std::to_string(book);
        const auto codebook = decoder.codebook(book), encoded_codebook = apple.codebook(book);
        same<float>(codebook.values(), source.dequantize(stem + ".codebook.weight"));
        check(codebook.region() == encoded_codebook.region(),
              "encode/decode duplicated shared RVQ mapping");
        const auto a = apple.input_projection(book), m = metal.input_projection(book);
        const auto expected =
            fold_dac_weight_norm(source.dequantize(stem + ".in_proj.weight_v"),
                                 source.dequantize(stem + ".in_proj.weight_g"), 1024);
        same<float>(a.values.values(), expected);
        for (std::size_t out = 0; out < 8; ++out)
        {
            for (std::size_t in = 0; in < 1024; ++in)
            {
                check(m.values.values()[out + in * 8] == expected[in + out * 1024],
                      "Metal RVQ layout differs");
            }
        }
        same<float>(a.bias.values(), source.dequantize(stem + ".in_proj.bias"));
        const auto projection = decoder.output_projection(book);
        const auto folded = fold_dac_weight_norm(source.dequantize(stem + ".out_proj.weight_v"),
                                                 source.dequantize(stem + ".out_proj.weight_g"), 8);
        same<float>(projection.values.values(), folded);
        same<float>(projection.bias.values(), source.dequantize(stem + ".out_proj.bias"));
        const auto search = apple.search_vectors(book);
        const auto raw = source.dequantize(stem + ".codebook.weight");
        for (std::size_t symbol = 0; symbol < 1024; ++symbol)
        {
            std::array<float, 8> normalized{};
            weights::normalize_search(std::span<const float, 8>(raw.data() + symbol * 8, 8),
                                      normalized);
            same<float>(search.values().subspan(symbol * 8, 8), normalized);
        }
    }
    rejects(
        [&]
        {
            decoder.codebook(2);
        });
    rejects(
        [&]
        {
            RvqDecoderModel invalid(directory, variant, 0);
        });
    rejects(
        [&]
        {
            // Intentionally exercise rejection of an unsupported layout.
            // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
            RvqEncoderModel invalid(directory, variant, static_cast<RvqInputLayout>(99), 2);
        });
}

#if TSAC_TEST_TRANSFORMER
std::uint16_t bf16(float value)
{
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return static_cast<std::uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

void bf16_constant(const TensorView<std::uint16_t>& view, const SourceModel& source,
                   const std::string& name)
{
    const auto expanded = source.dequantize(name);
    std::vector<std::uint16_t> expected(view.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        expected[i] = bf16(expanded[i]);
    }
    same<std::uint16_t>(view.values(), expected);
}

void linear(const LinearWeights& values, const SourceModel& source, const std::string& name)
{
    const auto& t = source.tensor(name);
    const auto payload = source.tensor_payload(name);
    const auto d = t.dimensions[0], o = t.dimensions[1], padded = (o + 7) / 8 * 8;
    same<std::uint8_t>(
        {reinterpret_cast<const std::uint8_t*>(values.values.data()), values.values.size()},
        payload.first(t.elements));
    check(values.values.shape() == TensorShape{{d, o}}, "linear shape differs");
    for (std::size_t block = 0; block < d / 256; ++block)
    {
        for (std::size_t out = 0; out < o; ++out)
        {
            const auto offset = t.data_plane_bytes + 2 * (block * padded + out);
            const auto half =
                std::uint32_t(payload[offset]) | (std::uint32_t(payload[offset + 1]) << 8);
            check(values.scales.values()[block * o + out] == std::bit_cast<float>(half << 16),
                  "linear scale differs");
        }
    }
    if (name.ends_with("/w"))
    {
        same<float>(values.bias.values(), source.dequantize(name.substr(0, name.size() - 1) + "b"));
    }
    else
    {
        for (float b : values.bias.values())
        {
            check(b == 0, "bias-free projection has bias");
        }
    }
}

void norm(const NormalizationWeights& n, const SourceModel& source, const std::string& stem)
{
    for (const auto& item : {std::pair{n.gain, stem + "g"}, std::pair{n.bias, stem + "b"}})
    {
        const auto payload = source.tensor_payload(item.second);
        same<std::uint8_t>(
            {reinterpret_cast<const std::uint8_t*>(item.first.data()), item.first.size() * 2},
            payload);
    }
}

void transformer(const ModelDirectory& directory, ModelVariant variant, const SourceModel& source,
                 const SourceModel& dac_source)
{
    TransformerModel model(directory, variant, 2);
    for (auto stack : {TransformerStack::temporal, TransformerStack::codebook})
    {
        const bool cb = stack == TransformerStack::codebook;
        const std::string prefix = cb ? "codebook_decoder." : "";
        for (unsigned layer = 0; layer < (cb ? 3u : 12u); ++layer)
        {
            const auto stem = prefix + "h" + std::to_string(layer) + "/";
            linear(model.attention_qkv(stack, layer), source, stem + "attn/c_attn/w");
            linear(model.attention_output(stack, layer), source, stem + "attn/c_proj/w");
            linear(model.mlp_input(stack, layer), source, stem + "mlp/c_fc/w");
            linear(model.mlp_output(stack, layer), source, stem + "mlp/c_proj/w");
            norm(model.attention_normalization(stack, layer), source, stem + "ln_1/");
            norm(model.mlp_normalization(stack, layer), source, stem + "ln_2/");
        }
        norm(model.final_normalization(stack), source, prefix + "ln_f/");
    }
    linear(model.temporal_projection(), source, "cb_proj");
    linear(model.codebook_projection(), source, "codebook_decoder.cb_proj");
    linear(model.output_projection(), source, "proj_f/w");
    const auto positions = model.positions(), rotations = model.rotations(),
               embeddings = model.feedback_embeddings();
    bf16_constant(positions, source, "codebook_decoder.wpe");
    bf16_constant(rotations, source, "rotary_embed");
    check(embeddings.region()->bytes().size() == std::size_t{4} * 1024 * 1024,
          "unused embeddings were mapped");
    for (unsigned book = 0; book < 2; ++book)
    {
        const auto stem = "quantizer.quantizers." + std::to_string(book);
        const auto vectors = dac_source.dequantize(stem + ".codebook.weight");
        const auto folded =
            fold_dac_weight_norm(dac_source.dequantize(stem + ".out_proj.weight_v"),
                                 dac_source.dequantize(stem + ".out_proj.weight_g"), 8);
        const auto bias = dac_source.dequantize(stem + ".out_proj.bias");
        for (unsigned symbol : {0u, 511u, 1023u})
        {
            std::array<float, 1024> column{};
            weights::project_codebook(static_cast<std::uint16_t>(symbol), vectors, folded, bias,
                                      column);
            for (std::size_t c = 0; c < column.size(); ++c)
            {
                const auto bits = std::bit_cast<std::uint32_t>(column[c]);
                const auto rounded =
                    static_cast<std::uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
                check(embeddings.values()[(std::size_t{book} * 1024 + symbol) * 1024 + c] ==
                          rounded,
                      "feedback embedding differs");
            }
        }
    }
    rejects(
        [&]
        {
            model.attention_qkv(TransformerStack::codebook, 3);
        });
}

#endif

void ownership(const std::filesystem::path& root)
{
    std::optional<TensorView<float>> retained;
    std::weak_ptr<const MappedRegion> weak;
    float first = 0;
    {
        ModelDirectory directory(root);
        // Verify that copying a directory shares its mapping cache.
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        ModelDirectory copy = directory;
        DacAnalysisModel a(directory, ModelVariant::mono), b(copy, ModelVariant::mono);
        auto weights = a.entry_convolution();
        auto other = b.entry_convolution();
        check(weights.gain.region() == other.gain.region(),
              "directory copies duplicated a mapping");
        weak = weights.gain.region();
        first = weights.gain.values()[0];
        retained = weights.gain;
        detail::ModelDirectoryAccess::prune(directory);
    }
    check(!weak.expired() && retained->values()[0] == first,
          "view did not retain mapping after adapter destruction");
    retained.reset();
    check(weak.expired(), "last tensor owner did not release mmap");
    ModelDirectory directory(root);
    rejects(
        [&]
        {
            detail::ModelDirectoryAccess::map(directory, ModelFile::count, 0, 16384);
        });
    rejects(
        [&]
        {
            detail::ModelDirectoryAccess::map(directory, ModelFile::mono_rvq_shared, 1, 16384);
        });
    rejects(
        [&]
        {
            detail::ModelDirectoryAccess::map(directory, ModelFile::mono_rvq_shared, 0, 0);
        });
}

void directory_behaviour(const std::filesystem::path& root)
{
    ModelDirectory directory(root);
    std::array<std::shared_ptr<const MappedRegion>, 8> handles;
    std::array<std::thread, 8> threads;
    for (std::size_t i = 0; i < threads.size(); ++i)
    {
        threads[i] = std::thread(
            [&, i]
            {
                handles[i] = detail::ModelDirectoryAccess::map(
                    directory, ModelFile::mono_rvq_shared, 0, 16384);
            });
    }
    for (auto& thread : threads)
    {
        thread.join();
    }
    for (const auto& handle : handles)
    {
        check(handle == handles[0], "concurrent directory requests duplicated mmap");
    }
    const std::weak_ptr<const MappedRegion> weak = handles[0];
    handles = {};
    detail::ModelDirectoryAccess::prune(directory);
    check(weak.expired(), "directory cache retained unused mapping");

    struct Temporary
    {
        std::filesystem::path path;

        Temporary()
        {
            char pattern[] = "/private/tmp/tsac-model-adapters-XXXXXX";
            auto result = ::mkdtemp(pattern);
            if (!result)
            {
                throw Error("cannot create adapter test directory");
            }
            path = result;
        }

        ~Temporary()
        {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
    } isolated;

    std::filesystem::path shared_file;
    for (const auto& item : std::filesystem::directory_iterator(root))
    {
        if (item.path().filename().string().starts_with("tsac-mono-rvq-shared-") &&
            item.path().extension() == ".weights")
        {
            shared_file = item.path();
        }
    }
    check(!shared_file.empty(), "missing RVQ fixture");
    std::filesystem::create_symlink(shared_file, isolated.path / shared_file.filename());
    ModelDirectory decode_only(isolated.path);
    {
        RvqDecoderModel decoder(decode_only, ModelVariant::mono, 2);
        check(decoder.codebook(1).size() == 8192, "decoder requested extra assets");
    }
    rejects(
        [&]
        {
            RvqEncoderModel encoder(decode_only, ModelVariant::mono, RvqInputLayout::input_major,
                                    2);
        });
    std::filesystem::remove(isolated.path / shared_file.filename());
    {
        std::ofstream truncated(isolated.path / shared_file.filename(), std::ios::binary);
        const std::vector<char> zero(16384);
        truncated.write(zero.data(), static_cast<std::streamsize>(zero.size()));
    }
    ModelDirectory shortened(isolated.path);
    auto prefix =
        detail::ModelDirectoryAccess::map(shortened, ModelFile::mono_rvq_shared, 0, 16384);
    rejects(
        [&]
        {
            detail::ModelDirectoryAccess::map(shortened, ModelFile::mono_rvq_shared, 16384, 16384);
        });
    rejects(
        [&]
        {
            TensorView<float> invalid(prefix, 16384, {{1}});
        });
    rejects(
        [&]
        {
            TensorView<float> invalid(prefix, 1, {{1}});
        });
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        check(argc == 3, "usage: tsac-model-adapters-test PACKED_DIRECTORY ORIGINALS");
        ownership(argv[1]);
        directory_behaviour(argv[1]);
        ModelDirectory directory(argv[1]);
        for (auto variant : {ModelVariant::mono, ModelVariant::stereo})
        {
            const std::string v = variant == ModelVariant::mono ? "mono" : "stereo";
            const auto original =
                SourceModel::load(std::filesystem::path(argv[2]) / ("dac_" + v + "_q8.bin"));
            DacAnalysisModel analysis(directory, variant);
            DacSynthesisModel synthesis(directory, variant);
            dac(analysis, original, false);
            dac(synthesis, original, true);
            rvq(directory, variant, original);
#if TSAC_TEST_TRANSFORMER
            const auto tf =
                SourceModel::load(std::filesystem::path(argv[2]) / ("tsac_" + v + "_q8.bin"));
            transformer(directory, variant, tf, original);
#else
            rejects(
                [&]
                {
                    TransformerModel absent(directory, variant, 2);
                });
#endif
        }
        std::cout << "Standalone adapter payloads, contracts and ownership passed\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
