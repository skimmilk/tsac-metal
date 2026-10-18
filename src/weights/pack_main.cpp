#include "../models/model_layout.hpp"
#include "common/numeric.hpp"
#include "common/numeric/bf16.hpp"
#include "models/contract.hpp"
#include "source_model.hpp"
#include "weights/normalization.hpp"
#include "weights/preparation.hpp"

#include <algorithm>
#include <bit>
#include <fstream>
#include <iostream>

namespace
{
using namespace tsac;
using namespace tsac::packing;
using namespace tsac::weights;
inline constexpr auto region_alignment = model_contract::mapping_alignment;
inline constexpr auto tensor_alignment = model_contract::tensor_alignment;
inline constexpr std::uint32_t absent = UINT32_MAX;

struct Ref
{
    std::uint32_t region = absent;
    std::size_t offset = 0, bytes = 0;
};

struct Region
{
    std::uint32_t file;
    std::size_t offset, bytes;
};

std::size_t align(std::size_t n, std::size_t a)
{
    return checked_add(n, a - 1) / a * a;
}

struct FileInfo
{
    std::string name;
    std::size_t bytes = 0;
};

struct Record
{
    unsigned channels;
    bool transformer;
    std::string name;
    Ref ref;
};

struct SourceInfo
{
    unsigned channels;
    bool transformer;
    Tensor tensor;
    Ref ref;
};

class Packer
{
public:
    std::filesystem::path output;
    std::string id;
    std::vector<FileInfo> files;
    std::vector<Region> regions;
    std::vector<SourceInfo> sources;
    std::vector<Record> prepared;
    std::ofstream stream;
    std::uint32_t file = 0, region = 0;
    bool include_transformer = true;

    void begin_file(unsigned channels, std::string_view role)
    {
        file = static_cast<std::uint32_t>(files.size());
        auto name = "tsac-" + std::string(channels == 1 ? "mono-" : "stereo-") + std::string(role) +
                    "-" + id + ".weights";
        files.push_back({name, 0});
        stream.open(output / name, std::ios::binary | std::ios::trunc);
        if (!stream)
        {
            throw Error("cannot create weight file");
        }
    }

    void pad(std::size_t alignment)
    {
        const auto target = align(files[file].bytes, alignment);
        std::array<char, region_alignment> zero{};
        const auto count = target - files[file].bytes;
        stream.write(zero.data(), static_cast<std::streamsize>(count));
        files[file].bytes = target;
    }

    void begin_region()
    {
        pad(region_alignment);
        region = static_cast<std::uint32_t>(regions.size());
        regions.push_back({file, files[file].bytes, 0});
    }

    void end_region()
    {
        pad(region_alignment);
        regions[region].bytes = files[file].bytes - regions[region].offset;
    }

    void end_file()
    {
        stream.flush();
        if (!stream)
        {
            throw Error("cannot write complete weight file");
        }
        stream.close();
    }

    Ref put(ByteView bytes)
    {
        pad(tensor_alignment);
        const Ref result{region, files[file].bytes - regions[region].offset, bytes.size()};
        stream.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        files[file].bytes = checked_add(files[file].bytes, bytes.size());
        return result;
    }

    Ref floats(std::span<const float> values)
    {
        detail::require_finite(values, "nonfinite prepared weights");
        return put(float32_le(values));
    }

    Ref halves(std::span<const std::uint16_t> values)
    {
        Bytes bytes(values.size() * 2);
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            detail::require((values[i] & 0x7f80u) != 0x7f80u, "nonfinite BF16 weight");
            bytes[i * 2] = static_cast<std::uint8_t>(values[i]);
            bytes[i * 2 + 1] = static_cast<std::uint8_t>(values[i] >> 8);
        }
        return put(bytes);
    }

    SourceInfo& source(unsigned channels, bool transformer, const std::string& name)
    {
        auto found = std::find_if(sources.begin(), sources.end(),
                                  [&](const auto& s)
                                  {
                                      return s.channels == channels &&
                                             s.transformer == transformer && s.tensor.name == name;
                                  });
        if (found == sources.end())
        {
            throw Error("unknown packed source: " + name);
        }
        return *found;
    }

    void bind(unsigned c, bool t, const std::string& name, Ref ref, Storage storage)
    {
        auto& s = source(c, t, name);
        s.ref = ref;
        s.tensor.storage = storage;
        s.tensor.payload_bytes = ref.bytes;
        s.tensor.data_plane_bytes = storage == Storage::bf8 || storage == Storage::bf8l
                                        ? s.tensor.data_plane_bytes
                                        : ref.bytes;
    }

    Ref constant(unsigned c, bool t, const SourceModel& model, const std::string& name)
    {
        const auto ref = floats(model.dequantize(name));
        bind(c, t, name, ref, Storage::f32);
        return ref;
    }

    void named(unsigned c, bool t, std::string name, Ref ref)
    {
        prepared.push_back({c, t, std::move(name), ref});
    }

    void dac(unsigned channels, const SourceModel& model)
    {
        for (const auto& stage : {std::string("analysis"), std::string("synthesis")})
        {
            const auto prefix = stage == "analysis" ? "encoder." : "decoder.";
            begin_file(channels, "dac-" + stage + "-source");
            begin_region();
            // Original tensor inventory is already in graph order (v/g order
            // here is explicit), with no per-record metadata in output files.
            for (const auto& tensor : model.tensors())
            {
                if (!tensor.name.starts_with(prefix) || !tensor.name.ends_with(".weight_v"))
                {
                    continue;
                }
                const auto stem = tensor.name.substr(0, tensor.name.size() - 9);
                const auto gain = stem + ".weight_g";
                auto values = model.dequantize(tensor.name);
                auto gains = model.dequantize(gain);
                // Validate the exact preparation contract offline too.
                (void)fold_dac_weight_norm(values, gains,
                                           tensor.dimensions[0] * tensor.dimensions[1]);
                bind(channels, false, tensor.name, put(model.tensor_payload(tensor.name)),
                     tensor.storage);
                bind(channels, false, gain, floats(gains), Storage::f32);
            }
            end_region();
            end_file();
            begin_file(channels, "dac-" + stage + "-shared");
            begin_region();
            for (const auto& tensor : model.tensors())
            {
                if (tensor.name.starts_with(prefix) &&
                    (tensor.name.ends_with(".bias") || tensor.name.ends_with(".alpha")))
                {
                    auto values = model.dequantize(tensor.name);
                    if (tensor.name.ends_with(".alpha"))
                    {
                        for (auto a : values)
                        {
                            detail::require(std::isfinite(a + snake_epsilon_value()) &&
                                                a + snake_epsilon_value() != 0,
                                            "invalid Snake denominator");
                        }
                    }
                    bind(channels, false, tensor.name, floats(values), Storage::f32);
                }
            }
            end_region();
            end_file();
        }
    }

    static float snake_epsilon_value()
    {
        return std::bit_cast<float>(std::uint32_t{0x3089705f});
    }

    void rvq(unsigned channels, unsigned books, const SourceModel& dac)
    {
        for (const auto& role : {std::string("shared"), std::string("encode-shared"),
                                 std::string("encode-apple"), std::string("encode-metal")})
        {
            begin_file(channels, "rvq-" + role);
            for (unsigned book = 0; book < books; ++book)
            {
                begin_region();
                const auto prefix = "quantizer.quantizers." + std::to_string(book);
                const auto key = prefix + ".";
                if (role == "shared")
                {
                    named(channels, false, key + "codebook",
                          constant(channels, false, dac, prefix + ".codebook.weight"));
                    const auto w =
                        fold_dac_weight_norm(dac.dequantize(prefix + ".out_proj.weight_v"),
                                             dac.dequantize(prefix + ".out_proj.weight_g"), 8);
                    named(channels, false, key + "output_weights", floats(w));
                    named(channels, false, key + "output_bias",
                          constant(channels, false, dac, prefix + ".out_proj.bias"));
                }
                else if (role == "encode-shared")
                {
                    const auto codebook = dac.dequantize(prefix + ".codebook.weight");
                    std::vector<float> normalized(codebook.size());
                    for (std::size_t symbol = 0; symbol < 1024; ++symbol)
                    {
                        normalize_search(std::span<const float, 8>(codebook.data() + symbol * 8, 8),
                                         std::span<float, 8>(normalized.data() + symbol * 8, 8));
                    }
                    named(channels, false, key + "normalized_search", floats(normalized));
                    named(channels, false, key + "input_bias",
                          constant(channels, false, dac, prefix + ".in_proj.bias"));
                }
                else
                {
                    auto w =
                        fold_dac_weight_norm(dac.dequantize(prefix + ".in_proj.weight_v"),
                                             dac.dequantize(prefix + ".in_proj.weight_g"), 1024);
                    if (role == "encode-metal")
                    {
                        auto packed = w;
                        for (std::size_t out = 0; out < 8; ++out)
                        {
                            for (std::size_t in = 0; in < 1024; ++in)
                            {
                                packed[out + 8 * in] = w[in + 1024 * out];
                            }
                        }
                        w = std::move(packed);
                    }
                    named(channels, false,
                          key + "input_" + (role == "encode-apple" ? "apple" : "metal"), floats(w));
                }
                end_region();
            }
            end_file();
        }
    }

    void transformer(unsigned channels, unsigned books, const SourceModel& dac,
                     const SourceModel& model)
    {
        begin_file(channels, "transformer-source");
        begin_region();
        for (const auto& tensor : model.tensors())
        {
            if (tensor.name.ends_with("/w") || tensor.name == "cb_proj" ||
                tensor.name == "codebook_decoder.cb_proj")
            {
                detail::require(tensor.storage == Storage::bf8l,
                                "Transformer projections require BF8L source weights");
            }
            if (tensor.storage == Storage::bf8l)
            {
                detail::require(tensor.dimensions.size() == 2 && tensor.dimensions[0] % 256 == 0 &&
                                    tensor.dimensions[1] % 8 == 0,
                                "unsupported Transformer packing shape");
                bind(channels, true, tensor.name,
                     put(model.tensor_payload(tensor.name).first(tensor.elements)), Storage::bf8l);
            }
        }
        end_region();
        end_file();
        begin_file(channels, "transformer-shared");
        begin_region();
        for (const auto& tensor : model.tensors())
        {
            if (tensor.storage == Storage::bf8l)
            {
                const auto d = tensor.dimensions[0], o = tensor.dimensions[1], padded = align(o, 8);
                const auto payload = model.tensor_payload(tensor.name);
                std::vector<float> scales(d / 256 * o);
                for (std::size_t b = 0; b < d / 256; ++b)
                {
                    for (std::size_t out = 0; out < o; ++out)
                    {
                        scales[b * o + out] =
                            detail::bf16(payload, tensor.data_plane_bytes + 2 * (b * padded + out));
                    }
                }
                named(channels, true, tensor.name + ".scales", floats(scales));
                std::vector<float> bias(o, 0);
                std::string bias_name;
                if (tensor.name.ends_with("/w"))
                {
                    bias_name = tensor.name.substr(0, tensor.name.size() - 1) + "b";
                    bias = model.dequantize(bias_name);
                }
                auto ref = floats(bias);
                named(channels, true, tensor.name + ".bias", ref);
                if (!bias_name.empty())
                {
                    bind(channels, true, bias_name, ref, Storage::f32);
                }
            }
            else if (!tensor.name.ends_with("/b") || tensor.name.find("ln_") != std::string::npos)
            {
                const auto values = model.dequantize(tensor.name);
                std::vector<std::uint16_t> halves_values(values.size());
                std::transform(values.begin(), values.end(), halves_values.begin(),
                               probability_bf16);
                bind(channels, true, tensor.name, halves(halves_values), Storage::bf16);
            }
        }
        end_region();
        for (unsigned book = 0; book < books; ++book)
        {
            begin_region();
            const auto prefix = "quantizer.quantizers." + std::to_string(book);
            const auto codebook = dac.dequantize(prefix + ".codebook.weight");
            const auto w = fold_dac_weight_norm(dac.dequantize(prefix + ".out_proj.weight_v"),
                                                dac.dequantize(prefix + ".out_proj.weight_g"), 8);
            const auto bias = dac.dequantize(prefix + ".out_proj.bias");
            std::vector<std::uint16_t> embeddings(std::size_t{1024} * 1024);
            std::array<float, 1024> column;
            for (std::size_t symbol = 0; symbol < 1024; ++symbol)
            {
                project_codebook(static_cast<std::uint16_t>(symbol), codebook, w, bias, column);
                std::transform(column.begin(), column.end(), embeddings.data() + symbol * 1024,
                               probability_bf16);
            }
            named(channels, true, "embeddings." + std::to_string(book), halves(embeddings));
            end_region();
        }
        end_file();
    }

    void validate_dac(unsigned channels)
    {
        const auto conv = [&](const std::string& stem, std::size_t in, std::size_t out,
                              std::size_t kernel, bool transposed = false)
        {
            const auto& values = source(channels, false, stem + ".weight_v");
            const std::vector<std::size_t> expected =
                transposed ? std::vector<std::size_t>{out, kernel, in}
                           : std::vector<std::size_t>{in, kernel, out};
            detail::require(values.tensor.dimensions == expected,
                            "unsupported DAC convolution shape");
            detail::require(source(channels, false, stem + ".bias").tensor.dimensions ==
                                    std::vector<std::size_t>{out} &&
                                source(channels, false, stem + ".weight_g").tensor.dimensions ==
                                    std::vector<std::size_t>{1, 1, transposed ? in : out},
                            "unsupported DAC bias/gain shape");
        };
        const auto alpha = [&](const std::string& name, std::size_t width)
        {
            detail::require(source(channels, false, name + ".alpha").tensor.dimensions ==
                                std::vector<std::size_t>{width, 1, 1},
                            "unsupported DAC activation shape");
        };
        for (bool synthesis : {false, true})
        {
            const std::string prefix = synthesis ? "decoder.model." : "encoder.block.";
            conv(prefix + "0", synthesis ? 1024 : channels, synthesis ? 1536 : 64, 7);
            std::size_t width = synthesis ? 1536 : 64;
            for (unsigned b = 0; b < 4; ++b)
            {
                const auto stem = prefix + std::to_string(b + 1) + ".block.";
                const auto stride = synthesis ? std::array<unsigned, 4>{8, 8, 4, 2}[b]
                                              : std::array<unsigned, 4>{2, 4, 8, 8}[b];
                if (synthesis)
                {
                    alpha(stem + "0", width);
                    conv(stem + "1", width, width / 2, std::size_t{2} * stride, true);
                    width /= 2;
                }
                for (unsigned r = 0; r < 3; ++r)
                {
                    const auto unit = stem + std::to_string(r + (synthesis ? 2 : 0)) + ".block.";
                    alpha(unit + "0", width);
                    conv(unit + "1", width, width, 7);
                    alpha(unit + "2", width);
                    conv(unit + "3", width, width, 1);
                }
                if (!synthesis)
                {
                    alpha(stem + "3", width);
                    conv(stem + "4", width, width * 2, std::size_t{2} * stride);
                    width *= 2;
                }
            }
            alpha(prefix + "5", width);
            conv(prefix + "6", width, synthesis ? channels : 1024, synthesis ? 7 : 3);
        }
    }

    void adapter_header()
    {
        namespace contract = model_contract;
        using Range = model_layout::Range;
        std::ofstream h(output / "adapter_offsets.hpp");
        h << "// Generated application entrypoints. Shapes and encodings live in the contract.\n"
             "#pragma once\n#include \"model_layout.hpp\"\nnamespace tsac::adapter_generated {\n"
             "using model_layout::Range;\n";
        h << "inline constexpr char generation_id[]=\"" << id << "\";\n";
        h << "inline constexpr bool has_transformer=" << (include_transformer ? "true" : "false")
          << ";\n";
        constexpr std::array<std::string_view, 10> roles{
            "dac-analysis-source",  "dac-analysis-shared", "dac-synthesis-source",
            "dac-synthesis-shared", "rvq-shared",          "rvq-encode-shared",
            "rvq-encode-apple",     "rvq-encode-metal",    "transformer-source",
            "transformer-shared"};
        h << "inline constexpr std::array<const char*,20> filenames{{\n";
        for (unsigned c : {1u, 2u})
        {
            for (auto role : roles)
            {
                const auto name = "tsac-" + std::string(c == 1 ? "mono-" : "stereo-") +
                                  std::string(role) + "-" + id + ".weights";
                const auto found = std::find_if(files.begin(), files.end(),
                                                [&](const auto& f)
                                                {
                                                    return f.name == name;
                                                });
                if (found == files.end())
                {
                    h << "nullptr,\n";
                }
                else
                {
                    h << '"' << found->name << "\",\n";
                }
            }
        }
        h << "}};\n";
        const auto range = [&](Ref ref)
        {
            detail::require(ref.region != absent, "missing application entrypoint");
            return Range{checked_add(regions.at(ref.region).offset, ref.offset), ref.bytes};
        };
        const auto source_range = [&](unsigned c, bool tf, const std::string& name,
                                      TensorShape shape, TensorEncoding encoding)
        {
            const auto& s = source(c, tf, name);
            const std::vector<std::size_t> expected(
                shape.axes.begin(), shape.axes.begin() + static_cast<std::ptrdiff_t>(shape.rank()));
            const auto storage = encoding == TensorEncoding::float32 ? Storage::f32
                                 : encoding == TensorEncoding::bf16  ? Storage::bf16
                                 : encoding == TensorEncoding::bf8   ? Storage::bf8
                                                                     : Storage::bf8l;
            detail::require(s.tensor.dimensions == expected && s.tensor.storage == storage &&
                                s.ref.bytes == contract::encoded_bytes(shape, encoding),
                            ("source disagrees with application contract: " + name).c_str());
            return range(s.ref);
        };
        const auto prepared_range = [&](unsigned c, bool tf, const std::string& name,
                                        TensorShape shape,
                                        TensorEncoding encoding = TensorEncoding::float32)
        {
            const auto p =
                std::find_if(prepared.begin(), prepared.end(),
                             [&](const auto& r)
                             {
                                 return r.channels == c && r.transformer == tf && r.name == name;
                             });
            detail::require(p != prepared.end() &&
                                p->ref.bytes == contract::encoded_bytes(shape, encoding),
                            ("prepared entry disagrees with contract: " + name).c_str());
            return range(p->ref);
        };
        const auto print = [&](Range r)
        {
            h << '{' << r.offset << ',' << r.bytes << '}';
        };
        const auto ranges = [&](const std::string& name, const std::vector<Range>& entries)
        {
            h << "inline constexpr std::array<Range," << entries.size() << "> " << name << "{{\n";
            for (auto r : entries)
            {
                print(r);
                h << ",\n";
            }
            h << "}};\n";
        };
        const auto rows = [&]<std::size_t N>(const std::string& name,
                                             const std::vector<std::array<Range, N>>& entries)
        {
            h << "inline constexpr std::array<std::array<Range," << N << ">," << entries.size()
              << "> " << name << "{{\n";
            for (const auto& row : entries)
            {
                h << "{{";
                for (auto r : row)
                {
                    print(r);
                    h << ',';
                }
                h << "}},\n";
            }
            h << "}};\n";
        };
        for (unsigned c : {1u, 2u})
        {
            const auto variant = c == 1 ? ModelVariant::mono : ModelVariant::stereo;
            const std::string v = c == 1 ? "mono_" : "stereo_";
            for (bool synthesis : {false, true})
            {
                const std::string prefix = synthesis ? "decoder.model." : "encoder.block.";
                const auto name = v + (synthesis ? "synthesis" : "analysis");
                std::vector<std::string> stems{prefix + "0"}, activation_names;
                for (unsigned stage = 0; stage < 4; ++stage)
                {
                    const auto block = prefix + std::to_string(stage + 1) + ".block.";
                    if (synthesis)
                    {
                        stems.push_back(block + "1");
                        activation_names.push_back(block + "0.alpha");
                    }
                    for (unsigned unit = 0; unit < 3; ++unit)
                    {
                        const auto residual =
                            block + std::to_string(unit + (synthesis ? 2 : 0)) + ".block.";
                        stems.push_back(residual + "1");
                        stems.push_back(residual + "3");
                        activation_names.push_back(residual + "0.alpha");
                        activation_names.push_back(residual + "2.alpha");
                    }
                    if (!synthesis)
                    {
                        stems.push_back(block + "4");
                        activation_names.push_back(block + "3.alpha");
                    }
                }
                stems.push_back(prefix + "6");
                activation_names.push_back(prefix + "5.alpha");
                std::vector<Range> values, gains, biases, alphas;
                for (unsigned i = 0; i < stems.size(); ++i)
                {
                    const auto shape = contract::dac_shape(synthesis, variant, i);
                    const bool transposed = synthesis && i > 0 && i < 29 && (i - 1) % 7 == 0;
                    values.push_back(source_range(c, false, stems[i] + ".weight_v", shape,
                                                  contract::dac_encoding(synthesis, i)));
                    gains.push_back(source_range(c, false, stems[i] + ".weight_g",
                                                 {{1, 1, shape.axes[2]}}, TensorEncoding::float32));
                    biases.push_back(source_range(c, false, stems[i] + ".bias",
                                                  {{shape.axes[transposed ? 0 : 2]}},
                                                  TensorEncoding::float32));
                }
                for (unsigned i = 0; i < activation_names.size(); ++i)
                {
                    alphas.push_back(source_range(c, false, activation_names[i],
                                                  contract::activation_shape(synthesis, i),
                                                  TensorEncoding::float32));
                }
                ranges(name + "_values", values);
                ranges(name + "_gains", gains);
                ranges(name + "_biases", biases);
                ranges(name + "_alphas", alphas);
                h << "inline constexpr model_layout::Dac " << name << '{' << name << "_values,"
                  << name << "_gains," << name << "_biases," << name << "_alphas};\n";
            }
            std::vector<std::array<Range, 7>> books;
            for (unsigned b = 0; b < contract::codebooks(variant); ++b)
            {
                const auto stem = "quantizer.quantizers." + std::to_string(b) + ".";
                books.push_back(
                    {prepared_range(c, false, stem + "codebook", contract::rvq_vectors),
                     prepared_range(c, false, stem + "output_weights",
                                    contract::rvq_output_weights),
                     prepared_range(c, false, stem + "output_bias", contract::rvq_output_bias),
                     prepared_range(c, false, stem + "normalized_search", contract::rvq_vectors),
                     prepared_range(c, false, stem + "input_bias", contract::rvq_input_bias),
                     prepared_range(c, false, stem + "input_apple",
                                    contract::rvq_input_shape(RvqInputLayout::input_major)),
                     prepared_range(c, false, stem + "input_metal",
                                    contract::rvq_input_shape(RvqInputLayout::output_major))});
            }
            rows(v + "books", books);
            std::vector<std::array<Range, 3>> projections;
            std::vector<std::array<Range, 2>> norms;
            std::array<Range, 2> constants{};
            Range embeddings{};
            if (include_transformer)
            {
                const auto linear = [&](const std::string& stem)
                {
                    const auto shape =
                        contract::projection_shape(static_cast<unsigned>(projections.size()));
                    projections.push_back(
                        {source_range(c, true, stem, shape, TensorEncoding::int8),
                         prepared_range(c, true, stem + ".scales",
                                        {{shape.axes[1], shape.axes[0] / 256}}),
                         prepared_range(c, true, stem + ".bias", {{shape.axes[1]}})});
                };
                const auto norm = [&](const std::string& stem)
                {
                    norms.push_back({source_range(c, true, stem + "g", contract::normalization,
                                                  TensorEncoding::bf16),
                                     source_range(c, true, stem + "b", contract::normalization,
                                                  TensorEncoding::bf16)});
                };
                for (bool cb : {false, true})
                {
                    const std::string prefix = cb ? "codebook_decoder." : "";
                    for (unsigned layer = 0; layer < (cb ? 3u : 12u); ++layer)
                    {
                        const auto stem = prefix + "h" + std::to_string(layer) + "/";
                        for (auto suffix :
                             {"attn/c_attn/w", "attn/c_proj/w", "mlp/c_fc/w", "mlp/c_proj/w"})
                        {
                            linear(stem + suffix);
                        }
                        norm(stem + "ln_1/");
                        norm(stem + "ln_2/");
                    }
                    norm(prefix + "ln_f/");
                }
                linear("cb_proj");
                linear("codebook_decoder.cb_proj");
                linear("proj_f/w");
                constants = {
                    source_range(c, true, "codebook_decoder.wpe",
                                 contract::positions_shape(contract::codebooks(variant)),
                                 TensorEncoding::bf16),
                    source_range(c, true, "rotary_embed", contract::rotary, TensorEncoding::bf16)};
                embeddings =
                    prepared_range(c, true, "embeddings.0", {{1024, 1024}}, TensorEncoding::bf16);
            }
            rows(v + "projections", projections);
            rows(v + "norms", norms);
            h << "inline constexpr model_layout::Transformer " << v << "transformer{" << v
              << "projections," << v << "norms,{{";
            for (auto r : constants)
            {
                print(r);
                h << ',';
            }
            h << "}},";
            print(embeddings);
            h << "};\n";
        }
        h << "}\n";
        h.close();
        if (!h)
        {
            throw Error("cannot generate adapter offsets header");
        }
    }

    void manifest()
    {
        std::ofstream manifest(output / "weights-manifest.cmake");
        manifest << "set(TSAC_WEIGHT_FILES\n";
        std::size_t total = 0;
        for (auto& f : files)
        {
            manifest << '"' << f.name << "\"\n";
            total += f.bytes;
        }
        manifest << ")\nset(TSAC_WEIGHT_BYTES " << total << ")\n";
        for (auto& f : files)
        {
            manifest << "set(\"TSAC_WEIGHT_SIZE_" << f.name << "\" " << f.bytes << ")\n";
        }
        manifest.close();
        if (!manifest)
        {
            throw Error("cannot generate manifest");
        }
        detail::require(total < 400000000, "weight package exceeds 400 MB budget");
        std::cout << "Packed " << files.size() << " files, " << total << " bytes (" << id << ")\n";
    }
};
} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc != 5)
        {
            throw tsac::Error(
                "usage: tsac-pack-models INPUT_DIR OUTPUT_DIR PACK_ID TRANSFORMER_0_OR_1");
        }
        Packer pack;
        pack.output = argv[2];
        pack.id = argv[3];
        tsac::detail::require(std::string_view(argv[4]) == "0" || std::string_view(argv[4]) == "1",
                              "invalid Transformer option");
        pack.include_transformer = std::string_view(argv[4]) == "1";
        tsac::detail::require(pack.id.size() == 16 && pack.id.find_first_not_of(
                                                          "0123456789abcdef") == std::string::npos,
                              "invalid pack ID");
        std::filesystem::create_directories(pack.output);
        for (unsigned channels : {1u, 2u})
        {
            const auto variant = channels == 1 ? "mono" : "stereo";
            const auto dac = tsac::packing::SourceModel::load(
                std::filesystem::path(argv[1]) / ("dac_" + std::string(variant) + "_q8.bin"));
            const auto transformer = tsac::packing::SourceModel::load(
                std::filesystem::path(argv[1]) / ("tsac_" + std::string(variant) + "_q8.bin"));
            const auto config = dac.dac_config();
            tsac::detail::require(config.channels == channels &&
                                      transformer.transformer_config().codebooks ==
                                          config.codebooks,
                                  "incompatible model roles");
            for (const auto* model : {&dac, &transformer})
            {
                for (const auto& tensor : model->tensors())
                {
                    pack.sources.push_back({channels, model == &transformer, tensor, {}});
                }
            }
            pack.dac(channels, dac);
            pack.rvq(channels, config.codebooks, dac);
            if (pack.include_transformer)
            {
                pack.transformer(channels, config.codebooks, dac, transformer);
            }
        }
        pack.validate_dac(1);
        pack.validate_dac(2);
        pack.adapter_header();
        pack.manifest();
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "tsac-pack-models: " << e.what() << '\n';
        return 1;
    }
}
