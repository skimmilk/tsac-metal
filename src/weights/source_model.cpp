#include "source_model.hpp"

#include "common/internal.hpp"

#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <unordered_set>
#include <utility>

namespace tsac::packing
{
namespace
{
// BF8 and BF8L share a signed-byte data plane and a BF16 scale plane. Only
// their axis-0 block widths differ. Keep the padded scale layout in one place
// so parsing and expansion cannot disagree about the required payload size.
struct QuantizedLayout
{
    std::size_t rows;
    std::size_t columns;
    std::size_t outer_slices;
    std::size_t block_width;
    std::size_t padded_columns;

    std::size_t scale_elements() const
    {
        return checked_mul(checked_mul(rows / block_width, padded_columns), outer_slices);
    }
};

QuantizedLayout quantized_layout(const Tensor& tensor)
{
    const std::size_t block_width = tensor.storage == Storage::bf8 ? 32 : 256;
    detail::require(tensor.dimensions.size() >= 2 && tensor.dimensions[0] % block_width == 0,
                    "unsupported quantized tensor shape");
    const auto rows = tensor.dimensions[0];
    const auto columns = tensor.dimensions[1];
    return {rows, columns, tensor.elements / rows / columns, block_width,
            checked_add(columns, 7) / 8 * 8};
}
} // namespace

std::string_view storage_name(Storage storage)
{
    switch (storage)
    {
    case Storage::f32:
        return "float32";
    case Storage::bf16:
        return "bfloat16";
    case Storage::bf8:
        return "bf8";
    case Storage::bf8l:
        return "bf8l";
    }
    throw Error("unsupported tensor storage");
}

SourceModel SourceModel::load(const std::filesystem::path& path, const Limits& limits)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw Error("cannot open input: " + path.string());
    }
    const auto end = file.tellg();
    detail::require(
        end >= 0 && static_cast<std::uintmax_t>(end) <= limits.file_bytes &&
            static_cast<std::uintmax_t>(end) <=
                static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max()),
        "input exceeds file size limit");
    const auto size = static_cast<std::size_t>(end);
    auto bytes = std::make_unique_for_overwrite<std::uint8_t[]>(size);
    file.seekg(0);
    if (size &&
        !file.read(reinterpret_cast<char*>(bytes.get()), static_cast<std::streamsize>(size)))
    {
        throw Error("cannot read complete input");
    }
    detail::require(file.peek() == std::char_traits<char>::eof(), "input changed while reading");
    auto model = parse_data(ByteView(bytes.get(), size), limits);
    model.file_bytes_ = std::move(bytes);
    model.file_size_ = size;
    return model;
}

SourceModel SourceModel::parse(Bytes bytes, const Limits& limits)
{
    auto model = parse_data(bytes, limits);
    model.bytes_ = std::move(bytes);
    return model;
}

SourceModel SourceModel::parse_data(ByteView data, const Limits& limits)
{
    detail::require(data.size() <= limits.file_bytes, "model exceeds file size limit");
    detail::require(detail::le32(data, 0) == 0x23f4aefb, "invalid model magic");
    const auto metadata_size = detail::le32(data, 4);
    detail::require(metadata_size > 0 && metadata_size <= limits.metadata_bytes,
                    "invalid metadata length");
    const auto metadata = detail::slice(data, 8, metadata_size);
    SourceModel model;
    model.metadata_.assign(metadata.begin(), metadata.end());
    auto offset = checked_add(8, metadata_size);
    std::unordered_set<std::string> names;
    while (offset < data.size())
    {
        detail::require(model.tensors_.size() < limits.tensors, "tensor count exceeds limit");
        detail::slice(data, offset, 16);
        detail::require(detail::le32(data, offset) == 0x23f4aefa, "invalid tensor magic");
        Tensor descriptor;
        descriptor.record_offset = offset;
        descriptor.storage = static_cast<Storage>(detail::le32(data, offset + 4));
        // Reject unsupported type/alignment/exponent combinations.
        storage_name(descriptor.storage);
        const auto rank = detail::le32(data, offset + 8);
        const auto name_size = detail::le32(data, offset + 12);
        detail::require(rank > 0 && rank <= 8, "invalid tensor rank");
        detail::require(name_size > 0 && name_size <= limits.name_bytes,
                        "invalid tensor name length");
        offset = checked_add(offset, 16);
        descriptor.elements = 1;
        for (std::size_t axis = 0; axis < rank; ++axis)
        {
            const auto dimension = detail::le32(data, offset);
            detail::require(dimension > 0, "zero tensor dimension");
            descriptor.elements = checked_mul(descriptor.elements, dimension);
            detail::require(descriptor.elements <= limits.tensor_elements,
                            "tensor element count exceeds limit");
            descriptor.dimensions.push_back(dimension);
            offset = checked_add(offset, 4);
        }
        const auto name = detail::slice(data, offset, name_size);
        for (auto c : name)
        {
            detail::require(c >= 32 && c < 127, "non-ASCII tensor name");
        }
        descriptor.name.assign(name.begin(), name.end());
        detail::require(names.insert(descriptor.name).second, "duplicate tensor name");
        offset = checked_add(offset, name_size);
        descriptor.payload_offset = offset;
        if (descriptor.storage == Storage::f32 || descriptor.storage == Storage::bf16)
        {
            descriptor.payload_bytes =
                checked_mul(descriptor.elements, descriptor.storage == Storage::f32 ? 4 : 2);
            descriptor.data_plane_bytes = descriptor.payload_bytes;
        }
        else
        {
            const auto layout = quantized_layout(descriptor);
            descriptor.data_plane_bytes = detail::align256(descriptor.elements);
            descriptor.payload_bytes =
                checked_add(descriptor.data_plane_bytes,
                            detail::align256(checked_mul(layout.scale_elements(), 2)));
        }
        detail::slice(data, offset, descriptor.payload_bytes);
        offset = checked_add(offset, descriptor.payload_bytes);
        model.tensors_.push_back(std::move(descriptor));
    }
    detail::require(!model.tensors_.empty(), "model has no tensors");
    return model;
}

DacConfig SourceModel::dac_config() const
{
    const auto expect = [&](std::string_view name, std::initializer_list<std::size_t> dimensions)
    {
        detail::require(tensor(name).dimensions == std::vector<std::size_t>(dimensions),
                        ("unsupported DAC tensor shape: " + std::string(name)).c_str());
    };
    const auto& output = tensor("decoder.model.6.weight_v").dimensions;
    detail::require(output.size() == 3 && output[0] == 96 && output[1] == 7 &&
                        (output[2] == 1 || output[2] == 2),
                    "unsupported DAC output channels or shape");
    const auto channels = static_cast<unsigned>(output[2]);
    const unsigned books = channels == 1 ? 9 : 12;
    expect("decoder.model.0.weight_v", {1024, 7, 1536});
    expect("decoder.model.0.weight_g", {1, 1, 1536});
    expect("decoder.model.0.bias", {1536});
    expect("decoder.model.6.weight_g", {1, 1, channels});
    expect("decoder.model.6.bias", {channels});
    for (unsigned book = 0; book < books; ++book)
    {
        const auto prefix = "quantizer.quantizers." + std::to_string(book);
        expect(prefix + ".codebook.weight", {8, 1024});
        expect(prefix + ".out_proj.weight_v", {8, 1, 1024});
        expect(prefix + ".out_proj.weight_g", {1, 1, 1024});
        expect(prefix + ".out_proj.bias", {1024});
    }
    // Reject mixed or out-of-range quantizer inventories, including gaps that
    // might otherwise remain unnoticed at lower stream qualities.
    constexpr std::string_view prefix = "quantizer.quantizers.";
    for (const auto& t : tensors_)
    {
        if (!t.name.starts_with(prefix))
        {
            continue;
        }
        const auto suffix = std::string_view(t.name).substr(prefix.size());
        const auto dot = suffix.find('.');
        unsigned book = 0;
        const auto result = std::from_chars(
            suffix.data(), suffix.data() + (dot == suffix.npos ? suffix.size() : dot), book);
        detail::require(dot != suffix.npos && result.ec == std::errc{} &&
                            result.ptr == suffix.data() + dot && book < books &&
                            suffix.substr(0, dot) == std::to_string(book),
                        "unsupported DAC quantizer inventory");
    }
    return {channels, books};
}

TransformerConfig SourceModel::transformer_config() const
{
    const auto& positions = tensor("codebook_decoder.wpe").dimensions;
    detail::require(positions.size() == 2 && positions[0] == 512 &&
                        (positions[1] == 9 || positions[1] == 12),
                    "unsupported Transformer codebook positions");
    const auto books = static_cast<unsigned>(positions[1]);
    std::size_t expected_tensors = 0;
    const auto expect = [&](const std::string& name, std::initializer_list<std::size_t> dimensions)
    {
        const auto& descriptor = tensor(name);
        detail::require(descriptor.dimensions == std::vector<std::size_t>(dimensions),
                        ("unsupported Transformer tensor shape: " + name).c_str());
        ++expected_tensors;
    };
    for (const auto& [prefix, layers] :
         std::array<std::pair<std::string, unsigned>, 2>{{{"", 12}, {"codebook_decoder.", 3}}})
    {
        for (unsigned layer = 0; layer < layers; ++layer)
        {
            const auto stem = prefix + "h" + std::to_string(layer) + "/";
            for (const auto norm : {"ln_1/", "ln_2/"})
            {
                expect(stem + norm + "g", {512});
                expect(stem + norm + "b", {512});
            }
            expect(stem + "attn/c_attn/w", {512, 1536});
            expect(stem + "attn/c_attn/b", {1536});
            expect(stem + "attn/c_proj/w", {512, 512});
            expect(stem + "attn/c_proj/b", {512});
            expect(stem + "mlp/c_fc/w", {512, 2048});
            expect(stem + "mlp/c_fc/b", {2048});
            expect(stem + "mlp/c_proj/w", {2048, 512});
            expect(stem + "mlp/c_proj/b", {512});
        }
        expect(prefix + "ln_f/g", {512});
        expect(prefix + "ln_f/b", {512});
    }
    expect("codebook_decoder.wpe", {512, books});
    expect("cb_proj", {1024, 512});
    expect("codebook_decoder.cb_proj", {1024, 512});
    expect("proj_f/w", {512, 1024});
    expect("proj_f/b", {1024});
    expect("rotary_embed", {128, 1024});
    detail::require(tensors_.size() == expected_tensors,
                    "unsupported Transformer tensor inventory");
    return {books};
}

ByteView SourceModel::tensor_payload(std::string_view name) const
{
    const auto& descriptor = tensor(name);
    return detail::slice(serialized(), descriptor.payload_offset, descriptor.payload_bytes);
}

const Tensor& SourceModel::tensor(std::string_view name) const
{
    for (const auto& descriptor : tensors_)
    {
        if (descriptor.name == name)
        {
            return descriptor;
        }
    }
    throw Error("tensor not found: " + std::string(name));
}

std::vector<float> SourceModel::dequantize(std::string_view name) const
{
    const auto& descriptor = tensor(name);
    const auto payload = tensor_payload(name);
    std::vector<float> values(descriptor.elements);
    if (descriptor.storage == Storage::f32)
    {
        static_assert(std::endian::native == std::endian::little);
        std::memcpy(values.data(), payload.data(), descriptor.elements * sizeof(float));
    }
    else if (descriptor.storage == Storage::bf16)
    {
        std::size_t i = 0;
        for (; i < descriptor.elements; ++i)
        {
            values[i] = detail::bf16(payload, i * 2);
        }
    }
    else
    {
        const auto layout = quantized_layout(descriptor);
        const auto scale_rows = layout.rows / layout.block_width;
        for (std::size_t outer = 0; outer < layout.outer_slices; ++outer)
        {
            for (std::size_t column = 0; column < layout.columns; ++column)
            {
                for (std::size_t block = 0; block < scale_rows; ++block)
                {
                    const auto index = (outer * layout.columns + column) * layout.rows +
                                       block * layout.block_width;
                    const auto scale_index =
                        (outer * scale_rows + block) * layout.padded_columns + column;
                    const auto scale =
                        detail::bf16(payload, descriptor.data_plane_bytes + scale_index * 2);
                    for (std::size_t row = 0; row < layout.block_width; ++row)
                    {
                        values[index + row] =
                            static_cast<float>(std::bit_cast<std::int8_t>(payload[index + row])) *
                            scale;
                    }
                }
            }
        }
    }
    return values;
}

Bytes float32_le(std::span<const float> values)
{
    Bytes bytes(checked_mul(values.size(), 4));
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        const auto bits = std::bit_cast<std::uint32_t>(values[i]);
        for (unsigned j = 0; j < 4; ++j)
        {
            bytes[i * 4 + j] = static_cast<std::uint8_t>(bits >> (j * 8));
        }
    }
    return bytes;
}
} // namespace tsac::packing
