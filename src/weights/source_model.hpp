#pragma once

#include "common/common.hpp"
#include "models/config.hpp"

#include <memory>
#include <string_view>

namespace tsac::packing
{
enum class Storage : std::uint32_t
{
    f32 = 0,
    bf16 = 1,
    bf8 = 0x8000b,
    bf8l = 0x8000e
};

struct Tensor
{
    std::string name;
    Storage storage;
    std::vector<std::size_t> dimensions; // Axis 0 is fastest.
    std::size_t elements = 0;
    std::size_t record_offset = 0;
    std::size_t payload_offset = 0;
    std::size_t payload_bytes = 0;
    std::size_t data_plane_bytes = 0;
};

// Original .bin input reader. Built only into the packer and source-format tests.
class SourceModel
{
public:
    static SourceModel parse(Bytes bytes, const Limits& limits = {});
    static SourceModel load(const std::filesystem::path& path, const Limits& limits = {});

    const std::vector<Tensor>& tensors() const
    {
        return tensors_;
    }

    const std::string& metadata_json() const
    {
        return metadata_;
    }

    std::string digest() const
    {
        return sha256(serialized());
    }

    DacConfig dac_config() const;
    TransformerConfig transformer_config() const;
    const Tensor& tensor(std::string_view name) const;
    ByteView tensor_payload(std::string_view name) const;
    std::vector<float> dequantize(std::string_view name) const;

private:
    static SourceModel parse_data(ByteView data, const Limits& limits);

    ByteView serialized() const
    {
        return file_bytes_ ? ByteView(file_bytes_.get(), file_size_) : ByteView(bytes_);
    }

    Bytes bytes_;
    std::shared_ptr<const std::uint8_t[]> file_bytes_;
    std::size_t file_size_ = 0;
    std::string metadata_;
    std::vector<Tensor> tensors_;
};

std::string_view storage_name(Storage storage);
Bytes float32_le(std::span<const float> values);
} // namespace tsac::packing
