#include "session.hpp"

#include "common/internal.hpp"

#include <algorithm>

namespace tsac::metal::transformer
{
Session::Session(std::shared_ptr<const PreparedModel> source)
    : prepared(std::move(source)), model(*prepared), recorder(model.residency)
{
}

void Session::reset(const Header& stream_header, std::size_t count)
{
    validate_header(stream_header, model.limits);
    detail::require(
        stream_header.method == 1 && stream_header.model_channels == model.channels &&
            stream_header.codebooks <= model.maximum_books &&
            count == checked_mul(std::min<std::size_t>(stream_header.batch, stream_header.frames),
                                 stream_header.groups),
        "Metal Transformer stream/model mismatch");
    const auto& h = stream_header;
    const auto base = h.frames / h.batch, extra = h.frames % h.batch;
    const auto contexts = checked_mul(
        h.groups,
        checked_add(checked_mul(h.batch - extra, std::min<std::size_t>(base, context_frames)),
                    checked_mul(extra, std::min<std::size_t>(base + 1, context_frames))));
    const auto temporal_bytes = checked_mul(contexts, std::size_t{12} * 512 * 4);
    const auto codebook_bytes =
        checked_mul(checked_mul(count, h.codebooks), std::size_t{3} * 512 * 4);
    detail::require(
        checked_add(temporal_bytes, codebook_bytes) <= model.limits.cache_bytes,
        "Transformer caches exceed byte limit; reduce the encoding batch size or channel count");
    detail::require(count <= model.limits.tensor_elements &&
                        std::size_t(h.codebooks) * 512 <= model.limits.tensor_elements,
                    "Transformer cache exceeds element limit");

    const bool same_geometry = initialized && header.frames == h.frames &&
                               header.batch == h.batch && header.groups == h.groups &&
                               header.codebooks == h.codebooks;
    if (same_geometry)
    {
        recorder.reset_session(cache_residency);
        header = h;
        for (std::size_t i = 0; i < lanes.size(); ++i)
        {
            reset_temporal(i);
            reset_codebook(i);
            lanes[i].probabilities.clear();
        }
        return;
    }
    initialized = false;
    recorder.reset_session({});
    cache_residency.reset();
    lanes.clear();
    temporal_keys.clear();
    temporal_values.clear();
    codebook_keys.clear();
    codebook_values.clear();
    header = h;
    temporal_extent = std::min<std::size_t>(base + (extra != 0), context_frames);
    const auto padded_elements = checked_mul(checked_mul(count, temporal_extent), 512);
    const auto padded_bytes = checked_mul(padded_elements, std::size_t{12} * 4);
    const bool uniform = padded_elements <= model.limits.tensor_elements &&
                         checked_add(padded_bytes, codebook_bytes) <= model.limits.cache_bytes;
    const auto temporal_stride = checked_mul(temporal_extent, 1024);
    const auto codebook_stride = checked_mul(h.codebooks, 1024);
    for (std::size_t layer = 0; layer < model.temporal.layers.size() && count && uniform; ++layer)
    {
        temporal_keys.push_back(
            Buffer::allocate(checked_mul(count, temporal_stride), BufferStorage::gpu_only));
        temporal_values.push_back(
            Buffer::allocate(checked_mul(count, temporal_stride), BufferStorage::gpu_only));
    }
    // Codebook caches use a common stride; their whole backing allocation must
    // also fit the per-buffer tensor limit before allocation.
    detail::require(checked_mul(count, std::size_t(h.codebooks) * 512) <=
                        model.limits.tensor_elements,
                    "Transformer codebook cache exceeds element limit");
    for (std::size_t layer = 0; layer < model.codebook.layers.size() && count; ++layer)
    {
        codebook_keys.push_back(
            Buffer::allocate(checked_mul(count, codebook_stride), BufferStorage::gpu_only));
        codebook_values.push_back(
            Buffer::allocate(checked_mul(count, codebook_stride), BufferStorage::gpu_only));
    }
    lanes.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        const auto extent = std::min<std::size_t>(base + (i / h.groups < extra), context_frames);
        detail::require(checked_mul(extent, 512) <= model.limits.tensor_elements,
                        "Transformer temporal cache exceeds element limit");
        Lane lane{{}, {}, Buffer::allocate(1024, BufferStorage::gpu_only), {}};
        const auto bytes = checked_mul(extent, 1024);
        for (std::size_t layer = 0; layer < model.temporal.layers.size(); ++layer)
        {
            lane.temporal.layers.push_back(
                {uniform ? temporal_keys[layer].slice(i * temporal_stride, bytes)
                         : Buffer::allocate(bytes, BufferStorage::gpu_only),
                 uniform ? temporal_values[layer].slice(i * temporal_stride, bytes)
                         : Buffer::allocate(bytes, BufferStorage::gpu_only),
                 0});
        }
        for (std::size_t layer = 0; layer < model.codebook.layers.size(); ++layer)
        {
            lane.codebook.layers.push_back(
                {codebook_keys[layer].slice(i * codebook_stride, codebook_stride),
                 codebook_values[layer].slice(i * codebook_stride, codebook_stride), 0});
        }
        lanes.push_back(std::move(lane));
    }
    std::vector<Buffer> resources;
    for (const auto& lane : lanes)
    {
        resources.push_back(lane.conditioning);
        for (const auto* cache : {&lane.temporal, &lane.codebook})
        {
            for (const auto& layer : cache->layers)
            {
                resources.push_back(layer.key);
                resources.push_back(layer.value);
            }
        }
    }
    cache_residency = ResidencySet::create(resources);
    recorder.reset_session(cache_residency);
    initialized = true;
}

std::size_t Session::cache_bytes() const
{
    std::size_t bytes = 0;
    if (!temporal_keys.empty())
    {
        for (std::size_t i = 0; i < temporal_keys.size(); ++i)
        {
            bytes =
                checked_add(bytes, checked_add(temporal_keys[i].size(), temporal_values[i].size()));
        }
    }
    for (const auto& lane : lanes)
    {
        if (temporal_keys.empty())
        {
            for (const auto& layer : lane.temporal.layers)
            {
                bytes = checked_add(bytes, checked_add(layer.key.size(), layer.value.size()));
            }
        }
    }
    for (std::size_t i = 0; i < codebook_keys.size(); ++i)
    {
        bytes = checked_add(bytes, checked_add(codebook_keys[i].size(), codebook_values[i].size()));
    }
    return bytes;
}

void Session::reset_temporal(std::size_t index)
{
    for (auto& layer : lanes.at(index).temporal.layers)
    {
        layer.position = 0;
    }
}

void Session::reset_codebook(std::size_t index)
{
    for (auto& layer : lanes.at(index).codebook.layers)
    {
        layer.position = 0;
    }
}
} // namespace tsac::metal::transformer
