#pragma once
#include "adapter_offsets.hpp"
#include "model_layout.hpp"
#include "models/contract.hpp"

#include <algorithm>
#include <initializer_list>

namespace tsac::model_access
{
inline ModelFile file(ModelVariant variant, ModelFile mono)
{
    const auto base = model_contract::channels(variant) == 1 ? 0u : 10u;
    return static_cast<ModelFile>(static_cast<unsigned>(mono) + base);
}

inline std::size_t end(model_layout::Range r)
{
    return checked_add(r.offset, r.bytes);
}

inline std::size_t extent(std::initializer_list<std::span<const model_layout::Range>> groups)
{
    std::size_t result = 0;
    for (auto group : groups)
    {
        for (auto r : group)
        {
            result = std::max(result, end(r));
        }
    }
    return result;
}

inline std::shared_ptr<const MappedRegion> prefix(const ModelDirectory& directory, ModelFile file,
                                                  std::size_t end)
{
    const auto alignment = model_contract::mapping_alignment;
    return detail::ModelDirectoryAccess::map(
        directory, file, 0, checked_add(end, alignment - 1) / alignment * alignment);
}

inline std::size_t relative(const MappedRegion& region, model_layout::Range range)
{
    if (range.offset < region.file_offset())
    {
        throw Error("tensor precedes mapping");
    }
    return range.offset - region.file_offset();
}

template <class T>
TensorView<T> tensor(std::shared_ptr<const MappedRegion> region, model_layout::Range range,
                     TensorShape shape)
{
    if (range.bytes != checked_mul(shape.elements(), sizeof(T)))
    {
        throw Error("stored tensor length disagrees with model contract");
    }
    const auto offset = relative(*region, range);
    return {std::move(region), offset, shape};
}

inline std::span<const std::array<model_layout::Range, 7>> book_locations(ModelVariant variant)
{
    if (model_contract::channels(variant) == 1)
    {
        return adapter_generated::mono_books;
    }
    return adapter_generated::stereo_books;
}

inline unsigned books(ModelVariant variant, unsigned count)
{
    if (!count || count > model_contract::codebooks(variant))
    {
        throw Error("invalid enabled codebook count");
    }
    return count;
}

inline void book(unsigned index, unsigned count)
{
    if (index >= count)
    {
        throw Error("codebook was not enabled");
    }
}
} // namespace tsac::model_access
