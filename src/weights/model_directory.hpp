#pragma once
#include "tsac/model_directory.hpp"

#include "common/common.hpp"

namespace tsac
{
// Stable file roles. Hashed filenames are private generated implementation data.
enum class ModelFile : std::uint32_t
{
    mono_dac_analysis_source,
    mono_dac_analysis_shared,
    mono_dac_synthesis_source,
    mono_dac_synthesis_shared,
    mono_rvq_shared,
    mono_rvq_encode_shared,
    mono_rvq_encode_apple,
    mono_rvq_encode_metal,
    mono_transformer_source,
    mono_transformer_shared,
    stereo_dac_analysis_source,
    stereo_dac_analysis_shared,
    stereo_dac_synthesis_source,
    stereo_dac_synthesis_shared,
    stereo_rvq_shared,
    stereo_rvq_encode_shared,
    stereo_rvq_encode_apple,
    stereo_rvq_encode_metal,
    stereo_transformer_source,
    stereo_transformer_shared,
    count
};

// A single mmap allocation. Share its handle; the final owner calls munmap.
class MappedRegion final
{
public:
    ~MappedRegion();
    MappedRegion(const MappedRegion&) = delete;
    MappedRegion& operator=(const MappedRegion&) = delete;

    ByteView bytes() const
    {
        return {data_, size_};
    }

    ModelFile file() const
    {
        return file_;
    }

    std::size_t file_offset() const
    {
        return offset_;
    }

private:
    friend struct detail::ModelDirectoryAccess;
    MappedRegion() = default;
    std::shared_ptr<const void> storage_; // Owns a portable allocation when mmap is unavailable.
    const std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0, offset_ = 0;
    ModelFile file_{};
};

namespace detail
{
struct ModelDirectoryAccess
{
    static std::shared_ptr<const MappedRegion> map(const ModelDirectory& directory, ModelFile file,
                                                   std::size_t offset, std::size_t bytes);
    static void prune(const ModelDirectory& directory);
};
} // namespace detail
} // namespace tsac
