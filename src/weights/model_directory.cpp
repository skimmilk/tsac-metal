#include "weights/model_directory.hpp"

#include "adapter_offsets.hpp"
#include "common/internal.hpp"
#include "models/contract.hpp"

#include <fcntl.h>
#include <map>
#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <tuple>
#include <unistd.h>

namespace tsac
{
struct ModelDirectory::State
{
    std::filesystem::path directory;
    Limits limits;
    std::mutex mutex;
    using Key = std::tuple<ModelFile, std::size_t, std::size_t>;
    std::map<Key, std::weak_ptr<const MappedRegion>> mappings;
};

MappedRegion::~MappedRegion()
{
    if (size_)
    {
        ::munmap(const_cast<std::uint8_t*>(data_), size_);
    }
}

ModelDirectory::ModelDirectory(std::filesystem::path directory, const Limits& limits)
    : state_(std::make_shared<State>())
{
#ifdef TSAC_BUILD_WEIGHT_ROOT
    if (directory == std::filesystem::path(TSAC_BUILD_BINARY_DIR) ||
        directory.parent_path() == std::filesystem::path(TSAC_BUILD_BINARY_DIR))
    {
        directory =
            std::filesystem::path(TSAC_BUILD_WEIGHT_ROOT) / adapter_generated::generation_id;
    }
#endif
    state_->directory = std::move(directory);
    state_->limits = limits;
}

const std::filesystem::path& ModelDirectory::path() const
{
    return state_->directory;
}

std::shared_ptr<const MappedRegion>
detail::ModelDirectoryAccess::map(const ModelDirectory& directory, ModelFile file,
                                  std::size_t offset, std::size_t bytes)
{
    const auto& state_ = directory.state_;
    const auto index = static_cast<std::size_t>(file);
    const auto& files = adapter_generated::filenames;
    static_assert(adapter_generated::filenames.size() ==
                  static_cast<std::size_t>(ModelFile::count));
    detail::require(index < files.size() && files[index] != nullptr,
                    "model file is not included in this package");
    detail::require(offset % model_contract::mapping_alignment == 0 && bytes &&
                        bytes % model_contract::mapping_alignment == 0,
                    "unsupported weight mapping alignment");
    const auto end = checked_add(offset, bytes);
    detail::require(end <= state_->limits.file_bytes, "weight mapping exceeds limit");
    std::lock_guard lock(state_->mutex);
    const ModelDirectory::State::Key key{file, offset, bytes};
    if (auto existing = state_->mappings[key].lock())
    {
        return existing;
    }
    // Allocate the RAII owner before acquiring any OS resources.
    auto region = std::shared_ptr<MappedRegion>(new MappedRegion);
    const auto path = state_->directory / files[index];
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0)
    {
        throw Error("cannot open weights: " + path.string());
    }
    struct stat status{};
    if (::fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) || status.st_size < 0 ||
        static_cast<std::uintmax_t>(status.st_size) < end)
    {
        ::close(fd);
        throw Error("weight file cannot cover requested mapping: " + path.string());
    }
    void* base = ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, static_cast<off_t>(offset));
    ::close(fd);
    if (base == MAP_FAILED)
    {
        throw Error("cannot map weights: " + path.string());
    }
    region->data_ = static_cast<const std::uint8_t*>(base);
    region->size_ = bytes;
    region->offset_ = offset;
    region->file_ = file;
    // Prefetch is advisory; the mapping remains usable if the hint is declined.
    (void)::madvise(base, bytes, MADV_WILLNEED);
    state_->mappings[key] = region;
    return region;
}

void detail::ModelDirectoryAccess::prune(const ModelDirectory& directory)
{
    const auto& state_ = directory.state_;
    std::lock_guard lock(state_->mutex);
    for (auto i = state_->mappings.begin(); i != state_->mappings.end();)
    {
        if (i->second.expired())
        {
            i = state_->mappings.erase(i);
        }
        else
        {
            ++i;
        }
    }
}
} // namespace tsac
