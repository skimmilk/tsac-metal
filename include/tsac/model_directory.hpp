#pragma once
#include "tsac/types.hpp"

#include <filesystem>
#include <memory>

namespace tsac
{
namespace detail
{
struct ModelDirectoryAccess;
}

// Copyable directory handle. Copies share a weak mapping cache, never payloads.
class ModelDirectory
{
public:
    explicit ModelDirectory(std::filesystem::path directory, const Limits& limits = {});
    const std::filesystem::path& path() const;

private:
    friend struct detail::ModelDirectoryAccess;
    struct State;
    std::shared_ptr<State> state_;
};
} // namespace tsac
