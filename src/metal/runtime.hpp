#pragma once
#include "common/common.hpp"
#include "metal/abi/types.h"
#include "weights/tensor_weights.hpp"

#include <array>
#include <memory>
#include <string_view>
#include <type_traits>

namespace tsac::metal
{
namespace abi = ::tsac_metal_abi;
bool available();

enum class BufferStorage
{
    shared,
    gpu_only
};

// CPU access is restricted to shared storage between completed submissions.
class Buffer
{
public:
    static Buffer allocate(std::size_t bytes, BufferStorage storage = BufferStorage::shared);
    static Buffer copy(const void* data, std::size_t bytes);
    static Buffer import_weights(const weights::View& view);
    void read(void* destination, std::size_t bytes) const;
    // CPU writes are permitted only between completed submissions.
    void write(const void* source, std::size_t bytes) const;
    Buffer slice(std::size_t offset, std::size_t bytes) const;
    std::size_t size() const;
    bool shares_storage(const Buffer& other) const;
    std::span<std::byte> mapped_bytes() const;
    // Borrow CPU-visible storage after completion. The Buffer retains ownership;
    // the view remains valid until its last backing owner is released.
    std::span<const std::byte> read_only_bytes() const;

    template <class T> std::span<const T> mapped() const
    {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto bytes = read_only_bytes();
        if (bytes.size() % sizeof(T) || reinterpret_cast<std::uintptr_t>(bytes.data()) % alignof(T))
        {
            throw Error("Invalid typed Metal buffer view");
        }
        return {reinterpret_cast<const T*>(bytes.data()), bytes.size() / sizeof(T)};
    }

private:
    struct Impl;
    explicit Buffer(std::shared_ptr<Impl> impl);
    std::shared_ptr<Impl> impl_;
    std::size_t offset_ = 0, length_ = 0;
    friend class Program;
    friend class ResidencySet;
    friend class ExecutionContext;
};

// Immutable, deduplicated backing allocations retained for a model or session.
class ResidencySet
{
public:
    static std::shared_ptr<const ResidencySet> create(std::span<const Buffer> buffers);

private:
    struct Impl;
    explicit ResidencySet(std::shared_ptr<Impl> impl);
    std::shared_ptr<Impl> impl_;
    friend class ExecutionContext;
};

class Pipeline
{
public:
    Pipeline() = default;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class Program;
    friend class ExecutionContext;
};

struct ResourceBinding
{
    std::uint32_t index;
    Buffer buffer;
};

inline ResourceBinding bind(std::uint32_t index, const Buffer& buffer)
{
    return {index, buffer};
}

class Program
{
public:
    struct Dispatch
    {
        std::string_view kernel;
        std::span<const ResourceBinding> arguments;
        std::size_t threads;
        std::size_t threadgroup_width = 0;
        // Immutable wire bytes copied into the submission's parameter arena at slot 0.
        std::span<const std::byte> parameters = {};
        const Pipeline* prepared = nullptr;
        // Optional complete threadgroup grid. A zero x keeps the 1D thread grid.
        std::array<std::size_t, 3> groups = {};
    };

    static Program load(std::span<const std::byte> library, std::string_view label = {});
    Pipeline resolve(std::string_view name) const;

private:
    struct Impl;
    explicit Program(std::shared_ptr<Impl> impl);
    std::shared_ptr<Impl> impl_;
    friend class ExecutionContext;
};

// One exclusive recording owner. Reuses command memory, argument bindings and
// constant storage after synchronous completion. Model and session residency
// are stable; only changed transient allocations update the working set.
class ExecutionContext
{
public:
    explicit ExecutionContext(const Program& program,
                              std::shared_ptr<const ResidencySet> model = {});
    ~ExecutionContext();
    ExecutionContext(const ExecutionContext&) = delete;
    ExecutionContext& operator=(const ExecutionContext&) = delete;
    void set_session_resources(std::shared_ptr<const ResidencySet> resources);
    void run(std::span<const Program::Dispatch> dispatches);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace tsac::metal
