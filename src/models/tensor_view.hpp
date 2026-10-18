#pragma once
#include "weights/model_directory.hpp"

#include <array>
#include <utility>

namespace tsac
{
// Axis zero is fastest. Zero trailing dimensions denote omitted axes.
struct TensorShape
{
    std::array<std::size_t, 3> axes{};
    std::size_t rank() const;
    std::size_t elements() const;
    bool operator==(const TensorShape&) const = default;
};

enum class TensorEncoding
{
    float32,
    bf16,
    int8,
    bf8
};

// Copying a view shares its mmap owner. Borrowed spans require a live view.
template <class T> class TensorView
{
public:
    TensorView(std::shared_ptr<const MappedRegion> region, std::size_t offset, TensorShape shape)
        : region_(std::move(region)), offset_(offset), shape_(shape)
    {
        const auto bytes = checked_mul(shape_.elements(), sizeof(T));
        if (!region_ || offset_ > region_->bytes().size() ||
            bytes > region_->bytes().size() - offset_ ||
            reinterpret_cast<std::uintptr_t>(region_->bytes().data() + offset_) % alignof(T))
        {
            throw Error("tensor view exceeds or misaligns its mapping");
        }
    }

    std::span<const T> values() const&
    {
        return {reinterpret_cast<const T*>(region_->bytes().data() + offset_), shape_.elements()};
    }

    std::span<const T> values() const&& = delete;

    const T* data() const&
    {
        return values().data();
    }

    const T* data() const&& = delete;

    std::size_t size() const
    {
        return shape_.elements();
    }

    TensorShape shape() const
    {
        return shape_;
    }

    std::shared_ptr<const MappedRegion> region() const
    {
        return region_;
    }

private:
    std::shared_ptr<const MappedRegion> region_;
    std::size_t offset_;
    TensorShape shape_;
};

// DAC sources have either FP32 values or signed-byte values plus BF16 scales.
// The encoding and shape come from the fixed model contract, never a file header.
class EncodedTensorView
{
public:
    EncodedTensorView(std::shared_ptr<const MappedRegion> region, std::size_t offset,
                      std::size_t bytes, TensorShape shape, TensorEncoding encoding);
    ByteView payload() const&;
    ByteView payload() const&& = delete;

    TensorShape shape() const
    {
        return shape_;
    }

    TensorEncoding encoding() const
    {
        return encoding_;
    }

    std::size_t data_plane_bytes() const;

    std::shared_ptr<const MappedRegion> region() const
    {
        return region_;
    }

private:
    std::shared_ptr<const MappedRegion> region_;
    std::size_t offset_, bytes_;
    TensorShape shape_;
    TensorEncoding encoding_;
};
} // namespace tsac
