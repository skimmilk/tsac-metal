#pragma once
#include "models/tensor_view.hpp"

#include <memory>

namespace tsac::weights
{
// Retains the entire mmap allocation while describing one immutable tensor.
struct View
{
    std::shared_ptr<const void> owner;
    ByteView mapping;
    std::size_t offset = 0, bytes = 0;

    View() = default;

    View(std::shared_ptr<const void> owner, ByteView mapping, std::size_t offset, std::size_t bytes)
        : owner(std::move(owner)), mapping(mapping), offset(offset), bytes(bytes)
    {
    }

    template <class T>
    explicit View(const TensorView<T>& tensor)
        : owner(tensor.region()), mapping(tensor.region()->bytes()),
          offset(static_cast<std::size_t>(reinterpret_cast<const std::uint8_t*>(tensor.data()) -
                                          mapping.data())),
          bytes(checked_mul(tensor.size(), sizeof(T)))
    {
    }

    explicit operator bool() const
    {
        return owner != nullptr;
    }

    ByteView data() const
    {
        return mapping.subspan(offset, bytes);
    }

    template <class T> std::span<const T> as() const
    {
        if (bytes % sizeof(T) || reinterpret_cast<std::uintptr_t>(data().data()) % alignof(T))
        {
            throw Error("unaligned weight tensor");
        }
        return {reinterpret_cast<const T*>(data().data()), bytes / sizeof(T)};
    }
};

template <class T> class Array
{
public:
    Array() = default;

    explicit Array(View view) : owner_(view.owner), values_(view.as<T>()), view_(std::move(view))
    {
    }

    explicit Array(const TensorView<T>& tensor) : Array(View(tensor))
    {
    }

    explicit Array(std::vector<T> values)
    {
        auto owner = std::make_shared<const std::vector<T>>(std::move(values));
        values_ = *owner;
        owner_ = std::move(owner);
    }

    auto data() const
    {
        return values_.data();
    }

    auto size() const
    {
        return values_.size();
    }

    auto begin() const
    {
        return values_.begin();
    }

    auto end() const
    {
        return values_.end();
    }

    auto& operator[](std::size_t i) const
    {
        return values_[i];
    }

    const View& view() const
    {
        return view_;
    }

    Array& operator=(std::vector<T> values)
    {
        *this = Array(std::move(values));
        return *this;
    }

    template <class It> void assign(It first, It last)
    {
        *this = std::vector<T>(first, last);
    }

    operator std::span<const T>() const
    {
        return values_;
    }

private:
    std::shared_ptr<const void> owner_;
    std::span<const T> values_;
    View view_;
};
} // namespace tsac::weights
