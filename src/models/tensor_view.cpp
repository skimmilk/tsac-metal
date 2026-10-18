#include "models/tensor_view.hpp"

#include "models/contract.hpp"

namespace tsac
{
std::size_t TensorShape::rank() const
{
    std::size_t n = axes.size();
    while (n && axes[n - 1] == 0)
    {
        --n;
    }
    return n;
}

std::size_t TensorShape::elements() const
{
    const auto n = rank();
    if (!n)
    {
        throw Error("empty tensor shape");
    }
    std::size_t result = 1;
    for (std::size_t i = 0; i < n; ++i)
    {
        if (!axes[i])
        {
            throw Error("tensor shape has an empty axis");
        }
        result = checked_mul(result, axes[i]);
    }
    return result;
}

EncodedTensorView::EncodedTensorView(std::shared_ptr<const MappedRegion> region, std::size_t offset,
                                     std::size_t bytes, TensorShape shape, TensorEncoding encoding)
    : region_(std::move(region)), offset_(offset), bytes_(bytes), shape_(shape), encoding_(encoding)
{
    if (!region_ || offset_ > region_->bytes().size() ||
        bytes_ > region_->bytes().size() - offset_ ||
        bytes_ != model_contract::encoded_bytes(shape_, encoding_))
    {
        throw Error("encoded tensor disagrees with contract or mapping");
    }
}

ByteView EncodedTensorView::payload() const&
{
    return region_->bytes().subspan(offset_, bytes_);
}

std::size_t EncodedTensorView::data_plane_bytes() const
{
    if (encoding_ != TensorEncoding::bf8)
    {
        return bytes_;
    }
    return checked_add(shape_.elements(), 255) / 256 * 256;
}
} // namespace tsac
