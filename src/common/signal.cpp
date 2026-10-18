#include "common/signal.hpp"

#include "common/internal.hpp"

namespace tsac
{
std::size_t SignalShape::elements(const Limits& limits) const
{
    detail::require(channels && frames && groups, "empty signal dimension");
    const auto count = checked_mul(checked_mul(channels, frames), groups);
    detail::require(count <= limits.tensor_elements, "signal exceeds element limit");
    return count;
}

} // namespace tsac
