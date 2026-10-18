#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <cstddef>
#include <cstdint>
#include <type_traits>
#endif

namespace tsac_metal_abi
{
#ifdef __METAL_VERSION__
using u32 = uint;
#else
using u32 = std::uint32_t;
template <class T> inline constexpr bool is_parameter_block = false;
#endif

// Slot zero is reserved for immutable per-dispatch parameters in every kernel.
enum : u32
{
    parameter_slot = 0,
    nonfinite_error = 1,
    range_interval_error = 2
};
} // namespace tsac_metal_abi
