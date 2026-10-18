#include "common/common.hpp"
#include "common/numeric/bf16.hpp"
#include "metal/abi/bindings.h"
#include "metal/probability/parameters.h"
#include "metal/runtime.hpp"

#include <array>
#include <iostream>

using namespace tsac;
namespace abi = tsac_metal_abi;

void check(bool condition, const char* message)
{
    if (!condition)
    {
        throw Error(message);
    }
}

void dispatch_constants_and_views(metal::ExecutionContext& execution)
{
    constexpr std::size_t count = 32;
    std::array<abi::AddParams, count> parameters{};
    std::vector<std::vector<metal::ResourceBinding>> bindings;
    std::vector<metal::Buffer> outputs;
    std::array<std::uint16_t, 36> values{};
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        values[i] = probability_bf16(static_cast<float>(i));
    }
    // The view remains the owner after the original allocation handle goes away.
    const auto input = metal::Buffer::copy(values.data(), sizeof values).slice(4, 64);
    const std::array right_values{probability_bf16(1), probability_bf16(2), probability_bf16(3),
                                  probability_bf16(4)};
    const auto right = metal::Buffer::copy(right_values.data(), sizeof right_values);
    const std::array<std::uint16_t, count> sentinel{};
    for (std::size_t i = 0; i < count; ++i)
    {
        parameters[i] = {static_cast<std::uint32_t>(i + 1), static_cast<std::uint32_t>(i % 4 + 1)};
        outputs.push_back(metal::Buffer::copy(sentinel.data(), sizeof sentinel));
        // Resource order intentionally differs from slot order.
        bindings.push_back({metal::bind(abi::add_binding::output, outputs.back()),
                            metal::bind(abi::add_binding::right, right),
                            metal::bind(abi::add_binding::input, input)});
    }
    std::vector<metal::Program::Dispatch> dispatches;
    dispatches.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        dispatches.push_back(
            {"add_kernel", bindings[i], count, 0, std::as_bytes(std::span(&parameters[i], 1))});
    }
    execution.run(dispatches);
    for (std::size_t i = 0; i < count; ++i)
    {
        std::array<std::uint16_t, count> actual{};
        outputs[i].read(actual.data(), sizeof actual);
        for (std::size_t j = 0; j < count; ++j)
        {
            const auto expected = j < parameters[i].element_count
                                      ? probability_bf16(static_cast<float>(
                                            j + 2 + j % parameters[i].right_period_elements + 1))
                                      : 0;
            check(actual[j] == expected, "Metal 4 parameter snapshots or buffer view changed");
        }
    }
}

void dependency_and_recovery(metal::ExecutionContext& execution)
{
    const std::array values{probability_bf16(1), probability_bf16(2), probability_bf16(3),
                            probability_bf16(4)};
    const auto input = metal::Buffer::copy(values.data(), sizeof values);
    const auto intermediate = metal::Buffer::allocate(sizeof values);
    const auto output = metal::Buffer::allocate(sizeof values);
    const abi::CopyParams copy{4};
    const abi::AddParams add{4, 4};
    const std::array copy_bindings{metal::bind(abi::copy_binding::input, input),
                                   metal::bind(abi::copy_binding::output, intermediate)};
    const std::array add_bindings{metal::bind(abi::add_binding::input, intermediate),
                                  metal::bind(abi::add_binding::right, input),
                                  metal::bind(abi::add_binding::output, output)};
    for (const auto bad_slot : {0u, 31u, static_cast<unsigned>(abi::copy_binding::input)})
    {
        const std::array invalid{copy_bindings[0], metal::bind(bad_slot, output)};
        bool rejected = false;
        try
        {
            const metal::Program::Dispatch dispatch{"copy_bf16", invalid, 4, 0,
                                                    std::as_bytes(std::span(&copy, 1))};
            execution.run(std::span(&dispatch, 1));
        }
        catch (const Error&)
        {
            rejected = true;
        }
        check(rejected, "Invalid or duplicate argument-table slot was accepted");
    }
    const std::array<metal::Program::Dispatch, 2> dispatches{
        {{"copy_bf16", copy_bindings, 4, 0, std::as_bytes(std::span(&copy, 1))},
         {"add_kernel", add_bindings, 4, 0, std::as_bytes(std::span(&add, 1))}}};
    for (const auto& invalid :
         {metal::Program::Dispatch{"copy_bf16", copy_bindings, 4, 0,
                                   std::as_bytes(std::span(&add, 1))},
          metal::Program::Dispatch{"copy_bf16", std::span(copy_bindings).first(1), 4, 0,
                                   std::as_bytes(std::span(&copy, 1))}})
    {
        bool rejected = false;
        try
        {
            execution.run(std::span(&invalid, 1));
        }
        catch (const Error&)
        {
            rejected = true;
        }
        check(rejected, "Malformed Metal parameter block or missing resource was accepted");
    }
    // A context without model residency still attaches session allocations to
    // its own queue, excluding them from the transient set.
    const std::array session_buffers{input};
    execution.set_session_resources(metal::ResidencySet::create(session_buffers));
    // Repeat to exercise residency and submission lifetimes across completions.
    for (unsigned iteration = 0; iteration < 8; ++iteration)
    {
        execution.run(dispatches);
        std::array<std::uint16_t, 4> actual{};
        output.read(actual.data(), sizeof actual);
        for (std::size_t i = 0; i < actual.size(); ++i)
        {
            check(actual[i] == probability_bf16(static_cast<float>((i + 1) * 2)),
                  "Dependent dispatch or recovery after invalid bindings failed");
        }
    }
    execution.set_session_resources({});
    // Empty work must not inspect parameters or unbound resources.
    const metal::Program::Dispatch empty{"copy_bf16", {}, 0};
    execution.run(std::span(&empty, 1));
}

void shared_cpu_gpu_views(metal::ExecutionContext& execution)
{
    auto backing = metal::Buffer::allocate(12);
    const auto input = backing.slice(2, 8);
    const auto output = metal::Buffer::allocate(8);
    check(input.shares_storage(backing) && !input.shares_storage(output),
          "Metal slice lost backing identity");
    const auto view = input.mapped<std::uint16_t>();
    check(view.data() == reinterpret_cast<const std::uint16_t*>(backing.mapped_bytes().data() + 2),
          "Mapped Metal view copied its storage");
    const abi::AddParams p{4, 4};
    const std::array bindings{metal::bind(abi::add_binding::input, input),
                              metal::bind(abi::add_binding::right, input),
                              metal::bind(abi::add_binding::output, output)};
    const metal::Program::Dispatch dispatch{"add_kernel", bindings, 4, 0,
                                            std::as_bytes(std::span(&p, 1))};
    for (unsigned iteration = 1; iteration <= 4; ++iteration)
    {
        auto* cpu = reinterpret_cast<std::uint16_t*>(input.mapped_bytes().data());
        for (unsigned i = 0; i < 4; ++i)
        {
            cpu[i] = probability_bf16(float(iteration + i));
        }
        execution.run(std::span(&dispatch, 1));
        const auto actual = output.mapped<std::uint16_t>();
        for (unsigned i = 0; i < 4; ++i)
        {
            check(view[i] == cpu[i] && actual[i] == probability_bf16(float(2 * (iteration + i))),
                  "Shared CPU/GPU contents were not visible after completion");
        }
    }
    for (const auto& invalid : {backing.slice(1, 4), backing.slice(0, 3),
                                metal::Buffer::allocate(4, metal::BufferStorage::gpu_only)})
    {
        bool rejected = false;
        try
        {
            (void)invalid.mapped<std::uint16_t>();
        }
        catch (const Error&)
        {
            rejected = true;
        }
        check(rejected, "Invalid CPU-visible typed buffer view was accepted");
    }
}

int main(int argc, char** argv)
{
    try
    {
        check(argc == 2, "Expected probability.metallib path");
        const auto bytes = read_file(argv[1]);
        const auto program = metal::Program::load(std::as_bytes(std::span(bytes)), "ABI tests");
        metal::ExecutionContext execution(program);
        dispatch_constants_and_views(execution);
        dependency_and_recovery(execution);
        shared_cpu_gpu_views(execution);
        std::cout << "Metal 4 ABI checks passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
