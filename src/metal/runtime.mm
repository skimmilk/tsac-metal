#include "metal/runtime.hpp"

#include "common/internal.hpp"
#include "metal/abi/kernel_contracts.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#import <dispatch/dispatch.h>
#include <mutex>
#include <optional>
#include <unistd.h>
#include <unordered_map>

namespace tsac::metal
{
namespace
{
struct Context
{
    id<MTLDevice> device;
    id<MTL4Compiler> compiler API_AVAILABLE(macos(26.0));

    Context()
    {
        @autoreleasepool
        {
            device = MTLCreateSystemDefaultDevice();
            if (@available(macOS 26.0, *))
            {
                if (device && device.hasUnifiedMemory && [device supportsFamily:MTLGPUFamilyMetal4])
                {
                    compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new]
                                                           error:nil];
                }
            }
        }
    }
};

Context& context()
{
    static Context instance;
    return instance;
}

std::string message(NSError* error, const char* fallback)
{
    const char* text = error ? error.localizedDescription.UTF8String : nullptr;
    return text ? std::string(text) : std::string(fallback);
}

NSString* string(std::string_view text)
{
    NSString* result = [[NSString alloc] initWithBytes:text.data()
                                                length:text.size()
                                              encoding:NSUTF8StringEncoding];
    detail::require(result != nil, "Metal function name is not UTF-8");
    return result;
}
} // namespace

struct Buffer::Impl
{
    id<MTLBuffer> buffer;
    std::size_t bytes;
    BufferStorage storage;
    bool read_only = false;
};

struct Pipeline::Impl
{
    id<MTLComputePipelineState> pipeline;
    std::string kernel;
};

struct Program::Impl
{
    id<MTLLibrary> library;
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines;
    mutable std::mutex mutex;
    std::string label;
    bool profile = std::getenv("TSAC_METAL_PROFILE") != nullptr;
    mutable std::atomic<std::uint64_t> submissions{0}, dispatches{0}, encoding_ns{0}, wait_ns{0},
        gpu_ns{0};

    ~Impl()
    {
        if (profile)
        {
            std::fprintf(stderr,
                         "Metal profile %s: submissions=%llu dispatches=%llu encode=%.6f wait=%.6f "
                         "gpu=%.6f seconds\n",
                         label.c_str(), static_cast<unsigned long long>(submissions.load()),
                         static_cast<unsigned long long>(dispatches.load()),
                         double(encoding_ns.load()) / 1e9, double(wait_ns.load()) / 1e9,
                         double(gpu_ns.load()) / 1e9);
        }
    }
};

bool available()
{
    if (@available(macOS 26.0, *))
    {
        return context().device != nil && context().compiler != nil;
    }
    return false;
}

Buffer::Buffer(std::shared_ptr<Impl> impl) : impl_(std::move(impl)), length_(impl_->bytes)
{
}

Buffer Buffer::allocate(std::size_t bytes, BufferStorage storage)
{
    @autoreleasepool
    {
        detail::require(available(), "Metal device is unavailable");
        detail::require(bytes <= context().device.maxBufferLength,
                        "Metal buffer exceeds device limit");
        auto impl = std::make_shared<Impl>();
        impl->bytes = bytes;
        impl->storage = storage;
        impl->buffer = [context().device
            newBufferWithLength:std::max<std::size_t>(1, bytes)
                        options:storage == BufferStorage::shared ? MTLResourceStorageModeShared
                                                                 : MTLResourceStorageModePrivate];
        detail::require(impl->buffer != nil, "cannot allocate Metal buffer");
        return Buffer(std::move(impl));
    }
}

Buffer Buffer::copy(const void* data, std::size_t bytes)
{
    detail::require(data != nullptr || bytes == 0, "null Metal upload source");
    auto result = allocate(bytes);
    if (bytes)
    {
        std::memcpy(result.impl_->buffer.contents, data, bytes);
    }
    return result;
}

Buffer Buffer::import_weights(const weights::View& view)
{
    detail::require(available() && view && view.offset <= view.mapping.size() &&
                        view.bytes <= view.mapping.size() - view.offset,
                    "invalid mapped Metal weights");
    const auto page = static_cast<std::size_t>(getpagesize());
    detail::require(reinterpret_cast<std::uintptr_t>(view.mapping.data()) % page == 0 &&
                        view.mapping.size() % page == 0 &&
                        view.mapping.size() <= context().device.maxBufferLength,
                    "mapped Metal weight region exceeds device limits");
    static std::mutex import_mutex;
    static std::unordered_map<const void*, std::weak_ptr<Impl>> imports;
    std::lock_guard lock(import_mutex);
    if (const auto found = imports.find(view.mapping.data()); found != imports.end())
    {
        if (auto existing = found->second.lock())
        {
            return Buffer(std::move(existing)).slice(view.offset, view.bytes);
        }
    }
    for (auto it = imports.begin(); it != imports.end();)
    {
        if (it->second.expired())
        {
            it = imports.erase(it);
        }
        else
        {
            ++it;
        }
    }
    auto impl = std::make_shared<Impl>();
    impl->bytes = view.mapping.size();
    impl->storage = BufferStorage::shared;
    impl->read_only = true;
    const auto owner = view.owner;
    impl->buffer =
        [context().device newBufferWithBytesNoCopy:const_cast<std::uint8_t*>(view.mapping.data())
                                            length:view.mapping.size()
                                           options:MTLResourceStorageModeShared
                                       deallocator:^(void*, NSUInteger) {
                                         (void)owner;
                                       }];
    detail::require(impl->buffer != nil, "cannot import mapped Metal weights");
    imports[view.mapping.data()] = impl;
    return Buffer(std::move(impl)).slice(view.offset, view.bytes);
}

void Buffer::read(void* destination, std::size_t bytes) const
{
    detail::require(impl_ && bytes <= length_ && (destination != nullptr || bytes == 0),
                    "Metal read exceeds buffer dimensions");
    if (bytes)
    {
        detail::require(impl_->storage == BufferStorage::shared,
                        "CPU read requires shared storage");
        std::memcpy(destination, static_cast<const std::byte*>(impl_->buffer.contents) + offset_,
                    bytes);
    }
}

std::span<const std::byte> Buffer::read_only_bytes() const
{
    detail::require(impl_ && impl_->storage == BufferStorage::shared,
                    "CPU read requires shared Metal storage");
    return {static_cast<const std::byte*>(impl_->buffer.contents) + offset_, length_};
}

void Buffer::write(const void* source, std::size_t bytes) const
{
    detail::require(impl_ && bytes <= length_ && (source || !bytes),
                    "Metal write exceeds buffer dimensions");
    if (bytes)
    {
        std::memcpy(mapped_bytes().data(), source, bytes);
    }
}

Buffer Buffer::slice(std::size_t offset, std::size_t bytes) const
{
    detail::require(offset <= length_ && bytes <= length_ - offset,
                    "Metal slice exceeds buffer dimensions");
    auto result = *this;
    result.offset_ += offset;
    result.length_ = bytes;
    return result;
}

bool Buffer::shares_storage(const Buffer& other) const
{
    return impl_ == other.impl_;
}

std::size_t Buffer::size() const
{
    return length_;
}

std::span<std::byte> Buffer::mapped_bytes() const
{
    detail::require(impl_ && impl_->storage == BufferStorage::shared && !impl_->read_only,
                    "CPU access requires a shared Metal buffer");
    return {static_cast<std::byte*>(impl_->buffer.contents) + offset_, length_};
}

struct ResidencySet::Impl
{
    id<MTLResidencySet> set;
    NSSet<id<MTLBuffer>>* allocations;
};

ResidencySet::ResidencySet(std::shared_ptr<Impl> impl) : impl_(std::move(impl))
{
}

std::shared_ptr<const ResidencySet> ResidencySet::create(std::span<const Buffer> buffers)
{
    @autoreleasepool
    {
        detail::require(available(), "Metal 4 residency requires a supported device");
        NSMutableSet<id<MTLBuffer>>* allocations = [NSMutableSet new];
        for (const auto& buffer : buffers)
        {
            detail::require(buffer.impl_ != nullptr, "Cannot register an empty Metal buffer");
            [allocations addObject:buffer.impl_->buffer];
        }
        MTLResidencySetDescriptor* descriptor = [MTLResidencySetDescriptor new];
        descriptor.initialCapacity = allocations.count;
        NSError* error = nil;
        auto impl = std::make_shared<Impl>();
        impl->set = [context().device newResidencySetWithDescriptor:descriptor error:&error];
        detail::require(impl->set != nil, "Cannot create Metal residency set");
        impl->allocations = [allocations copy];
        for (id<MTLBuffer> allocation in allocations)
        {
            [impl->set addAllocation:allocation];
        }
        [impl->set commit];
        return std::shared_ptr<const ResidencySet>(new ResidencySet(std::move(impl)));
    }
}

struct ExecutionContext::Impl
{
    explicit Impl(const Program& source) : program(source)
    {
    }

    Program program;
    std::shared_ptr<const ResidencySet> model, session;
    id<MTL4CommandQueue> queue API_AVAILABLE(macos(26.0));
    id<MTL4CommandAllocator> allocator API_AVAILABLE(macos(26.0));
    id<MTL4CommandBuffer> command API_AVAILABLE(macos(26.0));
    id<MTL4ArgumentTable> table API_AVAILABLE(macos(26.0));
    id<MTLResidencySet> transient;
    NSSet<id<MTLBuffer>>* transient_allocations;
    std::optional<Buffer> parameters;
    dispatch_semaphore_t completion = dispatch_semaphore_create(0);
    bool running = false;
};

ExecutionContext::ExecutionContext(const Program& program,
                                   std::shared_ptr<const ResidencySet> model)
    : impl_(std::make_unique<Impl>(program))
{
    @autoreleasepool
    {
        if (@available(macOS 26.0, *))
        {
            detail::require(available(), "Metal 4 requires macOS 26 and a supported Apple GPU");
            impl_->model = std::move(model);
            impl_->queue = [context().device newMTL4CommandQueue];
            impl_->allocator = [context().device newCommandAllocator];
            impl_->command = [context().device newCommandBuffer];
            detail::require(impl_->queue && impl_->allocator && impl_->command,
                            "Cannot allocate Metal 4 execution context");
            if (impl_->model)
            {
                [impl_->queue addResidencySet:impl_->model->impl_->set];
            }
            MTL4ArgumentTableDescriptor* descriptor = [MTL4ArgumentTableDescriptor new];
            descriptor.maxBufferBindCount = 31;
            descriptor.initializeBindings = YES;
            NSError* error = nil;
            impl_->table = [context().device newArgumentTableWithDescriptor:descriptor
                                                                      error:&error];
            detail::require(impl_->table != nil, "Cannot create Metal 4 argument table");
            MTLResidencySetDescriptor* residency = [MTLResidencySetDescriptor new];
            impl_->transient = [context().device newResidencySetWithDescriptor:residency
                                                                         error:&error];
            detail::require(impl_->transient != nil, "Cannot create transient Metal residency set");
            impl_->transient_allocations = [NSSet new];
        }
        else
        {
            throw Error("Metal 4 requires macOS 26 or later");
        }
    }
}

ExecutionContext::~ExecutionContext() = default;

void ExecutionContext::set_session_resources(std::shared_ptr<const ResidencySet> resources)
{
    detail::require(!impl_->running, "Cannot replace in-flight Metal session resources");
    // Completion is synchronous. Drop the previous stream's transient backing
    // references before releasing its persistent cache residency.
    [impl_->transient removeAllAllocations];
    [impl_->transient commit];
    impl_->transient_allocations = [NSSet new];
    if (@available(macOS 26.0, *))
    {
        if (resources != impl_->session)
        {
            if (impl_->session)
            {
                [impl_->queue removeResidencySet:impl_->session->impl_->set];
            }
            if (resources)
            {
                [impl_->queue addResidencySet:resources->impl_->set];
            }
        }
    }
    impl_->session = std::move(resources);
}

Program::Program(std::shared_ptr<Impl> impl) : impl_(std::move(impl))
{
}

Program Program::load(std::span<const std::byte> library, std::string_view label)
{
    @autoreleasepool
    {
        detail::require(available(), "Metal device is unavailable");
        detail::require(!library.empty(), "Metal library is empty");
        dispatch_data_t data = dispatch_data_create(library.data(), library.size(), nullptr,
                                                    DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        detail::require(data != nullptr, "Cannot create Metal library data");
        NSError* error = nil;
        auto impl = std::make_shared<Impl>();
        impl->library = [context().device newLibraryWithData:data error:&error];
        if (!impl->library)
        {
            throw Error("Metal library loading failed [" + std::string(label) +
                        "]: " + message(error, "unknown library error"));
        }
        // Preserve diagnostics even if Metal returns a valid library with them.
        if (error)
        {
            std::fprintf(stderr, "Metal library diagnostics [%s]:\n%s\n",
                         std::string(label).c_str(),
                         message(error, "unknown library diagnostic").c_str());
        }
        impl->pipelines = [NSMutableDictionary new];
        impl->label = label;
        return Program(std::move(impl));
    }
}

Pipeline Program::resolve(std::string_view name) const
{
    @autoreleasepool
    {
        if (@available(macOS 26.0, *))
        {
            detail::require(available(), "Metal 4 requires macOS 26 and a supported Apple GPU");
            std::lock_guard lock(impl_->mutex);
            NSString* key = string(name);
            id<MTLComputePipelineState> pipeline = impl_->pipelines[key];
            if (!pipeline)
            {
                MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
                function.library = impl_->library;
                function.name = key;
                MTL4ComputePipelineDescriptor* descriptor = [MTL4ComputePipelineDescriptor new];
                descriptor.computeFunctionDescriptor = function;
                NSError* error = nil;
                pipeline = [context().compiler newComputePipelineStateWithDescriptor:descriptor
                                                                 compilerTaskOptions:nil
                                                                               error:&error];
                if (!pipeline)
                {
                    throw Error("Metal 4 pipeline failed [" + std::string(name) +
                                "]: " + message(error, "unknown pipeline error"));
                }
                detail::require(pipeline.threadExecutionWidth == 32,
                                "Metal shaders require 32-thread SIMD groups");
                impl_->pipelines[key] = pipeline;
            }
            Pipeline result;
            result.impl_ = std::make_shared<Pipeline::Impl>();
            result.impl_->pipeline = pipeline;
            result.impl_->kernel = name;
            return result;
        }
        else
        {
            throw Error("Metal 4 requires macOS 26 or later");
        }
    }
}

void ExecutionContext::run(std::span<const Program::Dispatch> dispatches)
{
    @autoreleasepool
    {
        if (@available(macOS 26.0, *))
        {
            detail::require(!impl_->running, "Metal execution context has an exclusive owner");
            const auto& program = impl_->program;
            detail::require(program.impl_ != nullptr, "Metal program is not initialized");
            auto& stats = *program.impl_;
            detail::require(available(), "Metal 4 requires macOS 26 and a supported Apple GPU");
            using Clock = std::chrono::steady_clock;
            const auto started = stats.profile ? Clock::now() : Clock::time_point{};
            // A conservative arena allocation policy, separate from the scalar wire layouts.
            constexpr std::size_t parameter_alignment = 256;
            std::size_t active = 0;
            for (const auto& dispatch : dispatches)
            {
                if (dispatch.threads)
                {
                    const auto* contract = abi::find_kernel_contract(dispatch.kernel);
                    detail::require(contract != nullptr, "Unknown Metal kernel ABI");
                    detail::require(!dispatch.parameters.empty() &&
                                        dispatch.parameters.size() <= parameter_alignment &&
                                        dispatch.parameters.size() == contract->parameter_bytes,
                                    "Invalid Metal parameter block");
                    ++active;
                }
            }
            if (!active)
            {
                return;
            }
            const auto bytes = checked_mul(active, parameter_alignment);
            if (!impl_->parameters || impl_->parameters->size() < bytes)
            {
                impl_->parameters = Buffer::allocate(std::bit_ceil(bytes));
            }
            const auto& arena = *impl_->parameters;
            // Keep every backing allocation alive, including allocations reached through views.
            NSMutableSet<id<MTLBuffer>>* allocations = [NSMutableSet new];
            [allocations addObject:arena.impl_->buffer];
            for (const auto& dispatch : dispatches)
            {
                if (!dispatch.threads)
                {
                    continue;
                }
                std::uint32_t mask = 1u << abi::parameter_slot;
                for (const auto& argument : dispatch.arguments)
                {
                    detail::require(argument.index > abi::parameter_slot && argument.index < 31,
                                    "Invalid Metal resource binding slot");
                    detail::require(!(mask & (1u << argument.index)),
                                    "Duplicate Metal resource binding");
                    mask |= 1u << argument.index;
                    detail::require(argument.buffer.impl_ != nullptr,
                                    "Metal kernel received an empty buffer");
                    [allocations addObject:argument.buffer.impl_->buffer];
                }
                const auto* contract = abi::find_kernel_contract(dispatch.kernel);
                detail::require(mask == (contract->resource_mask | 1u),
                                "Metal kernel resource bindings do not match its ABI");
            }
            // Persistent model/cache allocations are already registered. Keep
            // only submission-specific resources in the reusable transient set.
            if (impl_->model)
            {
                [allocations minusSet:impl_->model->impl_->allocations];
            }
            if (impl_->session)
            {
                [allocations minusSet:impl_->session->impl_->allocations];
            }
            bool residency_changed = false;
            for (id<MTLBuffer> allocation in impl_->transient_allocations)
            {
                if (![allocations containsObject:allocation])
                {
                    [impl_->transient removeAllocation:allocation];
                    residency_changed = true;
                }
            }
            for (id<MTLBuffer> allocation in allocations)
            {
                if (![impl_->transient_allocations containsObject:allocation])
                {
                    [impl_->transient addAllocation:allocation];
                    residency_changed = true;
                }
            }
            if (residency_changed)
            {
                [impl_->transient commit];
                impl_->transient_allocations = [allocations copy];
            }
            // Resolve and validate every dispatch before opening the command
            // buffer, so invalid work cannot leave a half-recorded context.
            std::vector<Pipeline> pipelines;
            pipelines.reserve(active);
            for (const auto& dispatch : dispatches)
            {
                if (!dispatch.threads)
                {
                    continue;
                }
                pipelines.push_back(dispatch.prepared ? *dispatch.prepared
                                                      : program.resolve(dispatch.kernel));
                const auto& handle = pipelines.back();
                detail::require(handle.impl_ && handle.impl_->kernel == dispatch.kernel,
                                "Prepared Metal pipeline does not match the dispatch ABI");
                const auto width =
                    dispatch.threadgroup_width
                        ? dispatch.threadgroup_width
                        : std::min<NSUInteger>(handle.impl_->pipeline.maxTotalThreadsPerThreadgroup,
                                               256);
                detail::require(width &&
                                    width <= handle.impl_->pipeline.maxTotalThreadsPerThreadgroup &&
                                    (!dispatch.threadgroup_width || dispatch.threads % width == 0),
                                "Invalid Metal threadgroup width or incomplete barrier group");
                const auto& groups = dispatch.groups;
                if (groups[0])
                {
                    detail::require(dispatch.threadgroup_width && groups[1] && groups[2] &&
                                        groups[0] <= dispatch.threads / width &&
                                        dispatch.threads / width % groups[0] == 0 &&
                                        groups[1] <= dispatch.threads / width / groups[0] &&
                                        dispatch.threads / width / groups[0] % groups[1] == 0 &&
                                        groups[2] ==
                                            dispatch.threads / width / groups[0] / groups[1],
                                    "Metal threadgroup grid does not match thread count");
                }
                else
                {
                    detail::require(!groups[1] && !groups[2], "Incomplete Metal threadgroup grid");
                }
            }
            // Previous run has completed before returning, including GPU errors.
            [impl_->allocator reset];
            const auto command = impl_->command;
            const auto table = impl_->table;
            [command beginCommandBufferWithAllocator:impl_->allocator];
            [command useResidencySet:impl_->transient];
            id<MTL4ComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!encoder)
            {
                [command endCommandBuffer];
                throw Error("Cannot allocate Metal 4 compute encoder");
            }
            [encoder setArgumentTable:table];
            [encoder barrierAfterQueueStages:MTLStageDispatch
                                beforeStages:MTLStageDispatch
                           visibilityOptions:MTL4VisibilityOptionDevice];
            std::size_t parameter_index = 0;
            for (const auto& dispatch : dispatches)
            {
                const auto threads = dispatch.threads,
                           threadgroup_width = dispatch.threadgroup_width;
                if (!threads)
                {
                    continue;
                }
                const auto& handle = pipelines[parameter_index];
                id<MTLComputePipelineState> pipeline = handle.impl_->pipeline;
                const auto width =
                    threadgroup_width
                        ? threadgroup_width
                        : std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
                if (parameter_index)
                {
                    [encoder barrierAfterEncoderStages:MTLStageDispatch
                                   beforeEncoderStages:MTLStageDispatch
                                     visibilityOptions:MTL4VisibilityOptionDevice];
                }
                const auto offset = parameter_index++ * parameter_alignment;
                arena.slice(offset, dispatch.parameters.size())
                    .write(dispatch.parameters.data(), dispatch.parameters.size());
                [table setAddress:arena.impl_->buffer.gpuAddress + offset
                          atIndex:abi::parameter_slot];
                for (const auto& argument : dispatch.arguments)
                {
                    [table setAddress:argument.buffer.impl_->buffer.gpuAddress +
                                      argument.buffer.offset_
                              atIndex:argument.index];
                }
                [encoder setComputePipelineState:pipeline];
                if (dispatch.groups[0])
                {
                    const auto& groups = dispatch.groups;
                    [encoder dispatchThreadgroups:MTLSizeMake(groups[0], groups[1], groups[2])
                            threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
                }
                else
                {
                    [encoder dispatchThreads:MTLSizeMake(threads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
                }
            }
            [encoder endEncoding];
            [command endCommandBuffer];
            const auto completion = impl_->completion;
            __block id<MTL4CommitFeedback> feedback = nil;
            MTL4CommitOptions* options = [MTL4CommitOptions new];
            [options addFeedbackHandler:^(id<MTL4CommitFeedback> value) {
              feedback = value;
              dispatch_semaphore_signal(completion);
            }];
            const id<MTL4CommandBuffer> commands[] = {command};
            impl_->running = true;
            [impl_->queue commit:commands count:1 options:options];
            const auto committed = stats.profile ? Clock::now() : Clock::time_point{};
            dispatch_semaphore_wait(completion, DISPATCH_TIME_FOREVER);
            impl_->running = false;
            if (stats.profile)
            {
                ++stats.submissions;
                stats.dispatches += active;
                stats.encoding_ns += static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(committed - started)
                        .count());
                stats.wait_ns += static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - committed)
                        .count());
                stats.gpu_ns +=
                    static_cast<std::uint64_t>((feedback.GPUEndTime - feedback.GPUStartTime) * 1e9);
            }
            if (feedback.error)
            {
                throw Error("Metal 4 command failed: " +
                            message(feedback.error, "unknown command error"));
            }
        }
        else
        {
            throw Error("Metal 4 requires macOS 26 or later");
        }
    }
}
} // namespace tsac::metal
