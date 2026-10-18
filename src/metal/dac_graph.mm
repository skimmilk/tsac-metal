#include "metal/dac_graph.hpp"

#include "common/numeric.hpp"
#include "dac_metallib.hpp"
#include "metal/abi/bindings.h"
#include "metal/convolution.hpp"
#include "metal/dac/parameters.h"
#include "metal/runtime.hpp"
#include "weights/normalization.hpp"

#include <array>
#include <limits>
#include <unordered_map>

namespace tsac
{
namespace
{
namespace abi = ::tsac_metal_abi;
using metal::bind;

const metal::Program& program()
{
    static const auto compiled =
        metal::Program::load(std::as_bytes(std::span(metal::embedded::dac)), "dac");
    return compiled;
}

std::uint32_t dimension(std::size_t value)
{
    detail::require(value <= std::numeric_limits<std::uint32_t>::max(),
                    "Metal DAC dimension exceeds uint32");
    return static_cast<std::uint32_t>(value);
}

template <class T> metal::Buffer upload(std::span<const T> values)
{
    return metal::Buffer::copy(values.data(), checked_mul(values.size(), sizeof(T)));
}

} // namespace

struct MetalConvolution::Impl
{
    Conv1dConfig config;
    metal::Buffer weights, bias;

    Impl(Conv1dConfig value, metal::Buffer w, metal::Buffer b)
        : config(value), weights(std::move(w)), bias(std::move(b))
    {
    }
};

MetalConvolution::MetalConvolution(std::shared_ptr<const Impl> impl) : impl_(std::move(impl))
{
}

MetalConvolution MetalConvolution::from_prepared(Conv1dConfig config, const weights::View& weights,
                                                 const weights::View& bias)
{
    detail::require(weights.bytes ==
                            checked_mul(checked_mul(config.input_channels, config.output_channels),
                                        config.kernel) *
                                sizeof(float) &&
                        bias.bytes == config.output_channels * sizeof(float),
                    "invalid mapped convolution");
    (void)program();
    return MetalConvolution(std::make_shared<Impl>(config, metal::Buffer::import_weights(weights),
                                                   metal::Buffer::import_weights(bias)));
}

MetalConvolution MetalConvolution::prepare(Conv1dConfig config, std::span<const float> weights,
                                           std::span<const float> bias)
{
    detail::require(config.input_channels && config.output_channels && config.kernel &&
                        config.stride && config.dilation,
                    "empty Metal convolution dimension");
    detail::require(config.kind == ConvolutionKind::ordinary ||
                        config.kind == ConvolutionKind::transposed,
                    "invalid Metal convolution kind");
    const auto count =
        checked_mul(checked_mul(config.input_channels, config.output_channels), config.kernel);
    detail::require(weights.size() == count && bias.size() == config.output_channels,
                    "Metal convolution weight or bias shape mismatch");
    detail::require_finite(weights, "nonfinite Metal convolution weight");
    detail::require_finite(bias, "nonfinite Metal convolution bias");
    dimension(count);
    dimension(checked_add(checked_mul(config.kernel - 1, config.dilation), 1));
    dimension(checked_mul(config.padding, 2));
    // Consecutive output channels share each input scalar. Packing once makes
    // adjacent GPU threads read adjacent weights, with the same tap/input FMA
    // sequence and bias-last boundary as the original layout.
    auto packed = metal::Buffer::allocate(checked_mul(count, sizeof(float)));
    auto* packed_values = reinterpret_cast<float*>(packed.mapped_bytes().data());
    if (config.kind == ConvolutionKind::ordinary)
    {
        const auto rows = config.input_channels * config.kernel;
        // Cache-block the transpose. The original out-major weights can be
        // tens of MiB; visiting every output for each scalar input thrashes
        // cache lines and translations during model preparation.
        for (std::size_t out_base = 0; out_base < config.output_channels; out_base += 32)
        {
            const auto out_end = std::min(config.output_channels, out_base + 32);
            for (std::size_t in_base = 0; in_base < rows; in_base += 32)
            {
                const auto in_end = std::min(rows, in_base + 32);
                for (std::size_t in = in_base; in < in_end; ++in)
                {
                    for (std::size_t out = out_base; out < out_end; ++out)
                    {
                        packed_values[out + config.output_channels * in] = weights[in + rows * out];
                    }
                }
            }
        }
    }
    else
    {
        for (std::size_t tap = 0; tap < config.kernel; ++tap)
        {
            for (std::size_t in = 0; in < config.input_channels; ++in)
            {
                std::copy_n(weights.data() + config.output_channels * (tap + config.kernel * in),
                            config.output_channels,
                            packed_values +
                                config.output_channels * (in + config.input_channels * tap));
            }
        }
    }
    (void)program(); // Prepare library loading errors before retaining selected weights.
    return MetalConvolution(std::make_shared<Impl>(config, std::move(packed), upload(bias)));
}

namespace metal
{
struct DacGraph::Impl
{
    struct Op
    {
        std::string_view name;
        std::vector<ResourceBinding> buffers;
        std::size_t threads, width;
        std::vector<std::byte> constants;
        std::array<std::size_t, 3> groups = {};
        Pipeline pipeline;
    };

    ExecutionContext execution{program()};
    std::vector<Op> operations;
    std::unordered_map<std::string_view, Pipeline> pipelines;
    std::vector<Buffer> free;
    // Host writes must never alias scratch with pending GPU consumers.
    std::vector<Buffer> staging;
    std::size_t staging_cursor = 0;

    Buffer shared_input(std::size_t bytes)
    {
        const auto index = staging_cursor++;
        if (index == staging.size())
        {
            staging.push_back(Buffer::allocate(bytes));
        }
        else if (staging[index].size() < bytes)
        {
            staging[index] = Buffer::allocate(bytes);
        }
        return staging[index].slice(0, bytes);
    }

    Buffer error = Buffer::allocate(4);

    Buffer allocate(std::size_t bytes)
    {
        auto best = free.end();
        for (auto it = free.begin(); it != free.end(); ++it)
        {
            if (it->size() >= bytes && (best == free.end() || it->size() < best->size()))
            {
                best = it;
            }
        }
        if (best == free.end())
        {
            return Buffer::allocate(bytes);
        }
        auto result = *best;
        free.erase(best);
        return result;
    }

    template <class T>
    void record(std::string_view name, std::vector<ResourceBinding> args, std::size_t threads,
                const T& constants, std::size_t width = 0, std::array<std::size_t, 3> groups = {})
    {
        static_assert(abi::is_parameter_block<T>);
        auto bytes = std::as_bytes(std::span(&constants, 1));
        auto pipeline = pipelines.find(name);
        if (pipeline == pipelines.end())
        {
            pipeline = pipelines.emplace(name, program().resolve(name)).first;
        }
        operations.push_back({name,
                              std::move(args),
                              threads,
                              width,
                              {bytes.begin(), bytes.end()},
                              groups,
                              pipeline->second});
    }
};

DacGraph::DacGraph() : impl_(std::make_unique<Impl>())
{
    begin();
}

void DacGraph::begin()
{
    detail::require(impl_->operations.empty(), "Metal DAC chunk has not completed");
    impl_->staging_cursor = 0;
    const std::uint32_t zero = 0;
    impl_->error.write(&zero, sizeof zero);
}

DacGraph::~DacGraph() = default;

Buffer DacGraph::shared_temporary(std::size_t bytes)
{
    return impl_->shared_input(bytes);
}

DacTensor DacGraph::upload(std::span<const float> input, SignalShape shape, const Limits& limits)
{
    detail::require(input.size() == shape.elements(limits), "Metal graph input shape mismatch");
    detail::require_finite(input, "nonfinite Metal graph input");
    auto buffer = impl_->shared_input(input.size_bytes());
    buffer.write(input.data(), input.size_bytes());
    return {shape, std::move(buffer)};
}

DacTensor DacGraph::zeros(SignalShape shape, const Limits& limits)
{
    auto buffer = impl_->shared_input(checked_mul(shape.elements(limits), sizeof(float)));
    const auto bytes = buffer.mapped_bytes();
    std::fill(bytes.begin(), bytes.end(), std::byte{});
    return {shape, std::move(buffer)};
}

void DacGraph::release(const DacTensor& tensor)
{
    for (const auto& input : impl_->staging)
    {
        if (tensor.buffer.shares_storage(input))
        {
            return;
        }
    }
    impl_->free.push_back(tensor.buffer);
}

DacTensor DacGraph::convolution(const MetalConvolution& conv, const DacTensor& input,
                                const Limits& limits)
{
    const auto& c = conv.impl_->config;
    detail::require(input.shape.channels == c.input_channels,
                    "Metal convolution input shape mismatch");
    const auto span = checked_add(checked_mul(c.kernel - 1, c.dilation), 1);
    const auto padding = checked_mul(c.padding, 2);
    std::size_t frames;
    if (c.kind == ConvolutionKind::ordinary)
    {
        const auto padded = checked_add(input.shape.frames, padding);
        detail::require(padded >= span, "Metal convolution has no output frames");
        frames = (padded - span) / c.stride + 1;
        dimension(padded);
    }
    else
    {
        const auto full = checked_add(checked_mul(input.shape.frames - 1, c.stride), span);
        detail::require(full > padding, "Metal transposed convolution has no output frames");
        frames = full - padding;
        dimension(full);
    }
    const SignalShape shape{c.output_channels, frames, input.shape.groups};
    const auto count = shape.elements(limits);
    dimension(count);
    dimension(input.shape.elements(limits));
    const abi::ConvolutionParams p{dimension(c.input_channels),
                                   dimension(c.output_channels),
                                   dimension(c.kernel),
                                   dimension(c.stride),
                                   dimension(c.padding),
                                   dimension(c.dilation),
                                   c.kind == ConvolutionKind::transposed ? 1u : 0u,
                                   dimension(input.shape.frames),
                                   dimension(frames),
                                   dimension(input.shape.groups)};
    const auto destination = impl_->allocate(count * sizeof(float));
    const std::vector buffers{bind(abi::convolution_binding::input, input.buffer),
                              bind(abi::convolution_binding::weights, conv.impl_->weights),
                              bind(abi::convolution_binding::bias, conv.impl_->bias),
                              bind(abi::convolution_binding::output, destination)};
    if (c.output_channels >= 64 && c.output_channels % 4 == 0)
    {
        // Generic tiles retain eight float4 accumulators. Hot 192/384-channel
        // seven-tap layers use four with cached, statically specialized pipelines.
        const bool narrow = c.output_channels % 128 == 64;
        const bool matched = c.kind == ConvolutionKind::ordinary && c.output_channels == 96;
        const bool pointwise = c.kind == ConvolutionKind::ordinary && c.kernel == 1 &&
                               c.stride == 1 && c.padding == 0 &&
                               (c.output_channels == 96 || c.output_channels == 192);
        const bool hot = c.kind == ConvolutionKind::ordinary &&
                         c.input_channels == c.output_channels &&
                         (c.output_channels == 192 || c.output_channels == 384) && c.kernel == 7 &&
                         c.stride == 1 && (c.dilation == 1 || c.dilation == 3 || c.dilation == 9) &&
                         c.padding == 3 * c.dilation;
        const bool ordinary_static =
            c.kind == ConvolutionKind::ordinary && c.input_channels == c.output_channels &&
            (c.output_channels == 96 || c.output_channels == 768) && c.kernel == 7 &&
            c.stride == 1 && (c.dilation == 1 || c.dilation == 3 || c.dilation == 9) &&
            c.padding == 3 * c.dilation;
        const std::size_t channel_width = hot ? 64 : matched ? 96 : (narrow ? 64 : 128);
        const std::size_t frame_width = hot ? 32 : matched ? 32 : (narrow ? 64 : 32);
        const std::size_t threads = matched ? 96 : 128;
        static constexpr const char* hot_names[2][3] = {
            {"conv_hot_192_1", "conv_hot_192_3", "conv_hot_192_9"},
            {"conv_hot_384_1", "conv_hot_384_3", "conv_hot_384_9"}};
        static constexpr const char* ordinary_names[2][3] = {
            {"conv_static_96_1", "conv_static_96_3", "conv_static_96_9"},
            {"conv_static_768_1", "conv_static_768_3", "conv_static_768_9"}};
        const char* kernel =
            ordinary_static ? ordinary_names[c.output_channels == 768][c.dilation == 1   ? 0
                                                                       : c.dilation == 3 ? 1
                                                                                         : 2]
            : hot           ? hot_names[c.output_channels == 384][c.dilation == 1   ? 0
                                                                  : c.dilation == 3 ? 1
                                                                                    : 2]
            : pointwise
                ? (matched ? "point96" : "point64")
                : (matched ? "matched" : (narrow ? "convolution_tiled64" : "convolution_tiled"));
        const auto channel_tiles =
            checked_add(c.output_channels, channel_width - 1) / channel_width;
        // Match the shader fallback for sparse/extreme public API shapes.
        const auto phases =
            c.kind == ConvolutionKind::transposed && c.stride <= frames &&
                    c.stride <=
                        (std::numeric_limits<std::uint32_t>::max() - frames) / (frame_width - 1)
                ? c.stride
                : 1;
        // Static transposed kernels require the full stride-phase grid. Sparse
        // or extreme shapes keep the checked generic shader and its fallback.
        const char* transposed_kernel = nullptr;
        if (c.kind == ConvolutionKind::transposed && phases == c.stride &&
            c.input_channels == 2 * c.output_channels && c.kernel == 2 * c.stride &&
            c.padding == c.stride / 2 && c.dilation == 1)
        {
            if (c.input_channels == 192 && c.stride == 2)
            {
                transposed_kernel = "conv_transpose_192_96";
            }
            else if (c.input_channels == 384 && c.stride == 4)
            {
                transposed_kernel = "conv_transpose_384_192";
            }
            else if (c.input_channels == 768 && c.stride == 8)
            {
                transposed_kernel = "conv_transpose_768_384";
            }
            else if (c.input_channels == 1536 && c.stride == 8)
            {
                transposed_kernel = "conv_transpose_1536_768";
            }
        }
        if (transposed_kernel)
        {
            kernel = transposed_kernel;
        }
        const auto phase_frames = (frames - 1) / phases + 1;
        const auto frame_tiles = checked_add(phase_frames, frame_width - 1) / frame_width;
        const auto tiles = checked_mul(checked_mul(checked_mul(channel_tiles, frame_tiles), phases),
                                       input.shape.groups);
        impl_->record(kernel, buffers, checked_mul(tiles, threads), p, threads,
                      (hot || ordinary_static || transposed_kernel)
                          ? std::array<std::size_t, 3>{checked_mul(channel_tiles, phases),
                                                       frame_tiles, input.shape.groups}
                          : std::array<std::size_t, 3>{});
    }
    else if (c.output_channels % 4 == 0)
    {
        impl_->record("convolution4", buffers, count / 4, p);
    }
    else
    {
        impl_->record("convolution", buffers, count, p);
    }
    impl_->record("check_finite",
                  {bind(abi::check_finite_binding::input, destination),
                   bind(abi::check_finite_binding::error, impl_->error)},
                  count, abi::CheckFiniteParams{dimension(count)});
    return {shape, destination};
}

DacTensor DacGraph::snake(const DacTensor& input, const Buffer& alpha, const Limits& limits)
{
    const auto count = input.shape.elements(limits);
    detail::require(alpha.size() == input.shape.channels * sizeof(float),
                    "Metal graph Snake shape mismatch");
    const auto output = impl_->allocate(count * sizeof(float));
    const abi::SnakeParams p{dimension(count), dimension(input.shape.channels)};
    // Validate before any later operation can mask an invalid intermediate.
    impl_->record("snake",
                  {bind(abi::snake_binding::input, input.buffer),
                   bind(abi::snake_binding::alpha, alpha), bind(abi::snake_binding::output, output),
                   bind(abi::snake_binding::error, impl_->error)},
                  count, p);
    return {input.shape, output};
}

DacTensor DacGraph::residual(const DacTensor& left, const DacTensor& right, bool subtract,
                             const Limits& limits)
{
    detail::require(left.shape.channels == right.shape.channels &&
                        left.shape.frames == right.shape.frames &&
                        left.shape.groups == right.shape.groups,
                    "Metal graph residual shape mismatch");
    const auto count = left.shape.elements(limits);
    const auto output = impl_->allocate(count * sizeof(float));
    const abi::ResidualParams p{dimension(count), subtract ? 1u : 0u};
    impl_->record("residual",
                  {bind(abi::residual_binding::left, left.buffer),
                   bind(abi::residual_binding::right, right.buffer),
                   bind(abi::residual_binding::output, output),
                   bind(abi::residual_binding::error, impl_->error)},
                  count, p);
    return {left.shape, output};
}

Buffer DacGraph::search(const DacTensor& input, const RvqBuffers& weights, const Limits& limits,
                        bool normalize)
{
    detail::require(input.shape.channels == 8 && weights.normalized.size() == 8192 * sizeof(float),
                    "Metal RVQ search shape mismatch");
    const auto count = input.shape.elements(limits) / 8;
    const auto output = shared_temporary(count * sizeof(std::uint16_t));
    const abi::RvqSearchParams p{dimension(count), normalize ? 1u : 0u};
    impl_->record("rvq_search",
                  {bind(abi::rvq_search_binding::input, input.buffer),
                   bind(abi::rvq_search_binding::codebook, weights.normalized),
                   bind(abi::rvq_search_binding::output, output)},
                  count * 32, p, 32);
    return output;
}

DacTensor DacGraph::lookup(const Buffer& tokens, const RvqBuffers& weights, const Limits& limits)
{
    detail::require(tokens.size() && tokens.size() % sizeof(std::uint16_t) == 0 &&
                        weights.codebook.size() == 8192 * sizeof(float) &&
                        weights.weights.size() == 8192 * sizeof(float) &&
                        weights.bias.size() == 1024 * sizeof(float),
                    "Metal RVQ lookup shape mismatch");
    const SignalShape shape{1024, tokens.size() / sizeof(std::uint16_t), 1};
    const auto count = shape.elements(limits);
    dimension(count);
    auto output = impl_->allocate(checked_mul(count, sizeof(float)));
    impl_->record("rvq_project",
                  {bind(abi::rvq_project_binding::tokens, tokens),
                   bind(abi::rvq_project_binding::codebook, weights.codebook),
                   bind(abi::rvq_project_binding::weights, weights.weights),
                   bind(abi::rvq_project_binding::bias, weights.bias),
                   bind(abi::rvq_project_binding::output, output),
                   bind(abi::rvq_project_binding::error, impl_->error)},
                  count, abi::RvqProjectParams{dimension(shape.frames)});
    return {shape, std::move(output)};
}

DacTensor DacGraph::project(const Buffer& tokens, const RvqBuffers& weights,
                            const DacTensor& residual, bool subtract, const Limits& limits)
{
    const auto count = residual.shape.elements(limits);
    detail::require(residual.shape.channels == 1024 && tokens.size() == count / 1024 * 2,
                    "Metal RVQ project shape mismatch");
    const auto output = impl_->allocate(count * sizeof(float));
    const abi::RvqAccumulateParams p{dimension(count), subtract ? 1u : 0u};
    impl_->record("rvq_accumulate",
                  {bind(abi::rvq_accumulate_binding::tokens, tokens),
                   bind(abi::rvq_accumulate_binding::codebook, weights.codebook),
                   bind(abi::rvq_accumulate_binding::weights, weights.weights),
                   bind(abi::rvq_accumulate_binding::bias, weights.bias),
                   bind(abi::rvq_accumulate_binding::residual, residual.buffer),
                   bind(abi::rvq_accumulate_binding::output, output),
                   bind(abi::rvq_accumulate_binding::error, impl_->error)},
                  count, p);
    return {residual.shape, output};
}

void DacGraph::finish()
{
    std::vector<Program::Dispatch> dispatches;
    for (const auto& op : impl_->operations)
    {
        dispatches.push_back(
            {op.name, op.buffers, op.threads, op.width, op.constants, &op.pipeline, op.groups});
    }
    impl_->execution.run(dispatches);
    impl_->operations.clear();
    std::uint32_t error = 0;
    impl_->error.read(&error, 4);
    detail::require(!error, "nonfinite Metal DAC activation");
}

Signal DacGraph::download(const DacTensor& tensor, const Limits& limits)
{
    finish();
    Signal result{tensor.shape, std::vector<float>(tensor.shape.elements(limits))};
    tensor.buffer.read(result.values.data(), result.values.size() * sizeof(float));
    return result;
}
} // namespace metal

Signal MetalConvolution::apply(std::span<const float> input, SignalShape shape,
                               const Limits& limits) const
{
    metal::DacGraph graph;
    return graph.download(graph.convolution(*this, graph.upload(input, shape, limits), limits),
                          limits);
}

Signal metal_snake(std::span<const float> input, SignalShape shape, std::span<const float> alpha,
                   const Limits& limits)
{
    const auto count = shape.elements(limits);
    detail::require(input.size() == count && alpha.size() == shape.channels,
                    "Metal Snake shape mismatch");
    detail::require_finite(input, "nonfinite Metal Snake input");
    detail::require_finite(alpha, "nonfinite Metal Snake alpha");
    for (const auto a : alpha)
    {
        detail::require(std::isfinite(a + snake_epsilon) && a + snake_epsilon != 0,
                        "invalid Metal Snake denominator");
    }
    metal::DacGraph graph;
    return graph.download(graph.snake(graph.upload(input, shape, limits), upload(alpha), limits),
                          limits);
}

namespace
{
std::vector<std::uint16_t> rvq_search_impl(std::span<const float> vectors,
                                           std::span<const float> codebook, bool normalize,
                                           const Limits& limits)
{
    detail::require(vectors.size() % 8 == 0 && vectors.size() <= limits.tensor_elements &&
                        codebook.size() == 8192,
                    "Metal RVQ search shape mismatch");
    detail::require_finite(vectors, "nonfinite Metal RVQ search input");
    detail::require_finite(codebook, "nonfinite Metal RVQ search codebook");
    const auto count = vectors.size() / 8;
    std::vector<std::uint16_t> selected(count);
    if (!count)
    {
        return selected;
    }
    metal::DacGraph graph;
    const auto codebook_buffer = upload(codebook);
    const metal::RvqBuffers weights{codebook_buffer, codebook_buffer, codebook_buffer,
                                    codebook_buffer};
    const auto output =
        graph.search(graph.upload(vectors, {8, count, 1}, limits), weights, limits, normalize);
    graph.finish();
    const auto mapped = output.mapped<std::uint16_t>();
    std::copy(mapped.begin(), mapped.end(), selected.begin());
    for (const auto index : selected)
    {
        detail::require(index < 1024, "invalid Metal RVQ normalization norm");
    }
    return selected;
}
} // namespace

std::vector<std::uint16_t> metal_rvq_search(std::span<const float> vectors,
                                            std::span<const float> codebook, const Limits& limits)
{
    return rvq_search_impl(vectors, codebook, false, limits);
}

std::vector<std::uint16_t> metal_rvq_search_raw_projected(std::span<const float> vectors,
                                                          std::span<const float> codebook,
                                                          const Limits& limits)
{
    return rvq_search_impl(vectors, codebook, true, limits);
}

std::vector<float> metal_residual(std::span<const float> left, std::span<const float> right,
                                  bool subtract, const Limits& limits)
{
    detail::require(left.size() == right.size() && left.size() <= limits.tensor_elements,
                    "Metal residual shape mismatch");
    detail::require_finite(left, "nonfinite Metal residual input");
    detail::require_finite(right, "nonfinite Metal residual input");
    if (left.empty())
    {
        return {};
    }
    metal::DacGraph graph;
    const SignalShape shape{1, left.size(), 1};
    return graph
        .download(graph.residual(graph.upload(left, shape, limits),
                                 graph.upload(right, shape, limits), subtract, limits),
                  limits)
        .values;
}

std::vector<float> metal_rvq_project(std::span<const std::uint16_t> tokens,
                                     std::span<const float> codebook,
                                     std::span<const float> weights, std::span<const float> bias,
                                     const Limits& limits)
{
    detail::require(codebook.size() == 8192 && weights.size() == 8192 && bias.size() == 1024,
                    "Metal RVQ projection shape mismatch");
    detail::require_finite(codebook, "nonfinite Metal RVQ projection codebook");
    detail::require_finite(weights, "nonfinite Metal RVQ projection weight");
    detail::require_finite(bias, "nonfinite Metal RVQ projection bias");
    for (const auto token : tokens)
    {
        detail::require(token < 1024, "Metal RVQ token exceeds alphabet");
    }
    const auto count = checked_mul(tokens.size(), 1024);
    detail::require(count <= limits.tensor_elements, "Metal RVQ output exceeds element limit");
    if (!count)
    {
        return {};
    }
    metal::DacGraph graph;
    const auto codebook_buffer = upload(codebook);
    const metal::RvqBuffers resources{codebook_buffer, upload(weights), upload(bias),
                                      codebook_buffer};
    return graph.download(graph.lookup(upload(tokens), resources, limits), limits).values;
}
} // namespace tsac
