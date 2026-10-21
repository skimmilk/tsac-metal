#include "metal/prediction.hpp"

#include "common/internal.hpp"
#include "metal/abi/bindings.h"
#include "metal/linear.hpp"
#include "metal/probability/parameters.h"
#include "metal/runtime.hpp"
#include "metal/weight_preparation.hpp"
#include "probability_metallib.hpp"

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>

namespace tsac
{
namespace
{
namespace abi = ::tsac_metal_abi;
using metal::bind;

const metal::Program& program()
{
    static const auto instance =
        metal::Program::load(std::as_bytes(std::span(metal::embedded::probability)), "probability");
    return instance;
}

std::uint32_t dimension(std::size_t value)
{
    detail::require(value <= std::numeric_limits<std::uint32_t>::max(),
                    "Metal probability dimension exceeds uint32");
    return static_cast<std::uint32_t>(value);
}

template <class T> metal::Buffer upload(std::span<const T> values)
{
    return metal::Buffer::copy(values.data(), values.size_bytes());
}

void finite(std::span<const std::uint16_t> values)
{
    for (auto value : values)
    {
        detail::require((value & 0x7f80u) != 0x7f80u, "nonfinite Metal probability input");
    }
}
} // namespace

struct MetalLinear::Impl
{
    std::size_t input, output;
    metal::Buffer weights, scales, bias, packed;
    std::size_t packed_rows;
};

MetalLinear::MetalLinear(std::shared_ptr<Impl> impl) : impl_(std::move(impl))
{
}

MetalLinear MetalLinear::prepare(std::size_t input, std::size_t output,
                                 std::span<const std::int8_t> weights,
                                 std::span<const float> scales, std::span<const float> bias)
{
    detail::require(input && output && input % 256 == 0 &&
                        weights.size() == checked_mul(input, output) &&
                        scales.size() == checked_mul(input / 256, output) && bias.size() == output,
                    "Metal linear weight shape mismatch");
    dimension(input);
    dimension(output);
    for (float x : scales)
    {
        detail::require(std::isfinite(x), "nonfinite Metal linear scale");
    }
    for (float x : bias)
    {
        detail::require(std::isfinite(x), "nonfinite Metal linear bias");
    }
    const std::size_t rows = input == 512 && output >= 1024 ? 8 : 2;
    const auto padded = checked_mul((checked_add(output, rows - 1) / rows), rows);
    auto packed = metal::Buffer::allocate(checked_mul(input, padded));
    auto* packed_bytes = reinterpret_cast<std::int8_t*>(packed.mapped_bytes().data());
    metal::pack_linear_weights(weights, input, output, rows, {packed_bytes, packed.size()});
    auto impl = std::make_shared<Impl>(Impl{input, output, upload(weights), upload(scales),
                                            upload(bias), std::move(packed), rows});
    return MetalLinear(std::move(impl));
}

std::vector<metal::Buffer> MetalLinear::resources() const
{
    return {impl_->weights, impl_->scales, impl_->bias, impl_->packed};
}

MetalLinear MetalLinear::from_weights(const LinearWeights& ref)
{
    const auto input_width = ref.values.shape().axes[0], output_width = ref.values.shape().axes[1];
    dimension(input_width);
    dimension(output_width);
    const auto source = weights::View(ref.values);
    const auto scales = weights::View(ref.scales);
    const auto bias = weights::View(ref.bias);
    detail::require(input_width && input_width % 256 == 0 && output_width &&
                        source.bytes == input_width * output_width &&
                        scales.bytes == input_width / 256 * output_width * 4 &&
                        bias.bytes == output_width * 4,
                    "invalid mapped Metal projection");
    const auto rows = input_width == 512 && output_width >= 1024 ? 8u : 2u;
    const auto padded = (output_width + rows - 1) / rows * rows;
    auto packed = metal::Buffer::allocate(input_width * padded);
    auto* output = reinterpret_cast<std::int8_t*>(packed.mapped_bytes().data());
    const auto* input = reinterpret_cast<const std::int8_t*>(source.data().data());
    metal::pack_linear_weights({input, source.bytes}, input_width, output_width, rows,
                               {output, packed.size()});
    return MetalLinear(std::make_shared<Impl>(
        Impl{input_width, output_width, metal::Buffer::import_weights(source),
             metal::Buffer::import_weights(scales), metal::Buffer::import_weights(bias),
             std::move(packed), rows}));
}

std::vector<std::uint16_t> MetalLinear::apply(std::span<const std::uint16_t> values) const
{
    detail::require(impl_ && values.size() % impl_->input == 0,
                    "Metal linear activation shape mismatch");
    finite(values);
    if (values.empty())
    {
        return {};
    }
    metal::Prediction graph;
    graph.begin();
    const auto result = graph.linear(*this, graph.upload(values));
    graph.finish();
    const auto mapped = result.mapped<std::uint16_t>();
    return {mapped.begin(), mapped.end()};
}

namespace metal
{
struct Prediction::Impl
{
    struct Operation
    {
        std::string_view kernel;
        std::vector<ResourceBinding> arguments;
        std::size_t threads, width;
        std::vector<std::byte> parameters;
        Pipeline pipeline;
    };

    struct Slot
    {
        Buffer error = Buffer::allocate(sizeof(std::uint32_t));
        std::vector<Buffer> staging;
        Ticket ticket = 0;
    };

    std::array<Slot, 2> slots;
    std::size_t recording_slot = 0;
    Buffer error = slots[0].error;
    bool fuse_projections = false;
    BufferStorage scratch_storage = BufferStorage::shared;
    std::unique_ptr<ExecutionContext> execution;
    // CPU-populated staging never aliases GPU scratch. Its cursor advances for
    // the entire submission, even when recycle() reuses GPU-only activations.
    std::size_t staging_cursor = 0;
    // One caller owns begin/record/finish. Activation slots remain distinct
    // until recycle() follows their last recorded consumer. Operations retain
    // backing allocations, including across slot growth within a chunk.
    std::vector<Buffer> scratch;
    std::vector<Operation> operations;
    std::vector<Program::Dispatch> dispatches;
    std::size_t scratch_cursor = 0, operation_cursor = 0;
    using Clock = std::chrono::steady_clock;
    bool profile = std::getenv("TSAC_METAL_PROFILE") != nullptr;
    Clock::time_point recording_started;
    std::uint64_t recording_ns = 0, finalizing_ns = 0;

    template <class T>
    void record(std::string_view name, std::initializer_list<ResourceBinding> args,
                std::size_t threads, const T& p, std::size_t width = 0)
    {
        static_assert(abi::is_parameter_block<T>);
        if (operation_cursor == operations.size())
        {
            operations.push_back({{}, {}, threads, width, {}, {}});
        }
        auto& op = operations[operation_cursor++];
        if (op.kernel != name)
        {
            op.pipeline = program().resolve(name);
        }
        op.kernel = name;
        if (op.arguments.size() != args.size())
        {
            op.arguments.assign(args);
        }
        else
        {
            auto* destination = op.arguments.data();
            for (const auto& argument : args)
            {
                // Repeated decoding chunks keep most bindings identical.
                // Avoid retaining/releasing their shared owners again.
                if (destination->index != argument.index || destination->buffer != argument.buffer)
                {
                    *destination = argument;
                }
                ++destination;
            }
        }
        op.threads = threads;
        op.width = width;
        const auto bytes = std::as_bytes(std::span(&p, 1));
        op.parameters.assign(bytes.begin(), bytes.end());
    }
};

Prediction::Prediction() : impl_(std::make_unique<Impl>())
{
    impl_->execution = std::make_unique<ExecutionContext>(program());
}

Prediction::Prediction(std::shared_ptr<const ResidencySet> model_resources)
    : impl_(std::make_unique<Impl>())
{
    detail::require(model_resources != nullptr, "Metal prediction needs model residency");
    impl_->scratch_storage = BufferStorage::gpu_only;
    impl_->fuse_projections = true;
    impl_->execution = std::make_unique<ExecutionContext>(program(), std::move(model_resources));
}

void Prediction::reset_session(std::shared_ptr<const ResidencySet> session_resources)
{
    detail::require(impl_->execution != nullptr, "Prediction has no persistent execution context");
    drain();
    impl_->dispatches.clear();
    impl_->operations.clear();
    impl_->operation_cursor = 0;
    impl_->scratch_cursor = 0;
    impl_->staging_cursor = 0;
    impl_->execution->set_session_resources(std::move(session_resources));
}

Prediction::~Prediction()
{
    try
    {
        drain();
    }
    catch (...)
    {
    }
    if (std::getenv("TSAC_METAL_PROFILE"))
    {
        std::size_t bytes = 0;
        for (const auto& buffer : impl_->scratch)
        {
            bytes += buffer.size();
        }
        std::fprintf(stderr, "Metal probability scratch: slots=%zu capacity_bytes=%zu\n",
                     impl_->scratch.size(), bytes);
        std::fprintf(stderr, "Metal probability host: record=%.6f finalize=%.6f seconds\n",
                     double(impl_->recording_ns) / 1e9, double(impl_->finalizing_ns) / 1e9);
    }
}

void Prediction::fuse_projections(bool enabled)
{
    impl_->fuse_projections = enabled;
}

void Prediction::recycle()
{
    impl_->scratch_cursor = 0;
}

void Prediction::begin()
{
    auto& slot = impl_->slots[impl_->recording_slot];
    wait(slot.ticket);
    impl_->error = slot.error;
    if (impl_->profile)
    {
        impl_->recording_started = Impl::Clock::now();
    }
    impl_->staging_cursor = 0;
    impl_->scratch_cursor = 0;
    impl_->operation_cursor = 0;
    const std::uint32_t clear = 0;
    impl_->error.write(&clear, sizeof clear);
}

Buffer Prediction::temporary(std::size_t bytes)
{
    auto& s = impl_->scratch;
    // Retain high-water allocations; geometric growth handles longer temporal
    // histories without allocating each layer/step. Return only the requested
    // view so kernels observe logical shapes rather than spare capacity.
    const auto i = impl_->scratch_cursor++;
    if (i == s.size())
    {
        s.push_back(Buffer::allocate(std::bit_ceil(std::max<std::size_t>(bytes, 1)),
                                     impl_->scratch_storage));
    }
    else if (s[i].size() < bytes)
    {
        s[i] = Buffer::allocate(std::bit_ceil(bytes), impl_->scratch_storage);
    }
    return s[i].slice(0, bytes);
}

Buffer Prediction::shared_temporary(std::size_t bytes)
{
    auto& slots = impl_->slots[impl_->recording_slot].staging;
    const auto i = impl_->staging_cursor++;
    if (i == slots.size())
    {
        slots.push_back(Buffer::allocate(std::bit_ceil(std::max<std::size_t>(bytes, 1))));
    }
    else if (slots[i].size() < bytes)
    {
        slots[i] = Buffer::allocate(std::bit_ceil(bytes));
    }
    return slots[i].slice(0, bytes);
}

Buffer Prediction::upload(std::span<const std::uint16_t> values)
{
    auto result = shared_temporary(values.size_bytes());
    result.write(values.data(), values.size_bytes());
    return result;
}

Buffer Prediction::embed(const Buffer& embeddings, std::span<const std::uint16_t> tokens,
                         std::size_t columns)
{
    detail::require(!tokens.empty() && columns && tokens.size() % columns == 0 &&
                        tokens.size() / columns * 1024 * 1024 * 2 <= embeddings.size(),
                    "Metal graph embedding shape mismatch");
    for (auto t : tokens)
    {
        detail::require(t < 1024, "Metal graph token exceeds alphabet");
    }
    auto selected = upload(tokens), output = temporary(columns * 1024 * 2);
    abi::EmbeddingParams p{};
    p.prefix_books = dimension(tokens.size() / columns);
    p.column_count = dimension(columns);
    impl_->record("embed_tokens",
                  {bind(abi::embedding_binding::embeddings, embeddings),
                   bind(abi::embedding_binding::tokens, selected),
                   bind(abi::embedding_binding::output, output)},
                  columns * 1024, p);
    return output;
}

Buffer Prediction::embed_resident(const Buffer& embeddings, const Buffer& tokens, const Header& h,
                                  std::size_t step, std::size_t prefix, std::size_t first,
                                  std::size_t columns)
{
    const bool initial = step == std::numeric_limits<std::uint32_t>::max();
    return embed_resident(embeddings, tokens, h, ResidentPosition{initial ? 0 : step, initial},
                          prefix, first, columns);
}

Buffer Prediction::embed_resident(const Buffer& embeddings, const Buffer& tokens, const Header& h,
                                  ResidentPosition position, std::size_t prefix, std::size_t first,
                                  std::size_t columns)
{
    detail::require(prefix && prefix <= h.codebooks && tokens.size() == h.symbol_count() * 2 &&
                        first + columns <= std::min<std::size_t>(h.batch, h.frames) * h.groups,
                    "invalid resident embedding");
    abi::ResidentEmbeddingParams p{};
    p.frame_count = dimension(h.frames);
    p.lane_count = h.batch;
    p.group_count = h.groups;
    p.book_count = h.codebooks;
    p.initial_feedback = position.initial_feedback;
    p.step_index = dimension(position.step_index);
    p.prefix_books = dimension(prefix);
    p.first_sequence = dimension(first);
    p.column_count = dimension(columns);
    auto output = temporary(columns * 1024 * 2);
    impl_->record("embed_resident",
                  {bind(abi::embedding_binding::embeddings, embeddings),
                   bind(abi::embedding_binding::tokens, tokens),
                   bind(abi::embedding_binding::output, output)},
                  columns * 1024, p);
    return output;
}

Buffer Prediction::add(const Buffer& left, const Buffer& right, std::size_t count)
{
    detail::require(count * 2 <= left.size() && right.size() && count * 2 % right.size() == 0,
                    "Metal graph add shape mismatch");
    auto output = temporary(count * 2);
    abi::AddParams p{};
    p.element_count = dimension(count);
    p.right_period_elements = dimension(right.size() / 2);
    impl_->record("add_kernel",
                  {bind(abi::add_binding::input, left), bind(abi::add_binding::right, right),
                   bind(abi::add_binding::output, output)},
                  count, p);
    return output;
}

Buffer Prediction::norm(const Buffer& input, const Buffer& gain, const Buffer& bias)
{
    detail::require(input.size() && input.size() % 1024 == 0 && gain.size() == 1024 &&
                        bias.size() == 1024,
                    "Metal graph norm shape mismatch");
    auto output = temporary(input.size());
    abi::NormalizationParams p{};
    p.column_count = dimension(input.size() / 1024);
    p.epsilon = 1e-5f;
    impl_->record("norm512_parallel",
                  {bind(abi::normalization_binding::input, input),
                   bind(abi::normalization_binding::gain, gain),
                   bind(abi::normalization_binding::bias, bias),
                   bind(abi::normalization_binding::output, output),
                   bind(abi::normalization_binding::error, impl_->error)},
                  input.size() / 1024 * 128, p, 128);
    return output;
}

Prediction::QuantizedActivations Prediction::quantized_temporary(const MetalLinear& weights,
                                                                 std::size_t columns)
{
    const auto& w = *weights.impl_;
    const auto storage = columns == 3 && w.output % w.packed_rows == 0 ? QuantizedStorage::words
                                                                       : QuantizedStorage::bytes;
    const auto count = checked_mul(w.input, columns);
    return {temporary(
                checked_mul(count, storage == QuantizedStorage::words ? sizeof(std::int32_t) : 1)),
            temporary(count / 128), storage};
}

Buffer Prediction::linear(const MetalLinear& weights, const Buffer& input, bool activate_gelu,
                          const Buffer* residual)
{
    const auto& w = *weights.impl_;
    detail::require(input.size() && input.size() % (w.input * 2) == 0,
                    "Metal graph linear input shape mismatch");
    const auto columns = input.size() / (w.input * 2);
    // Full fusion repeats preparation per output tile. Paired geometry probes
    // favor it for one column; wider batches keep shared preparation/matrices.
    if (impl_->fuse_projections && columns == 1 && w.input <= 2048)
    {
        auto output = temporary(w.output * columns * 2);
        if (residual)
        {
            detail::require(residual->size() == output.size(),
                            "Metal fused residual shape mismatch");
        }
        abi::PreparedProjectionParams p{};
        p.input_width = dimension(w.input);
        p.output_width = dimension(w.output);
        p.column_count = dimension(columns);
        p.apply_gelu = activate_gelu;
        p.add_residual = residual != nullptr;
        impl_->record(
            w.packed_rows == 8 ? "prepared8" : "prepared2",
            {bind(abi::prepared_projection_binding::weights, w.packed),
             bind(abi::prepared_projection_binding::scales, w.scales),
             bind(abi::prepared_projection_binding::bias, w.bias),
             bind(abi::prepared_projection_binding::input, input),
             bind(abi::prepared_projection_binding::output, output),
             bind(abi::prepared_projection_binding::gain, input),
             bind(abi::prepared_projection_binding::norm_bias, input),
             bind(abi::prepared_projection_binding::residual, residual ? *residual : input),
             bind(abi::prepared_projection_binding::error, impl_->error)},
            ((w.output + 8 * w.packed_rows - 1) / (8 * w.packed_rows)) * columns * 256, p, 256);
        return output;
    }
    auto quantized = quantized_temporary(weights, columns);
    abi::QuantizationParams p{};
    p.input_width = dimension(w.input);
    p.column_count = dimension(columns);
    p.apply_gelu = activate_gelu;
    impl_->record(quantized.storage == QuantizedStorage::words ? "quant_bf16_parallel_wide"
                                                               : "quant_bf16_parallel",
                  {bind(abi::quantization_checked_binding::input, input),
                   bind(abi::quantization_checked_binding::output, quantized.values),
                   bind(abi::quantization_checked_binding::scales, quantized.scales),
                   bind(abi::quantization_checked_binding::error, impl_->error)},
                  w.input * columns, p, 256);
    return quantized_linear(weights, quantized, columns, residual);
}

Buffer Prediction::quantized_linear(const MetalLinear& weights, const QuantizedActivations& input,
                                    std::size_t columns, const Buffer* residual)
{
    const auto& w = *weights.impl_;
    const auto& quantized = input.values;
    const auto& scales = input.scales;
    const bool wide = input.storage == QuantizedStorage::words;
    auto output = temporary(w.output * columns * 2);
    abi::ProjectionParams p{};
    p.input_width = dimension(w.input);
    p.output_width = dimension(w.output);
    p.column_count = dimension(columns);
    if (residual)
    {
        detail::require(residual->size() == output.size(), "Metal linear residual shape mismatch");
    }
    const bool packed = p.column_count == 3 && w.output % w.packed_rows == 0;
    detail::require(wide == packed, "Metal quantized activation format does not match projection");
    const bool matrix = p.column_count >= (w.input >= 2048 ? 16u : 8u);
    const bool fused = residual && (packed || matrix);
    if (residual && matrix)
    {
        impl_->record("linear_matrix_add",
                      {bind(abi::projection_binding::weights, w.weights),
                       bind(abi::projection_binding::scales, w.scales),
                       bind(abi::projection_binding::bias, w.bias),
                       bind(abi::projection_binding::input, quantized),
                       bind(abi::projection_binding::activation_scales, scales),
                       bind(abi::projection_binding::output, output),
                       bind(abi::projection_binding::residual, *residual)},
                      ((w.output + 31) / 32) * ((p.column_count + 31) / 32) * 128, p, 128);
    }
    else if (fused)
    {
        impl_->record(w.packed_rows == 8 ? "linear_packed_add_wide8" : "linear_packed_add_wide2",
                      {bind(abi::projection_binding::weights, w.packed),
                       bind(abi::projection_binding::scales, w.scales),
                       bind(abi::projection_binding::bias, w.bias),
                       bind(abi::projection_binding::input, quantized),
                       bind(abi::projection_binding::activation_scales, scales),
                       bind(abi::projection_binding::output, output),
                       bind(abi::projection_binding::residual, *residual)},
                      (w.output / w.packed_rows) * 32, p, 32);
    }
    else if (packed)
    {
        impl_->record(w.packed_rows == 8 ? "linear_packed_wide8" : "linear_packed_wide2",
                      {bind(abi::projection_binding::weights, w.packed),
                       bind(abi::projection_binding::scales, w.scales),
                       bind(abi::projection_binding::bias, w.bias),
                       bind(abi::projection_binding::input, quantized),
                       bind(abi::projection_binding::activation_scales, scales),
                       bind(abi::projection_binding::output, output)},
                      (w.output / w.packed_rows) * 32, p, 32);
    }
    else if (p.column_count >= (w.input >= 2048 ? 16u : 8u))
    {
        impl_->record("linear_matrix",
                      {bind(abi::projection_binding::weights, w.weights),
                       bind(abi::projection_binding::scales, w.scales),
                       bind(abi::projection_binding::bias, w.bias),
                       bind(abi::projection_binding::input, quantized),
                       bind(abi::projection_binding::activation_scales, scales),
                       bind(abi::projection_binding::output, output)},
                      ((w.output + 31) / 32) * ((p.column_count + 31) / 32) * 128, p, 128);
    }
    else if (p.column_count == 1)
    {
        impl_->record("linear_kernel",
                      {bind(abi::projection_binding::weights, w.weights),
                       bind(abi::projection_binding::scales, w.scales),
                       bind(abi::projection_binding::bias, w.bias),
                       bind(abi::projection_binding::input, quantized),
                       bind(abi::projection_binding::activation_scales, scales),
                       bind(abi::projection_binding::output, output)},
                      ((w.output + 7) / 8) * 256, p, 256);
    }
    else
    {
        const std::size_t rows = w.input == 512 ? 4 : 2;
        impl_->record(rows == 4 ? "linear_small4" : "linear_small2",
                      {bind(abi::projection_binding::weights, w.weights),
                       bind(abi::projection_binding::scales, w.scales),
                       bind(abi::projection_binding::bias, w.bias),
                       bind(abi::projection_binding::input, quantized),
                       bind(abi::projection_binding::activation_scales, scales),
                       bind(abi::projection_binding::output, output)},
                      ((w.output + rows - 1) / rows) * ((p.column_count + 2) / 3) * 32, p, 32);
    }
    return residual && !fused ? add(output, *residual, w.output * columns) : output;
}

Buffer Prediction::embed_linear_resident(const MetalLinear& weights, const Buffer& embeddings,
                                         const Buffer& tokens, const Header& h, std::size_t step,
                                         std::size_t prefix, std::size_t first, std::size_t columns)
{
    const bool initial = step == std::numeric_limits<std::uint32_t>::max();
    return embed_linear_resident(weights, embeddings, tokens, h,
                                 ResidentPosition{initial ? 0 : step, initial}, prefix, first,
                                 columns);
}

Buffer Prediction::embed_linear_resident(const MetalLinear& weights, const Buffer& embeddings,
                                         const Buffer& tokens, const Header& h,
                                         ResidentPosition position, std::size_t prefix,
                                         std::size_t first, std::size_t columns)
{
    detail::require(weights.impl_->input == 1024 && prefix && prefix <= h.codebooks &&
                        tokens.size() == h.symbol_count() * 2 &&
                        first + columns <= std::min<std::size_t>(h.batch, h.frames) * h.groups,
                    "invalid resident embedding projection");
    abi::ResidentEmbeddingParams p{};
    p.frame_count = dimension(h.frames);
    p.lane_count = h.batch;
    p.group_count = h.groups;
    p.book_count = h.codebooks;
    p.initial_feedback = position.initial_feedback;
    p.step_index = dimension(position.step_index);
    p.prefix_books = dimension(prefix);
    p.first_sequence = dimension(first);
    p.column_count = dimension(columns);
    auto quantized = quantized_temporary(weights, columns);
    impl_->record(quantized.storage == QuantizedStorage::words ? "embed_quant_resident_wide"
                                                               : "embed_quant_resident",
                  {bind(abi::embedding_quantized_binding::embeddings, embeddings),
                   bind(abi::embedding_quantized_binding::tokens, tokens),
                   bind(abi::embedding_quantized_binding::output, quantized.values),
                   bind(abi::embedding_quantized_binding::scales, quantized.scales),
                   bind(abi::embedding_quantized_binding::error, impl_->error)},
                  columns * 1024, p, 256);
    return quantized_linear(weights, quantized, columns);
}

Buffer Prediction::norm_linear(const MetalLinear& weights, const Buffer& input, const Buffer& gain,
                               const Buffer& bias)
{
    const auto& w = *weights.impl_;
    detail::require(w.input == 512 && input.size() && input.size() % 1024 == 0 &&
                        gain.size() == 1024 && bias.size() == 1024,
                    "Metal fused norm/linear shape mismatch");
    const auto columns = input.size() / 1024;
    if (impl_->fuse_projections && columns == 1)
    {
        auto output = temporary(w.output * columns * 2);
        abi::PreparedProjectionParams p{};
        p.input_width = 512;
        p.output_width = dimension(w.output);
        p.column_count = dimension(columns);
        p.normalize_input = 1;
        p.epsilon = 1e-5f;
        impl_->record(w.packed_rows == 8 ? "prepared8" : "prepared2",
                      {bind(abi::prepared_projection_binding::weights, w.packed),
                       bind(abi::prepared_projection_binding::scales, w.scales),
                       bind(abi::prepared_projection_binding::bias, w.bias),
                       bind(abi::prepared_projection_binding::input, input),
                       bind(abi::prepared_projection_binding::output, output),
                       bind(abi::prepared_projection_binding::gain, gain),
                       bind(abi::prepared_projection_binding::norm_bias, bias),
                       bind(abi::prepared_projection_binding::residual, input),
                       bind(abi::prepared_projection_binding::error, impl_->error)},
                      ((w.output + 8 * w.packed_rows - 1) / (8 * w.packed_rows)) * columns * 256, p,
                      256);
        return output;
    }
    auto quantized = quantized_temporary(weights, columns);
    abi::NormalizationParams p{};
    p.column_count = dimension(columns);
    p.epsilon = 1e-5f;
    impl_->record(quantized.storage == QuantizedStorage::words ? "norm_quant512_wide"
                                                               : "norm_quant512",
                  {bind(abi::normalization_quantized_binding::input, input),
                   bind(abi::normalization_quantized_binding::gain, gain),
                   bind(abi::normalization_quantized_binding::bias, bias),
                   bind(abi::normalization_quantized_binding::output, quantized.values),
                   bind(abi::normalization_quantized_binding::scales, quantized.scales),
                   bind(abi::normalization_quantized_binding::error, impl_->error)},
                  columns * 128, p, 128);
    return quantized_linear(weights, quantized, columns);
}

Buffer Prediction::attention(const Buffer& qkv, const Buffer& keys, const Buffer& values,
                             const Buffer& rotation, std::size_t position, bool rotated,
                             bool prefill, const Buffer* destination)
{
    detail::require(qkv.size() == std::size_t{1536} * 2 &&
                        keys.size() >= (position + 1) * 512 * 2 && values.size() == keys.size() &&
                        (!rotated || rotation.size() == std::size_t{128} * 2),
                    "Metal graph attention shape mismatch");
    const auto time = position + 1, blocks = (time + 63) / 64;
    abi::AttentionParams p{};
    p.head_count = 4;
    p.head_width = 128;
    p.key_count = dimension(time);
    p.score_scale = 0.0883883461356163f;
    p.column_count = 1;
    p.round_block_values = time > 64 || prefill;
    p.key_column_stride_elements = dimension(keys.size() / 2);
    p.value_column_stride_elements = dimension(values.size() / 2);
    const abi::CacheAppendParams cache_params{1, dimension(position), p.key_column_stride_elements,
                                              p.value_column_stride_elements, rotated ? 1u : 0u};
    auto query = temporary(std::size_t{512} * 2), metadata = temporary((4 * blocks * 2 + 4) * 4),
         output = destination ? *destination : temporary(std::size_t{512} * 2);
    impl_->record("cache_qkv",
                  {bind(abi::cache_binding::qkv, qkv), bind(abi::cache_binding::rotation, rotation),
                   bind(abi::cache_binding::query, query), bind(abi::cache_binding::keys, keys),
                   bind(abi::cache_binding::values, values),
                   bind(abi::cache_binding::error, impl_->error)},
                  256, cache_params);
    // Short histories fuse attention; middle histories keep ordered stages.
    // cache_qkv precedes all attention reads in this recorded command stream.
    if (time <= 32)
    {
        impl_->record("attention_short",
                      {bind(abi::attention_short_binding::query, query),
                       bind(abi::attention_short_binding::keys, keys),
                       bind(abi::attention_short_binding::values, values),
                       bind(abi::attention_short_binding::output, output)},
                      std::size_t{4} * 128, p, 128);
    }
    else if (time > 64)
    {
        auto partials = temporary(blocks * 512 * 4), summaries = temporary(4 * blocks * 2 * 4);
        impl_->record("attention_blocks",
                      {bind(abi::attention_blocks_binding::query, query),
                       bind(abi::attention_blocks_binding::keys, keys),
                       bind(abi::attention_blocks_binding::values, values),
                       bind(abi::attention_blocks_binding::partials, partials),
                       bind(abi::attention_blocks_binding::summaries, summaries)},
                      4 * blocks * 128, p, 128);
        impl_->record("attention_block_metadata",
                      {bind(abi::attention_metadata_binding::summaries, summaries),
                       bind(abi::attention_metadata_binding::metadata, metadata)},
                      4, p);
        impl_->record("attention_combine",
                      {bind(abi::attention_combine_binding::partials, partials),
                       bind(abi::attention_combine_binding::metadata, metadata),
                       bind(abi::attention_combine_binding::output, output)},
                      512, p);
    }
    else
    {
        auto scores = temporary(4 * time * 4);
        impl_->record("attention_scores",
                      {bind(abi::attention_scores_binding::query, query),
                       bind(abi::attention_scores_binding::keys, keys),
                       bind(abi::attention_scores_binding::scores, scores)},
                      4 * time, p);
        impl_->record("attention_weights",
                      {bind(abi::attention_weights_binding::scores, scores),
                       bind(abi::attention_weights_binding::metadata, metadata)},
                      4, p);
        impl_->record("attention_output",
                      {bind(abi::attention_output_binding::values, values),
                       bind(abi::attention_output_binding::scores, scores),
                       bind(abi::attention_output_binding::metadata, metadata),
                       bind(abi::attention_output_binding::output, output)},
                      512, p);
    }
    return output;
}

Buffer Prediction::attention_columns(const Buffer& qkv, const Buffer& keys, const Buffer& values,
                                     std::size_t position, std::size_t columns,
                                     std::size_t cache_frames)
{
    detail::require(
        columns && position < cache_frames && position < 32 && qkv.size() == columns * 1536 * 2 &&
            keys.size() == columns * cache_frames * 512 * 2 && keys.size() == values.size(),
        "Metal column attention shape mismatch");
    abi::AttentionParams p{};
    p.head_count = 4;
    p.head_width = 128;
    p.key_count = dimension(position + 1);
    p.score_scale = 0.0883883461356163f;
    p.column_count = dimension(columns);
    p.key_column_stride_elements = dimension(cache_frames * 512);
    p.value_column_stride_elements = p.key_column_stride_elements;
    auto output = temporary(columns * 512 * 2);
    impl_->record("attention_codebook",
                  {bind(abi::attention_codebook_binding::qkv, qkv),
                   bind(abi::attention_codebook_binding::keys, keys),
                   bind(abi::attention_codebook_binding::values, values),
                   bind(abi::attention_codebook_binding::output, output),
                   bind(abi::attention_codebook_binding::error, impl_->error)},
                  columns * 4 * 128, p, 128);
    return output;
}

Buffer Prediction::attention_temporal(const Buffer& qkv, const Buffer& keys, const Buffer& values,
                                      const Buffer& rotation, std::size_t position,
                                      std::size_t columns, std::size_t cache_frames)
{
    detail::require(columns && position < cache_frames && (position < 32 || position >= 64) &&
                        qkv.size() == columns * 1536 * 2 &&
                        keys.size() == columns * cache_frames * 512 * 2 &&
                        keys.size() == values.size() && rotation.size() == 256,
                    "Metal temporal batch shape mismatch");
    const auto time = position + 1, blocks = (time + 63) / 64;
    abi::AttentionParams p{};
    p.head_count = 4;
    p.head_width = 128;
    p.key_count = dimension(time);
    p.score_scale = 0.0883883461356163f;
    p.column_count = dimension(columns);
    p.key_column_stride_elements = dimension(cache_frames * 512);
    p.value_column_stride_elements = p.key_column_stride_elements;
    p.round_block_values = time > 64;
    const abi::CacheAppendParams cache_params{dimension(columns), dimension(position),
                                              p.key_column_stride_elements,
                                              p.value_column_stride_elements, 1};
    auto query = temporary(columns * 512 * 2), output = temporary(columns * 512 * 2);
    impl_->record("cache_qkv",
                  {bind(abi::cache_binding::qkv, qkv), bind(abi::cache_binding::rotation, rotation),
                   bind(abi::cache_binding::query, query), bind(abi::cache_binding::keys, keys),
                   bind(abi::cache_binding::values, values),
                   bind(abi::cache_binding::error, impl_->error)},
                  columns * 256, cache_params);
    if (time <= 32)
    {
        impl_->record("attention_short",
                      {bind(abi::attention_short_binding::query, query),
                       bind(abi::attention_short_binding::keys, keys),
                       bind(abi::attention_short_binding::values, values),
                       bind(abi::attention_short_binding::output, output)},
                      columns * 4 * 128, p, 128);
    }
    else
    {
        auto partials = temporary(columns * blocks * 512 * 4),
             summaries = temporary(columns * 4 * blocks * 2 * 4),
             metadata = temporary(columns * (4 * blocks * 2 + 4) * 4);
        impl_->record("attention_blocks",
                      {bind(abi::attention_blocks_binding::query, query),
                       bind(abi::attention_blocks_binding::keys, keys),
                       bind(abi::attention_blocks_binding::values, values),
                       bind(abi::attention_blocks_binding::partials, partials),
                       bind(abi::attention_blocks_binding::summaries, summaries)},
                      columns * 4 * blocks * 128, p, 128);
        impl_->record("attention_block_metadata",
                      {bind(abi::attention_metadata_binding::summaries, summaries),
                       bind(abi::attention_metadata_binding::metadata, metadata)},
                      columns * 4, p);
        impl_->record("attention_combine",
                      {bind(abi::attention_combine_binding::partials, partials),
                       bind(abi::attention_combine_binding::metadata, metadata),
                       bind(abi::attention_combine_binding::output, output)},
                      columns * 512, p);
    }
    return output;
}

Buffer Prediction::softmax(const Buffer& input, const Buffer* destination)
{
    detail::require(input.size() && input.size() % (std::size_t{1024} * 2) == 0,
                    "Metal graph softmax shape mismatch");
    auto output = destination ? *destination : temporary(input.size());
    detail::require(output.size() == input.size(), "Metal softmax output shape mismatch");
    abi::SoftmaxParams p{};
    p.column_count = dimension(input.size() / 2048);
    impl_->record("softmax1024_parallel",
                  {bind(abi::softmax_binding::input, input),
                   bind(abi::softmax_binding::output, output),
                   bind(abi::softmax_binding::error, impl_->error)},
                  input.size() / 2048 * 256, p, 256);
    return output;
}

void Prediction::softmax_decode(const Buffer& input, const Buffer& probabilities,
                                const Buffer& tree)
{
    detail::require(input.size() && input.size() % 2048 == 0 &&
                        probabilities.size() == input.size() && tree.size() == input.size() * 2,
                    "invalid decode softmax shape");
    abi::SoftmaxParams p{};
    p.column_count = dimension(input.size() / 2048);
    impl_->record("softmax1024_decode",
                  {bind(abi::softmax_decode_binding::input, input),
                   bind(abi::softmax_decode_binding::tree, tree),
                   bind(abi::softmax_decode_binding::output, probabilities),
                   bind(abi::softmax_decode_binding::error, impl_->error)},
                  input.size() / 2048 * 256, p, 256);
}

void Prediction::range_decode(const Buffer& probabilities, const Buffer& tree,
                              const Buffer& payload, const Buffer& state, const Buffer& tokens,
                              const Header& h, std::size_t step, std::size_t book,
                              std::size_t columns)
{
    detail::require(columns && probabilities.size() >= columns * 2048 &&
                        tree.size() >= columns * 4096 && state.size() == sizeof(abi::RangeState) &&
                        tokens.size() == h.symbol_count() * 2 && book < h.codebooks,
                    "invalid GPU range decode shape");
    abi::RangeDecodeParams p{};
    p.frame_count = dimension(h.frames);
    p.lane_count = h.batch;
    p.group_count = h.groups;
    p.book_count = h.codebooks;
    p.step_index = dimension(step);
    p.book_index = dimension(book);
    p.column_count = dimension(columns);
    p.payload_bytes = dimension(payload.size());
    impl_->record("range_decode",
                  {bind(abi::range_binding::probabilities, probabilities),
                   bind(abi::range_binding::tree, tree), bind(abi::range_binding::payload, payload),
                   bind(abi::range_binding::state, state), bind(abi::range_binding::tokens, tokens),
                   bind(abi::range_binding::error, impl_->error)},
                  1, p);
}

void Prediction::copy(const Buffer& source, const Buffer& destination)
{
    detail::require(source.size() == destination.size(), "Metal graph copy shape mismatch");
    abi::CopyParams p{};
    p.element_count = dimension(source.size() / 2);
    impl_->record(
        "copy_bf16",
        {bind(abi::copy_binding::input, source), bind(abi::copy_binding::output, destination)},
        p.element_count, p);
}

Prediction::Ticket Prediction::submit()
{
    const auto started = impl_->profile ? Impl::Clock::now() : Impl::Clock::time_point{};
    impl_->dispatches.clear();
    for (std::size_t i = 0; i < impl_->operation_cursor; ++i)
    {
        auto& op = impl_->operations[i];
        impl_->dispatches.push_back(
            {op.kernel, op.arguments, op.threads, op.width, op.parameters, &op.pipeline});
    }
    if (impl_->profile)
    {
        impl_->recording_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(started - impl_->recording_started)
                .count());
        impl_->finalizing_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Impl::Clock::now() - started)
                .count());
    }
    auto& slot = impl_->slots[impl_->recording_slot];
    detail::require(!slot.ticket, "Prediction slot is still in flight");
    const auto ticket = impl_->execution->submit(impl_->dispatches);
    slot.ticket = ticket;
    impl_->recording_slot = (impl_->recording_slot + 1) % impl_->slots.size();
    if (!ticket)
    {
        // Empty batches have no GPU completion to wait for.
        std::uint32_t error = 0;
        slot.error.read(&error, sizeof error);
        detail::require(!(error & 2u), "invalid Metal range state");
        detail::require(!error, "nonfinite Metal Transformer activation");
    }
    return ticket;
}

void Prediction::wait(Ticket ticket)
{
    if (!ticket)
    {
        return;
    }
    std::exception_ptr failure;
    try
    {
        impl_->execution->wait(ticket);
    }
    catch (...)
    {
        failure = std::current_exception();
        // Even a rejected ticket or a host exception must not let us inspect
        // error words while another queued command may still be writing them.
        try
        {
            impl_->execution->drain();
        }
        catch (...)
        {
        }
    }
    // Completion of a later ticket also completes its producers. Check their
    // error words in submission order before allowing either slot to be reused.
    for (std::size_t i = 0; i < impl_->slots.size(); ++i)
    {
        Impl::Slot* first = nullptr;
        for (auto& slot : impl_->slots)
        {
            if (slot.ticket && slot.ticket <= ticket && (!first || slot.ticket < first->ticket))
            {
                first = &slot;
            }
        }
        if (!first)
        {
            break;
        }
        first->ticket = 0;
        std::uint32_t error = 0;
        first->error.read(&error, sizeof error);
        if (error && !failure)
        {
            failure = std::make_exception_ptr(
                Error((error & 2u) ? "invalid Metal range state"
                                   : "nonfinite Metal Transformer activation"));
        }
    }
    if (failure)
    {
        std::rethrow_exception(failure);
    }
}

void Prediction::drain()
{
    Ticket last = 0;
    for (const auto& slot : impl_->slots)
    {
        last = std::max(last, slot.ticket);
    }
    wait(last);
}

void Prediction::finish()
{
    wait(submit());
}

} // namespace metal
} // namespace tsac
