// CPU-only build: preserve the public C++ surface without loading Metal or
// compiling shaders. Any accidental GPU execution fails at the boundary.
#include "tsac/metal_analyzer.hpp"

#include "metal/dac_graph.hpp"
#include "metal/prediction.hpp"
#include "metal/transformer.hpp"

namespace
{
[[noreturn]] void disabled()
{
    throw tsac::Error("Metal support was disabled at build time");
}
} // namespace

#define GPU_STUB(type, name, args)                                                                 \
    type name args                                                                                 \
    {                                                                                              \
        disabled();                                                                                \
    }

namespace tsac::metal
{
bool available()
{
    return false;
}
GPU_STUB(Buffer, Buffer::allocate, (std::size_t, BufferStorage))
GPU_STUB(Buffer, Buffer::copy, (const void*, std::size_t))
GPU_STUB(Buffer, Buffer::import_weights, (const weights::View&))
GPU_STUB(void, Buffer::read, (void*, std::size_t) const)
GPU_STUB(void, Buffer::write, (const void*, std::size_t) const)
GPU_STUB(Buffer, Buffer::slice, (std::size_t, std::size_t) const)
GPU_STUB(std::size_t, Buffer::size, () const)
GPU_STUB(bool, Buffer::shares_storage, (const Buffer&) const)
GPU_STUB(std::span<std::byte>, Buffer::mapped_bytes, () const)
GPU_STUB(std::span<const std::byte>, Buffer::read_only_bytes, () const)

struct Prediction::Impl
{
};

Prediction::Prediction()
{
    disabled();
}

// Match the enabled backend's ownership-taking signature.
// NOLINTNEXTLINE(performance-unnecessary-value-param)
Prediction::Prediction(std::shared_ptr<const ResidencySet>)
{
    disabled();
}

Prediction::~Prediction() = default;
GPU_STUB(void, Prediction::begin, ())
// Match the enabled backend's ownership-taking signature.
// NOLINTNEXTLINE(performance-unnecessary-value-param)
GPU_STUB(void, Prediction::reset_session, (std::shared_ptr<const ResidencySet>))
GPU_STUB(void, Prediction::fuse_projections, (bool))
GPU_STUB(void, Prediction::recycle, ())
GPU_STUB(Buffer, Prediction::temporary, (std::size_t))
GPU_STUB(Buffer, Prediction::shared_temporary, (std::size_t))
GPU_STUB(Buffer, Prediction::upload, (std::span<const std::uint16_t>))
GPU_STUB(Buffer, Prediction::embed, (const Buffer&, std::span<const std::uint16_t>, std::size_t))
GPU_STUB(Buffer, Prediction::embed_resident,
         (const Buffer&, const Buffer&, const Header&, std::size_t, std::size_t, std::size_t,
          std::size_t))
GPU_STUB(Buffer, Prediction::embed_resident,
         (const Buffer&, const Buffer&, const Header&, ResidentPosition, std::size_t, std::size_t,
          std::size_t))
GPU_STUB(Buffer, Prediction::embed_linear_resident,
         (const MetalLinear&, const Buffer&, const Buffer&, const Header&, std::size_t, std::size_t,
          std::size_t, std::size_t))
GPU_STUB(Buffer, Prediction::embed_linear_resident,
         (const MetalLinear&, const Buffer&, const Buffer&, const Header&, ResidentPosition,
          std::size_t, std::size_t, std::size_t))
GPU_STUB(Buffer, Prediction::add, (const Buffer&, const Buffer&, std::size_t))
GPU_STUB(Buffer, Prediction::norm, (const Buffer&, const Buffer&, const Buffer&))
GPU_STUB(Buffer, Prediction::linear, (const MetalLinear&, const Buffer&, bool, const Buffer*))
GPU_STUB(Buffer, Prediction::norm_linear,
         (const MetalLinear&, const Buffer&, const Buffer&, const Buffer&))
GPU_STUB(Buffer, Prediction::attention,
         (const Buffer&, const Buffer&, const Buffer&, const Buffer&, std::size_t, bool, bool,
          const Buffer*))
GPU_STUB(Buffer, Prediction::attention_columns,
         (const Buffer&, const Buffer&, const Buffer&, std::size_t, std::size_t, std::size_t))
GPU_STUB(Buffer, Prediction::attention_temporal,
         (const Buffer&, const Buffer&, const Buffer&, const Buffer&, std::size_t, std::size_t,
          std::size_t))
GPU_STUB(Buffer, Prediction::softmax, (const Buffer&, const Buffer*))
GPU_STUB(void, Prediction::softmax_decode, (const Buffer&, const Buffer&, const Buffer&))
GPU_STUB(void, Prediction::range_decode,
         (const Buffer&, const Buffer&, const Buffer&, const Buffer&, const Buffer&, const Header&,
          std::size_t, std::size_t, std::size_t))
GPU_STUB(void, Prediction::copy, (const Buffer&, const Buffer&))
GPU_STUB(void, Prediction::finish, ())

struct DacGraph::Impl
{
};

DacGraph::DacGraph()
{
    disabled();
}

DacGraph::~DacGraph() = default;
GPU_STUB(DacTensor, DacGraph::upload, (std::span<const float>, SignalShape, const Limits&))
GPU_STUB(DacTensor, DacGraph::zeros, (SignalShape, const Limits&))
GPU_STUB(DacTensor, DacGraph::convolution,
         (const MetalConvolution&, const DacTensor&, const Limits&))
GPU_STUB(DacTensor, DacGraph::snake, (const DacTensor&, const Buffer&, const Limits&))
GPU_STUB(DacTensor, DacGraph::residual, (const DacTensor&, const DacTensor&, bool, const Limits&))
GPU_STUB(Buffer, DacGraph::search, (const DacTensor&, const RvqBuffers&, const Limits&, bool))
GPU_STUB(DacTensor, DacGraph::lookup, (const Buffer&, const RvqBuffers&, const Limits&))
GPU_STUB(DacTensor, DacGraph::project,
         (const Buffer&, const RvqBuffers&, const DacTensor&, bool, const Limits&))
GPU_STUB(void, DacGraph::finish, ())
GPU_STUB(void, DacGraph::begin, ())
GPU_STUB(Buffer, DacGraph::shared_temporary, (std::size_t))
GPU_STUB(void, DacGraph::release, (const DacTensor&))
GPU_STUB(Signal, DacGraph::download, (const DacTensor&, const Limits&))
} // namespace tsac::metal

namespace tsac
{
MetalAnalyzer::MetalAnalyzer(ModelDirectory directory, const Limits& limits)
    : directory_(std::move(directory)), limits_(limits)
{
    disabled();
}
GPU_STUB(Bytes, MetalAnalyzer::encode, (PcmView, const EncodeOptions&, const EntropyObserver&))
GPU_STUB(PcmAudio, MetalAnalyzer::decode, (ByteView, const EntropyObserver&))

struct MetalTransformer::Impl
{
};

MetalTransformer::MetalTransformer(std::unique_ptr<Impl> impl) : impl_(std::move(impl))
{
}

MetalTransformer::~MetalTransformer() = default;
MetalTransformer::MetalTransformer(MetalTransformer&&) noexcept = default;
MetalTransformer& MetalTransformer::operator=(MetalTransformer&&) noexcept = default;
GPU_STUB(MetalTransformer, MetalTransformer::from_model, (TransformerModel, const Limits&))
GPU_STUB(MetalTransformer, MetalTransformer::new_session, () const)
GPU_STUB(void, MetalTransformer::clear_session, ())
GPU_STUB(Bytes, MetalTransformer::encode,
         (Header, std::span<const std::uint16_t>, const EntropyObserver&))
GPU_STUB(std::vector<std::uint16_t>, MetalTransformer::decode, (ByteView, const EntropyObserver&))
GPU_STUB(MetalLinear, MetalLinear::from_weights, (const LinearWeights&))
GPU_STUB(MetalConvolution, MetalConvolution::from_weights,
         (Conv1dConfig, const ConvolutionWeights&, const Limits&))
GPU_STUB(MetalLinear, MetalLinear::prepare,
         (std::size_t, std::size_t, std::span<const std::int8_t>, std::span<const float>,
          std::span<const float>))
GPU_STUB(std::vector<std::uint16_t>, MetalLinear::apply, (std::span<const std::uint16_t>) const)
GPU_STUB(std::vector<metal::Buffer>, MetalLinear::resources, () const)
GPU_STUB(MetalConvolution, MetalConvolution::prepare,
         (Conv1dConfig, std::span<const float>, std::span<const float>))
GPU_STUB(MetalConvolution, MetalConvolution::from_prepared,
         (Conv1dConfig, const weights::View&, const weights::View&))
GPU_STUB(Signal, MetalConvolution::apply,
         (std::span<const float>, SignalShape, const Limits&) const)
GPU_STUB(Signal, metal_snake,
         (std::span<const float>, SignalShape, std::span<const float>, const Limits&))
GPU_STUB(std::vector<std::uint16_t>, metal_rvq_search,
         (std::span<const float>, std::span<const float>, const Limits&))
GPU_STUB(std::vector<float>, metal_residual,
         (std::span<const float>, std::span<const float>, bool, const Limits&))
GPU_STUB(std::vector<std::uint16_t>, metal_rvq_search_raw_projected,
         (std::span<const float>, std::span<const float>, const Limits&))
GPU_STUB(std::vector<float>, metal_rvq_project,
         (std::span<const std::uint16_t>, std::span<const float>, std::span<const float>,
          std::span<const float>, const Limits&))
} // namespace tsac

#undef GPU_STUB
