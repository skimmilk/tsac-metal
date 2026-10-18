#include "common/cpu_transformer.hpp"
#include "common/numeric.hpp"
#include "common/packed_linear.hpp"
#include "common/probability.hpp"
#include "models/contract.hpp"

#include <array>

namespace tsac
{
namespace
{
template <class Kernels> struct TransformerGraph
{
    using Linear = PackedLinear<Kernels>;
    using Bf16 = std::vector<std::uint16_t>;
    using Bf16Weights = weights::Array<std::uint16_t>;

    struct Workspace
    {
        typename Linear::Workspace linear;
        AttentionWorkspace attention;
        Bf16 normalized, qkv, query, key, projected, mlp;
        std::array<float, 512> values, squares;
    };

    static void add_in_place(Bf16& left, std::span<const std::uint16_t> right)
    {
        detail::require(left.size() == right.size(), "Transformer residual shape mismatch");
        Kernels::add(left, right);
    }

    struct Norm
    {
        Bf16Weights gain, bias;

        void apply_into(std::span<const std::uint16_t> input, Bf16& output,
                        Workspace& workspace) const
        {
            detail::require(input.size() == 512, "Transformer normalization width mismatch");
            output.resize(input.size());
            probability_layer_norm_bf16_into(input, gain, bias, output, workspace.values,
                                             workspace.squares);
        }
    };

    struct Layer
    {
        Norm first, second;
        Linear qkv, attention_out, mlp_in, mlp_out;
    };

    struct Stack
    {
        std::vector<Layer> layers;
        Norm final;
    };

    struct LayerCache
    {
        Bf16 key, value;
    };

    struct StackCache
    {
        std::vector<LayerCache> layers;
    };

    static void rotary(Bf16& vector, std::span<const std::uint16_t> coefficients)
    {
        detail::require(vector.size() == 512 && coefficients.size() == 128,
                        "rotary shape mismatch");
        for (std::size_t head = 0; head < 4; ++head)
        {
            auto values = std::span(vector).subspan(head * 128, 128);
            probability_rotary_bf16_into(values, coefficients, values);
        }
    }

    // Completed four-position tiles use [coordinate][position]; tails remain row-major.
    // Only integer BF16 payloads move, preserving cache sizes and numerical values.
    static void repack_key_tile(Bf16& key, std::size_t position)
    {
        if ((position + 1) % 4 == 0)
        {
            std::array<std::uint16_t, 4 * 512> tile;
            auto* source = key.data() + (position - 3) * 512;
            for (std::size_t j = 0; j < 512; ++j)
            {
                for (std::size_t t = 0; t < 4; ++t)
                {
                    tile[j * 4 + t] = source[t * 512 + j];
                }
            }
            std::copy(tile.begin(), tile.end(), source);
        }
    }

    static void evaluate_stack(const Stack& stack, StackCache& cache, Bf16& input,
                               std::size_t position, std::span<const std::uint16_t> rotation,
                               const Limits& limits, bool prefill, Workspace& workspace)
    {
        detail::require(input.size() == 512 && stack.layers.size() == cache.layers.size(),
                        "Transformer stack input mismatch");
        for (std::size_t i = 0; i < stack.layers.size(); ++i)
        {
            const auto& layer = stack.layers[i];
            auto& state = cache.layers[i];
            layer.first.apply_into(input, workspace.normalized, workspace);
            layer.qkv.apply_into(workspace.normalized, workspace.qkv, workspace.linear);
            const auto& qkv = workspace.qkv;
            detail::require(qkv.size() == 1536 && state.key.size() == position * 512 &&
                                state.value.size() == position * 512,
                            "Transformer cache position mismatch");
            auto& query = workspace.query;
            auto& key = workspace.key;
            query.assign(qkv.begin(), qkv.begin() + 512);
            key.assign(qkv.begin() + 512, qkv.begin() + 1024);
            if (!rotation.empty())
            {
                rotary(query, rotation);
                rotary(key, rotation);
            }
            detail::require(checked_mul(position + 1, 512) <= limits.tensor_elements,
                            "Transformer cache exceeds element limit");
            state.key.insert(state.key.end(), key.begin(), key.end());
            repack_key_tile(state.key, position);
            state.value.insert(state.value.end(), qkv.begin() + 1024, qkv.end());
            const auto attention =
                Kernels::attention_into(query, state.key, state.value, 4, 128, 0.0883883461356163f,
                                        prefill, true, workspace.attention);
            layer.attention_out.apply_into(attention, workspace.projected, workspace.linear);
            add_in_place(input, workspace.projected);
            layer.second.apply_into(input, workspace.normalized, workspace);
            layer.mlp_in.apply_into(workspace.normalized, workspace.mlp, workspace.linear);
            probability_gelu_bf16(workspace.mlp);
            layer.mlp_out.apply_into(workspace.mlp, workspace.projected, workspace.linear);
            add_in_place(input, workspace.projected);
        }
        stack.final.apply_into(input, input, workspace);
    }
};
} // namespace

template <class Kernels> struct CpuTransformer<Kernels>::Impl
{
    using Graph = TransformerGraph<Kernels>;
    using Bf16 = typename Graph::Bf16;
    using Bf16Weights = typename Graph::Bf16Weights;
    using Stack = typename Graph::Stack;
    using StackCache = typename Graph::StackCache;
    unsigned channels, maximum_books;
    Limits limits;
    Stack temporal, codebook;
    PackedLinear<Kernels> temporal_projection, codebook_projection, output_projection;
    Bf16Weights embeddings, positions, rotations;

    // Immutable prepared weights; independent mutable CPU caches per lane.
    struct Lane
    {
        StackCache temporal, codebook;
        Bf16 conditioning, input, embedded, prefix, prefix_tokens, logits, distribution;
        std::array<float, 1024> exponents;
        typename Graph::Workspace workspace;
        std::vector<float> probabilities;
    };

    std::vector<Lane> lanes;
    Header header;

    void embed_into(std::span<const std::uint16_t> tokens, Bf16& result) const
    {
        detail::require(!tokens.empty() && tokens.size() <= maximum_books,
                        "invalid Transformer feedback shape");
        result.resize(1024);
        for (std::size_t book = 0; book < tokens.size(); ++book)
        {
            detail::require(tokens[book] < 1024, "Transformer feedback token exceeds alphabet");
            const auto embedding =
                std::span(embeddings).subspan((book * 1024 + tokens[book]) * 1024, 1024);
            if (book == 0)
            {
                std::copy(embedding.begin(), embedding.end(), result.begin());
            }
            else
            {
                Graph::add_in_place(result, embedding);
            }
        }
    }
};

template <class Kernels>
CpuTransformer<Kernels>::CpuTransformer(std::unique_ptr<Impl> impl) : impl_(std::move(impl))
{
}

template <class Kernels> CpuTransformer<Kernels>::~CpuTransformer() = default;
template <class Kernels>
CpuTransformer<Kernels>::CpuTransformer(CpuTransformer&&) noexcept = default;
template <class Kernels>
CpuTransformer<Kernels>& CpuTransformer<Kernels>::operator=(CpuTransformer&&) noexcept = default;

template <class Kernels>
CpuTransformer<Kernels> CpuTransformer<Kernels>::from_model(TransformerModel transformer,
                                                            const Limits& limits)
{
    using Graph = TransformerGraph<Kernels>;
    using Norm = typename Graph::Norm;
    using Stack = typename Graph::Stack;
    using Bf16Weights = typename Graph::Bf16Weights;
    const auto norm = [&](const NormalizationWeights& n)
    {
        return Norm{Bf16Weights(n.gain), Bf16Weights(n.bias)};
    };
    const auto linear = [&](const LinearWeights& weights)
    {
        return PackedLinear<Kernels>::from_weights(weights, limits);
    };
    const auto stack = [&](TransformerStack kind)
    {
        Stack result;
        for (unsigned layer = 0; layer < model_contract::layers(kind); ++layer)
        {
            result.layers.push_back({norm(transformer.attention_normalization(kind, layer)),
                                     norm(transformer.mlp_normalization(kind, layer)),
                                     linear(transformer.attention_qkv(kind, layer)),
                                     linear(transformer.attention_output(kind, layer)),
                                     linear(transformer.mlp_input(kind, layer)),
                                     linear(transformer.mlp_output(kind, layer))});
        }
        result.final = norm(transformer.final_normalization(kind));
        return result;
    };
    auto impl = std::make_unique<Impl>(Impl{model_contract::channels(transformer.variant()),
                                            transformer.codebooks(),
                                            limits,
                                            stack(TransformerStack::temporal),
                                            stack(TransformerStack::codebook),
                                            linear(transformer.temporal_projection()),
                                            linear(transformer.codebook_projection()),
                                            linear(transformer.output_projection()),
                                            Bf16Weights(transformer.feedback_embeddings()),
                                            Bf16Weights(transformer.positions()),
                                            Bf16Weights(transformer.rotations()),
                                            {},
                                            {}});
    return CpuTransformer<Kernels>(std::move(impl));
}

template <class Kernels>
void CpuTransformer<Kernels>::reset(const Header& header, std::size_t count)
{
    validate_header(header, impl_->limits);
    detail::require(header.model_channels == impl_->channels &&
                        header.codebooks <= impl_->maximum_books &&
                        count == std::min<std::size_t>(header.batch, header.frames) * header.groups,
                    "Transformer stream/model mismatch");
    const auto base = header.frames / header.batch;
    const auto extra = header.frames % header.batch;
    const auto contexts = checked_mul(
        header.groups,
        checked_add(checked_mul(header.batch - extra, std::min<std::size_t>(base, 1024)),
                    checked_mul(extra, std::min<std::size_t>(base + 1, 1024))));
    const auto temporal_bytes =
        checked_mul(contexts, std::size_t{12} * 512 * 2 * sizeof(std::uint16_t));
    const auto codebook_bytes = checked_mul(checked_mul(count, header.codebooks),
                                            std::size_t{3} * 512 * 2 * sizeof(std::uint16_t));
    detail::require(
        checked_add(temporal_bytes, codebook_bytes) <= impl_->limits.cache_bytes,
        "Transformer caches exceed byte limit; reduce the encoding batch size or channel count");
    impl_->header = header;
    impl_->lanes.clear();
    impl_->lanes.resize(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        auto& lane = impl_->lanes[i];
        lane.temporal.layers.resize(12);
        lane.codebook.layers.resize(3);
        const auto extent = std::min<std::size_t>(base + (i / header.groups < extra), 1024);
        detail::require(extent * 512 <= impl_->limits.tensor_elements &&
                            std::size_t(header.codebooks) * 512 <= impl_->limits.tensor_elements,
                        "Transformer cache exceeds element limit");
        for (auto& layer : lane.temporal.layers)
        {
            layer.key.reserve(extent * 512);
            layer.value.reserve(extent * 512);
        }
        for (auto& layer : lane.codebook.layers)
        {
            layer.key.reserve(std::size_t(header.codebooks) * 512);
            layer.value.reserve(std::size_t(header.codebooks) * 512);
        }
    }
}

template <class Kernels> void CpuTransformer<Kernels>::clear_session()
{
    impl_->lanes.clear();
}

template <class Kernels> bool CpuTransformer<Kernels>::independent_lanes() const
{
    return true;
}

template <class Kernels> void CpuTransformer<Kernels>::reset_temporal(std::size_t index)
{
    auto& state = impl_->lanes.at(index).temporal;
    for (auto& layer : state.layers)
    {
        layer.key.clear();
        layer.value.clear();
    }
}

template <class Kernels>
void CpuTransformer<Kernels>::temporal_step(std::size_t index, std::size_t position,
                                            std::span<const std::uint16_t> feedback, bool replay)
{
    detail::require(position < 1024 && feedback.size() == impl_->header.codebooks,
                    "invalid temporal step");
    auto& lane = impl_->lanes.at(index);
    impl_->embed_into(feedback, lane.embedded);
    impl_->temporal_projection.apply_into(lane.embedded, lane.conditioning, lane.workspace.linear);
    TransformerGraph<Kernels>::evaluate_stack(
        impl_->temporal, lane.temporal, lane.conditioning, position,
        std::span(impl_->rotations).subspan(position * 128, 128), impl_->limits, replay,
        lane.workspace);
}

template <class Kernels> void CpuTransformer<Kernels>::reset_codebook(std::size_t index)
{
    auto& lane = impl_->lanes.at(index);
    lane.prefix_tokens.clear();
    auto& state = lane.codebook;
    for (auto& layer : state.layers)
    {
        layer.key.clear();
        layer.value.clear();
    }
}

template <class Kernels>
std::span<const float> CpuTransformer<Kernels>::codebook_step(std::size_t index, std::size_t book,
                                                              std::span<const std::uint16_t> prefix)
{
    detail::require(book < impl_->header.codebooks && prefix.size() == book,
                    "invalid codebook step");
    auto& lane = impl_->lanes.at(index);
    if (book == 0)
    {
        lane.input = lane.conditioning;
    }
    else
    {
        // Preserve the copy of book zero and each ordered BF16 rounding
        // boundary. Rebuild if a direct backend caller changes an earlier token.
        if (prefix.size() == lane.prefix_tokens.size() + 1 && !lane.prefix_tokens.empty() &&
            std::equal(lane.prefix_tokens.begin(), lane.prefix_tokens.end(), prefix.begin()))
        {
            detail::require(prefix.back() < 1024, "Transformer feedback token exceeds alphabet");
            TransformerGraph<Kernels>::add_in_place(
                lane.prefix, std::span(impl_->embeddings)
                                 .subspan(((book - 1) * 1024 + prefix.back()) * 1024, 1024));
        }
        else
        {
            impl_->embed_into(prefix, lane.prefix);
        }
        lane.prefix_tokens.assign(prefix.begin(), prefix.end());
        impl_->codebook_projection.apply_into(lane.prefix, lane.input, lane.workspace.linear);
    }
    TransformerGraph<Kernels>::add_in_place(lane.input,
                                            std::span(impl_->positions).subspan(book * 512, 512));
    TransformerGraph<Kernels>::evaluate_stack(impl_->codebook, lane.codebook, lane.input, book, {},
                                              impl_->limits, false, lane.workspace);
    impl_->output_projection.apply_into(lane.input, lane.logits, lane.workspace.linear);
    lane.distribution.resize(1024);
    probability_softmax_bf16_into(lane.logits, lane.distribution, lane.exponents);
    lane.probabilities.resize(1024);
    std::transform(lane.distribution.begin(), lane.distribution.end(), lane.probabilities.begin(),
                   probability_float);
    return lane.probabilities;
}
} // namespace tsac
