#pragma once
#include "format/container.hpp"
#include "metal/linear.hpp"
#include "metal/runtime.hpp"

namespace tsac::metal
{
// Reusable recorder for a prediction or a bounded decoding chunk. Intermediate
// activations remain on Metal, with one completion boundary per recorded graph.
// One exclusive caller owns recording and scratch reuse. Prepared weights can
// be shared; recorded buffers and the error word survive graph completion.
class Prediction
{
public:
    struct ResidentPosition
    {
        std::size_t step_index;
        bool initial_feedback = false;
    };

    Prediction();
    explicit Prediction(std::shared_ptr<const ResidencySet> model_resources);
    ~Prediction();
    Prediction(const Prediction&) = delete;
    Prediction& operator=(const Prediction&) = delete;
    void begin();
    // Drop recorded references after completion while retaining reusable arenas.
    void reset_session(std::shared_ptr<const ResidencySet> session_resources);
    // Enable geometry-selected full preparation/projection kernels. Wider
    // batches retain shared preparation when it is faster.
    void fuse_projections(bool enabled);
    // Reuse activation slots after their last recorded consumer. Does not
    // discard commands or clear the error word; no CPU writes may alias them.
    void recycle();
    Buffer temporary(std::size_t bytes);
    // Shared staging has a separate cursor that recycle() never rewinds.
    Buffer shared_temporary(std::size_t bytes);
    Buffer upload(std::span<const std::uint16_t> values);
    Buffer embed(const Buffer& embeddings, std::span<const std::uint16_t> tokens,
                 std::size_t columns = 1);
    Buffer embed_resident(const Buffer& embeddings, const Buffer& tokens, const Header& header,
                          std::size_t step, std::size_t prefix, std::size_t first,
                          std::size_t columns);
    Buffer embed_resident(const Buffer& embeddings, const Buffer& tokens, const Header& header,
                          ResidentPosition position, std::size_t prefix, std::size_t first,
                          std::size_t columns);
    Buffer embed_linear_resident(const MetalLinear& weights, const Buffer& embeddings,
                                 const Buffer& tokens, const Header& header, std::size_t step,
                                 std::size_t prefix, std::size_t first, std::size_t columns);
    Buffer embed_linear_resident(const MetalLinear& weights, const Buffer& embeddings,
                                 const Buffer& tokens, const Header& header,
                                 ResidentPosition position, std::size_t prefix, std::size_t first,
                                 std::size_t columns);
    Buffer add(const Buffer& left, const Buffer& right, std::size_t count);
    Buffer norm(const Buffer& input, const Buffer& gain, const Buffer& bias);
    Buffer linear(const MetalLinear& weights, const Buffer& input, bool activate_gelu = false,
                  const Buffer* residual = nullptr);
    Buffer norm_linear(const MetalLinear& weights, const Buffer& input, const Buffer& gain,
                       const Buffer& bias);
    Buffer attention(const Buffer& qkv, const Buffer& keys, const Buffer& values,
                     const Buffer& rotation, std::size_t position, bool rotated, bool prefill,
                     const Buffer* destination = nullptr);
    Buffer attention_columns(const Buffer& qkv, const Buffer& keys, const Buffer& values,
                             std::size_t position, std::size_t columns, std::size_t cache_frames);
    Buffer attention_temporal(const Buffer& qkv, const Buffer& keys, const Buffer& values,
                              const Buffer& rotation, std::size_t position, std::size_t columns,
                              std::size_t cache_frames);
    Buffer softmax(const Buffer& input, const Buffer* destination = nullptr);
    void softmax_decode(const Buffer& input, const Buffer& probabilities, const Buffer& tree);
    void range_decode(const Buffer& probabilities, const Buffer& tree, const Buffer& payload,
                      const Buffer& state, const Buffer& tokens, const Header& header,
                      std::size_t step, std::size_t book, std::size_t columns);
    void copy(const Buffer& source, const Buffer& destination);
    void finish();

private:
    // Packed three-column projections consume sign-extended int32 activations.
    enum class QuantizedStorage
    {
        bytes,
        words
    };

    struct QuantizedActivations
    {
        Buffer values, scales;
        QuantizedStorage storage;
    };

    QuantizedActivations quantized_temporary(const MetalLinear& weights, std::size_t columns);
    Buffer quantized_linear(const MetalLinear& weights, const QuantizedActivations& input,
                            std::size_t columns, const Buffer* residual = nullptr);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace tsac::metal
