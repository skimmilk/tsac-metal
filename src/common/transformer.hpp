#pragma once
#include "common/execution.hpp"
#include "format/entropy.hpp"

#include <deque>
#include <optional>

namespace tsac
{
// Numerical boundary for the original two-stack probability model. This is an
// interface, not an approximate CPU implementation. A production implementation
// must validate its model role and reproduce the captured BF16/BF8L kernels.
// Each independent_lane identifies one entropy lane AND one channel group.
class TransformerBackend
{
public:
    virtual ~TransformerBackend() = default;

    // Opt in only when operations on distinct lanes are thread-safe and do not
    // invalidate another lane's borrowed distribution. Reset remains exclusive.
    virtual bool independent_lanes() const
    {
        return false;
    }

    // Clears both stacks, all lane state, and all previously borrowed output.
    virtual void reset(const Header& header, std::size_t independent_lanes) = 0;
    virtual void reset_temporal(std::size_t independent_lane) = 0;
    // feedback is the complete preceding frame's codebooks (zero indices before
    // the first frame). Form DAC embeddings in codebook order, including index 0,
    // with captured BF16 addition boundaries, then cb_proj and the temporal stack.
    // The last replay output is the temporal conditioning for the current frame.
    virtual void temporal_step(std::size_t independent_lane, std::size_t context_position,
                               std::span<const std::uint16_t> feedback, bool replay) = 0;
    virtual void reset_codebook(std::size_t independent_lane) = 0;
    // book 0 consumes the temporal output directly. Later books use the cumulative
    // DAC embedding of committed_prefix via codebook_decoder.cb_proj. Apply stored
    // wpe, three-layer codebook stack, proj_f and the captured softmax operation.
    // Distribution storage remains borrowed until the next backend operation.
    virtual std::span<const float>
    codebook_step(std::size_t independent_lane, std::size_t book,
                  std::span<const std::uint16_t> committed_prefix) = 0;
};

// Shared state/scheduling for the validated numerical backends. The isolated
// scheduling tests also accept synthetic backends to expose lane/cache events.
// A backend exception invalidates the current session; call reset() before reuse.
class TransformerPredictor final : public TokenPredictor
{
public:
    explicit TransformerPredictor(TransformerBackend& backend, const Limits& limits = {},
                                  const ParallelExecutor* executor = nullptr);
    void reset(const Header& header) override;
    std::span<const float> predict(const EntropyPosition& position) override;
    void commit(const EntropyPosition& position, std::uint16_t token) override;

private:
    struct Lane
    {
        std::size_t completed_frames = 0, next_book = 0, context_used = 0;
        std::vector<std::uint16_t> current;
        std::vector<float> probabilities;
        bool prepared = false;
        std::deque<std::vector<std::uint16_t>> history;
    };

    void prepare(std::size_t index);
    std::size_t lane_index(const EntropyPosition& position) const;
    TransformerBackend& backend_;
    const ParallelExecutor* executor_;
    Limits limits_;
    Header header_;
    bool initialized_ = false;
    std::vector<Lane> lanes_;
    std::vector<std::size_t> ready_;
    std::optional<EntropyPosition> pending_;
};
} // namespace tsac
