#include "common/transformer.hpp"

#include "common/internal.hpp"

#include <algorithm>

namespace tsac
{
namespace
{
constexpr std::size_t context = 1024, replay_frames = 256;

bool same_position(const EntropyPosition& a, const EntropyPosition& b)
{
    return a.frame == b.frame && a.group == b.group && a.codebook == b.codebook &&
           a.lane == b.lane && a.step == b.step && a.token_offset == b.token_offset;
}
} // namespace

TransformerPredictor::TransformerPredictor(TransformerBackend& backend, const Limits& limits,
                                           const ParallelExecutor* executor)
    : backend_(backend), executor_(executor), limits_(limits)
{
}

void TransformerPredictor::reset(const Header& header)
{
    initialized_ = false;
    pending_.reset();
    lanes_.clear();
    validate_header(header, limits_);
    detail::require(header.method == 1, "Transformer predictor requires normal framing");
    const auto independent =
        checked_mul(std::min<std::size_t>(header.batch, header.frames), header.groups);
    detail::require(independent <= limits_.tensor_elements, "Transformer lane count exceeds limit");
    backend_.reset(header, independent);
    header_ = header;
    lanes_.resize(independent);
    initialized_ = true;
}

std::size_t TransformerPredictor::lane_index(const EntropyPosition& p) const
{
    detail::require(initialized_, "Transformer predictor is not initialized");
    detail::require(p.lane < std::min<std::size_t>(header_.batch, header_.frames) &&
                        p.group < header_.groups && p.codebook < header_.codebooks,
                    "invalid Transformer entropy position");
    const auto base = header_.frames / header_.batch;
    const auto extra = header_.frames % header_.batch;
    const auto length = base + (p.lane < extra);
    const auto start = p.lane * base + std::min(p.lane, extra);
    detail::require(p.step < length && p.frame == start + p.step &&
                        p.token_offset ==
                            (p.frame * header_.groups + p.group) * header_.codebooks + p.codebook,
                    "inconsistent Transformer entropy position");
    return p.lane * header_.groups + p.group;
}

std::span<const float> TransformerPredictor::predict(const EntropyPosition& p)
{
    detail::require(!pending_.has_value(), "Transformer prediction needs its token committed");
    const auto index = lane_index(p);
    auto& lane = lanes_[index];
    detail::require(p.step == lane.completed_frames && p.codebook == lane.next_book,
                    "out-of-order Transformer prediction");
    if (!lane.prepared)
    {
        try
        {
            if (executor_ && executor_->parallel() && backend_.independent_lanes())
            {
                ready_.clear();
                for (std::size_t i = 0; i < lanes_.size(); ++i)
                {
                    const auto& candidate = lanes_[i];
                    const auto length = header_.frames / header_.batch +
                                        (i / header_.groups < header_.frames % header_.batch);
                    if (!candidate.prepared && candidate.completed_frames == p.step &&
                        candidate.next_book == p.codebook && p.step < length)
                    {
                        ready_.push_back(i);
                    }
                }
                // Prepare independent CPU lanes; entropy commits remain serial.
                executor_->parallel_for(ready_.size(),
                                        [&](std::size_t i)
                                        {
                                            prepare(ready_[i]);
                                        });
            }
            else
            {
                prepare(index);
            }
        }
        catch (...)
        {
            initialized_ = false;
            throw;
        }
    }
    pending_ = p;
    return lane.probabilities;
}

void TransformerPredictor::prepare(std::size_t index)
{
    auto& lane = lanes_[index];
    if (lane.next_book == 0)
    {
        lane.current.assign(header_.codebooks, 0);
        if (lane.context_used == context)
        {
            detail::require(lane.history.size() == replay_frames,
                            "incomplete Transformer replay history");
            backend_.reset_temporal(index);
            lane.context_used = 0;
            // Original resets the temporal cache and feeds the last 256 complete
            // token frames; the final replay result predicts the current frame.
            // Do not feed the current feedback a second time after replay.
            for (const auto& feedback : lane.history)
            {
                backend_.temporal_step(index, lane.context_used++, feedback, true);
            }
        }
        else
        {
            const auto feedback = lane.history.empty()
                                      ? std::span<const std::uint16_t>(lane.current)
                                      : std::span<const std::uint16_t>(lane.history.back());
            backend_.temporal_step(index, lane.context_used++, feedback, false);
        }
        backend_.reset_codebook(index);
    }
    const auto distribution = backend_.codebook_step(
        index, lane.next_book, std::span<const std::uint16_t>(lane.current).first(lane.next_book));
    detail::require(distribution.size() == 1024, "Transformer backend alphabet mismatch");
    // RvqInputLayout spans are borrowed. Retain the result until this lane's turn in
    // stream order, even if another lane is prepared or committed first.
    lane.probabilities.assign(distribution.begin(), distribution.end());
    lane.prepared = true;
}

void TransformerPredictor::commit(const EntropyPosition& p, std::uint16_t token)
{
    detail::require(pending_.has_value() && same_position(*pending_, p),
                    "Transformer commit does not match prediction");
    detail::require(token < 1024, "Transformer feedback token exceeds alphabet");
    auto& lane = lanes_[lane_index(p)];
    lane.prepared = false;
    lane.current[p.codebook] = token;
    ++lane.next_book;
    // Only a complete frame becomes temporal feedback. Keep the last 256
    // frames for the 1024-position context reset/replay in prepare().
    if (lane.next_book == header_.codebooks)
    {
        lane.history.push_back(std::move(lane.current));
        if (lane.history.size() > replay_frames)
        {
            lane.history.pop_front();
        }
        lane.next_book = 0;
        ++lane.completed_frames;
    }
    pending_.reset();
}
} // namespace tsac
