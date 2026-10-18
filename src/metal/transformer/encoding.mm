#include "common/internal.hpp"
#include "common/numeric/bf16.hpp"
#include "session.hpp"

#include <algorithm>
#include <deque>
#include <optional>

namespace tsac::metal::transformer
{
namespace
{
bool same_position(const EntropyPosition& a, const EntropyPosition& b)
{
    return a.frame == b.frame && a.group == b.group && a.codebook == b.codebook &&
           a.lane == b.lane && a.step == b.step && a.token_offset == b.token_offset;
}

// Private adapter to the shared container/range traversal. All prediction and
// replay scheduling belongs to this GPU session, with one recording owner.
class Encoder final : public TokenPredictor
{
public:
    Encoder(Session& backend, const Limits& limits);
    void reset(const Header& header) override;
    void known_tokens(std::span<const std::uint16_t> tokens) override;
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

    void prepare_known(const EntropyPosition& position);
    std::size_t lane_index(const EntropyPosition& position) const;
    Session& backend_;
    Limits limits_;
    Header header_;
    bool initialized_ = false;
    std::vector<Lane> lanes_;
    std::optional<EntropyPosition> pending_;
    std::span<const std::uint16_t> known_tokens_;
    std::optional<Buffer> known_probabilities_;
    std::vector<std::size_t> known_lanes_;
    std::size_t known_start_ = 0, known_steps_ = 0;
};

Encoder::Encoder(Session& backend, const Limits& limits) : backend_(backend), limits_(limits)
{
}

void Encoder::reset(const Header& header)
{
    initialized_ = false;
    known_tokens({});
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

std::size_t Encoder::lane_index(const EntropyPosition& p) const
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

void Encoder::known_tokens(std::span<const std::uint16_t> tokens)
{
    if (!tokens.empty())
    {
        detail::require(initialized_ && tokens.size() == header_.symbol_count(),
                        "known-token shape mismatch");
        for (auto token : tokens)
        {
            detail::require(token < 1024, "known token exceeds alphabet");
        }
    }
    known_tokens_ = tokens;
    known_probabilities_.reset();
    known_lanes_.clear();
    known_steps_ = 0;
}

void Encoder::prepare_known(const EntropyPosition& p)
{
    known_lanes_.clear();
    if (lanes_.size() > batch_columns)
    {
        known_steps_ = 0;
        return;
    }
    known_steps_ = lookahead_steps;
    known_start_ = p.step;
    for (std::size_t i = 0; i < lanes_.size(); ++i)
    {
        const auto length =
            header_.frames / header_.batch + (i / header_.groups < header_.frames % header_.batch);
        auto& lane = lanes_[i];
        if (p.step >= length)
        {
            continue;
        }
        detail::require(lane.completed_frames == p.step && lane.next_book == 0,
                        "known-token lane mismatch");
        known_lanes_.push_back(i);
        known_steps_ =
            std::min({known_steps_, length - p.step, context_frames - lane.context_used});
    }
    if (!known_steps_)
    {
        return; // Replay is recorded by the exclusive session owner.
    }
    // Bound lookahead and independent codebook caches even with many lanes.
    known_steps_ =
        std::min(known_steps_, std::max<std::size_t>(1, batch_columns / known_lanes_.size()));
    const auto columns = checked_mul(known_steps_, known_lanes_.size());
    const auto elements = checked_mul(checked_mul(columns, header_.codebooks), 1024);
    const auto persistent_bytes = backend_.cache_bytes();
    // Lookahead's per-frame codebook caches and BF16 distributions are in
    // addition to the persistent lane caches already allocated by reset().
    if (elements > limits_.tensor_elements ||
        checked_add(persistent_bytes, checked_mul(elements, 8)) > limits_.cache_bytes)
    {
        known_steps_ = 0;
        return;
    }
    std::vector<Session::Request> requests;
    std::vector<std::uint16_t> zeros(header_.codebooks);
    for (std::size_t t = 0; t < known_steps_; ++t)
    {
        for (auto i : known_lanes_)
        {
            const auto lane = i / header_.groups, group = i % header_.groups;
            const auto frame = lane * (header_.frames / header_.batch) +
                               std::min(lane, header_.frames % header_.batch) + p.step + t;
            const auto current = known_tokens_.subspan(
                (frame * header_.groups + group) * header_.codebooks, header_.codebooks);
            const auto feedback =
                p.step + t == 0 ? std::span<const std::uint16_t>(zeros)
                                : known_tokens_.subspan(((frame - 1) * header_.groups + group) *
                                                            header_.codebooks,
                                                        header_.codebooks);
            requests.push_back({i, lanes_[i].context_used + t, 0, feedback, current});
        }
    }
    known_probabilities_ = backend_.predict_known(requests, known_lanes_.size());
    detail::require(known_probabilities_->size() == elements * sizeof(std::uint16_t),
                    "known-token result shape mismatch");
}

std::span<const float> Encoder::predict(const EntropyPosition& p)
{
    detail::require(!pending_.has_value(), "Transformer prediction needs its token committed");
    const auto index = lane_index(p);
    auto& lane = lanes_[index];
    detail::require(p.step == lane.completed_frames && p.codebook == lane.next_book,
                    "out-of-order Transformer prediction");
    if (!known_tokens_.empty())
    {
        try
        {
            if (p.codebook == 0 &&
                (p.step < known_start_ || p.step >= known_start_ + known_steps_) && !lane.prepared)
            {
                prepare_known(p);
            }
            if (known_steps_ && p.step >= known_start_ && p.step < known_start_ + known_steps_)
            {
                const auto it = std::find(known_lanes_.begin(), known_lanes_.end(), index);
                detail::require(it != known_lanes_.end(), "known-token lane missing");
                const auto column = (p.step - known_start_) * known_lanes_.size() +
                                    std::size_t(it - known_lanes_.begin());
                const auto offset =
                    (p.codebook * known_steps_ * known_lanes_.size() + column) * 1024;
                lane.probabilities.resize(1024);
                const auto probabilities = known_probabilities_->mapped<std::uint16_t>();
                std::transform(probabilities.begin() + static_cast<std::ptrdiff_t>(offset),
                               probabilities.begin() + static_cast<std::ptrdiff_t>(offset + 1024),
                               lane.probabilities.begin(), probability_float);
                if (p.codebook == 0)
                {
                    lane.current.assign(header_.codebooks, 0);
                    ++lane.context_used;
                }
                pending_ = p;
                return lane.probabilities;
            }
        }
        catch (...)
        {
            initialized_ = false;
            throw;
        }
    }
    if (!lane.prepared)
    {
        try
        {
            std::vector<std::size_t> ready;
            for (std::size_t i = 0; i < lanes_.size(); ++i)
            {
                const auto& candidate = lanes_[i];
                const auto length = header_.frames / header_.batch +
                                    (i / header_.groups < header_.frames % header_.batch);
                if (!candidate.prepared && candidate.completed_frames == p.step &&
                    candidate.next_book == p.codebook && p.step < length)
                {
                    ready.push_back(i);
                }
            }
            // Bound scratch independently of the stream's lane count. The
            // requested lane must be in this batch even for direct callers
            // visiting independent lanes in a different order.
            if (ready.size() > batch_columns)
            {
                const auto requested = std::find(ready.begin(), ready.end(), index);
                if (requested >= ready.begin() + batch_columns)
                {
                    std::iter_swap(ready.begin(), requested);
                }
                ready.resize(batch_columns);
            }
            // Prepare independent columns together; replay stays on this
            // same recording owner and never enters the CPU worker pool.
            const bool replay =
                p.codebook == 0 && std::any_of(ready.begin(), ready.end(),
                                               [&](std::size_t i)
                                               {
                                                   return lanes_[i].context_used == context_frames;
                                               });
            if (replay)
            {
                std::vector<std::vector<std::uint16_t>> history;
                std::vector<Session::Replay> replay_lanes;
                history.reserve(ready.size());
                for (auto i : ready)
                {
                    auto& state = lanes_[i];
                    detail::require(state.context_used == context_frames &&
                                        state.history.size() == replay_frames,
                                    "Incomplete Metal replay history");
                    auto& packed = history.emplace_back();
                    packed.reserve(checked_mul(replay_frames, header_.codebooks));
                    for (const auto& frame : state.history)
                    {
                        packed.insert(packed.end(), frame.begin(), frame.end());
                    }
                    replay_lanes.push_back({i, packed});
                }
                backend_.replay_temporal(replay_lanes);
                for (auto i : ready)
                {
                    lanes_[i].context_used = replay_frames;
                }
            }
            std::vector<Session::Request> requests;
            for (auto i : ready)
            {
                auto& state = lanes_[i];
                if (p.codebook == 0)
                {
                    state.current.assign(header_.codebooks, 0);
                }
                const auto feedback = state.history.empty()
                                          ? std::span<const std::uint16_t>(state.current)
                                          : std::span<const std::uint16_t>(state.history.back());
                requests.push_back(
                    {i, state.context_used, state.next_book, feedback,
                     std::span<const std::uint16_t>(state.current).first(state.next_book), replay});
            }
            const auto results = backend_.predict_batch(requests);
            detail::require(results.size() == ready.size(), "Transformer batch result mismatch");
            for (std::size_t j = 0; j < ready.size(); ++j)
            {
                auto& state = lanes_[ready[j]];
                detail::require(results[j].size() == 1024, "Transformer backend alphabet mismatch");
                state.probabilities.assign(results[j].begin(), results[j].end());
                state.prepared = true;
                if (p.codebook == 0 && !replay)
                {
                    ++state.context_used;
                }
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

void Encoder::commit(const EntropyPosition& p, std::uint16_t token)
{
    detail::require(pending_.has_value() && same_position(*pending_, p),
                    "Transformer commit does not match prediction");
    detail::require(token < 1024, "Transformer feedback token exceeds alphabet");
    if (!known_tokens_.empty())
    {
        detail::require(token == known_tokens_[p.token_offset], "known-token commit mismatch");
    }
    auto& lane = lanes_[lane_index(p)];
    lane.prepared = false;
    lane.current[p.codebook] = token;
    ++lane.next_book;
    // Only a complete frame becomes temporal feedback. Keep the last 256
    // frames for the 1024-position context reset/replay.
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
} // namespace

Bytes Session::encode(Header header, std::span<const std::uint16_t> tokens,
                      const EntropyObserver& observer)
{
    Encoder encoder(*this, model.limits);
    return pack_normal(header, tokens, encoder, model.limits, observer);
}
} // namespace tsac::metal::transformer
