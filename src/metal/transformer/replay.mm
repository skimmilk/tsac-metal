#include "common/internal.hpp"
#include "session.hpp"

#include <algorithm>

namespace tsac::metal::transformer
{
void Session::replay_temporal(std::span<const Replay> replay)
{
    detail::require(!replay.empty() && replay.size() <= batch_columns,
                    "Invalid Metal replay batch");
    std::vector<StackCache*> caches;
    for (const auto& r : replay)
    {
        detail::require(r.tokens.size() == checked_mul(replay_frames, header.codebooks),
                        "Invalid Metal replay token count");
        reset_temporal(r.lane);
        caches.push_back(&lanes.at(r.lane).temporal);
    }
    std::vector<std::size_t> positions(replay.size());
    std::vector<std::uint16_t> tokens;
    tokens.reserve(checked_mul(replay.size(), header.codebooks));
    recorder.begin();
    for (std::size_t step = 0; step < replay_frames; ++step)
    {
        // Only GPU scratch is recycled. Each CPU upload occupies a distinct
        // staging range until this entire bounded submission has completed.
        recorder.recycle();
        tokens.clear();
        for (const auto& r : replay)
        {
            const auto frame = r.tokens.subspan(step * header.codebooks, header.codebooks);
            tokens.insert(tokens.end(), frame.begin(), frame.end());
        }
        auto input = recorder.linear(model.temporal_projection,
                                     recorder.embed(model.embeddings, tokens, replay.size()));
        std::fill(positions.begin(), positions.end(), step);
        const auto output = evaluate_batch(recorder, model.temporal, caches, input, positions,
                                           model.rotations, true, {}, {}, 0, true, true);
        if (step + 1 == replay_frames)
        {
            for (std::size_t c = 0; c < replay.size(); ++c)
            {
                recorder.copy(output.slice(c * 1024, 1024), lanes[replay[c].lane].conditioning);
            }
        }
        if ((step + 1) % submission_steps == 0 || step + 1 == replay_frames)
        {
            recorder.finish();
            recorder.begin();
        }
    }
}
} // namespace tsac::metal::transformer
