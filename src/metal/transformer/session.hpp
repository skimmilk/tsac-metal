#pragma once

#include "format/entropy.hpp"
#include "metal/prediction.hpp"
#include "model.hpp"

namespace tsac::metal::transformer
{
inline constexpr std::size_t context_frames = 1024, replay_frames = 256;
inline constexpr std::size_t batch_columns = 96, lookahead_steps = 32, submission_steps = 10;

struct LayerCache
{
    Buffer key, value;
    std::size_t position = 0;
};

struct StackCache
{
    std::vector<LayerCache> layers;
};

Buffer evaluate_stack(Prediction& graph, const Stack& stack, StackCache& cache, Buffer input,
                      std::size_t position, const Buffer& rotation, bool rotated, bool replay);
Buffer evaluate_batch(Prediction& graph, const Stack& stack, std::span<StackCache*> caches,
                      Buffer input, std::span<const std::size_t> positions, const Buffer& rotations,
                      bool rotated, std::span<const Buffer> bulk_keys = {},
                      std::span<const Buffer> bulk_values = {}, std::size_t cache_frames = 0,
                      bool final_norm = true, bool replay = false);

// Private GPU scheduler boundary, with no CPU backend capabilities or worker pool.
class Session
{
public:
    explicit Session(std::shared_ptr<const PreparedModel> prepared);
    Bytes encode(Header header, std::span<const std::uint16_t> tokens,
                 const EntropyObserver& observer);
    std::vector<std::uint16_t> decode(ByteView stream, const EntropyObserver& observer);

    struct Request
    {
        std::size_t lane, position, book;
        std::span<const std::uint16_t> feedback, prefix;
        bool use_conditioning = false;
    };

    void reset(const Header& header, std::size_t count);

    struct Replay
    {
        std::size_t lane;
        std::span<const std::uint16_t> tokens;
    };

    void replay_temporal(std::span<const Replay> replay);
    void reset_temporal(std::size_t index);
    void reset_codebook(std::size_t index);
    std::vector<std::span<const float>> predict_batch(std::span<const Request> requests);
    Buffer predict_known(std::span<const Request> requests, std::size_t lane_count);
    std::size_t cache_bytes() const;

    const std::shared_ptr<const PreparedModel> prepared;
    const PreparedModel& model;
    Prediction recorder;
    std::shared_ptr<const ResidencySet> cache_residency;
    bool initialized = false;

    struct Lane
    {
        StackCache temporal, codebook;
        Buffer conditioning;
        std::vector<float> probabilities;
    };

    std::vector<Lane> lanes;
    Header header;
    std::vector<Buffer> temporal_keys, temporal_values, codebook_keys, codebook_values;
    std::size_t temporal_extent = 0;
};
} // namespace tsac::metal::transformer
