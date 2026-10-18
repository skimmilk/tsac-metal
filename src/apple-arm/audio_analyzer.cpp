#include "tsac/apple_arm_analyzer.hpp"

#include "apple-arm/pipeline.hpp"
#include "common/audio_pipeline.hpp"
#include "common/rvq.hpp"

#include <exception>
#include <mutex>
#include <optional>

namespace tsac
{
// Lease mutable sessions exclusively. A single idle session bounds retained
// prepared weights; concurrent calls get independent sessions and workers.
struct AppleArmAnalyzer::Cache
{
    struct Session
    {
        ModelVariant variant;
        unsigned books;
        bool encoding, normal;
        const unsigned threads;
        apple_arm::Execution execution;
        std::optional<apple_arm::DacAnalysis> analysis;
        std::optional<apple_arm::RvqEncoder> encoder;
        std::optional<apple_arm::DacSynthesis> synthesis;
        std::optional<RvqDecoder> decoder;
        std::optional<apple_arm::CpuTransformer> transformer;

        Session(const Header& header, bool encode, unsigned budget)
            : variant(detail::audio_variant(header)), books(header.codebooks), encoding(encode),
              normal(header.method == 1), threads(budget), execution(threads)
        {
        }

        bool matches(const Header& header, bool encode) const
        {
            return variant == detail::audio_variant(header) && books == header.codebooks &&
                   encoding == encode && normal == (header.method == 1);
        }
    };

    struct Lease
    {
        Cache& cache;
        std::unique_ptr<Session> session;
        int exceptions = std::uncaught_exceptions();

        ~Lease()
        {
            if (std::uncaught_exceptions() != exceptions)
            {
                return;
            }
            if (session->transformer)
            {
                session->transformer->clear_session();
            }
            std::lock_guard lock(cache.mutex);
            if (!cache.idle)
            {
                cache.idle = std::move(session);
            }
        }
    };

    Lease take(const Header& header, bool encode, unsigned threads)
    {
        std::unique_ptr<Session> session;
        {
            std::lock_guard lock(mutex);
            session = std::move(idle);
        }
        if (session && !session->matches(header, encode))
        {
            session.reset();
        }
        if (!session)
        {
            session = std::make_unique<Session>(header, encode, threads);
        }
        return {*this, std::move(session)};
    }

    std::mutex mutex;
    std::unique_ptr<Session> idle;
};

AppleArmAnalyzer::AppleArmAnalyzer(ModelDirectory directory, const Limits& limits, unsigned threads)
    : cache_(std::make_shared<Cache>()), directory_(std::move(directory)), limits_(limits),
      threads_(threads)
{
    detail::require(threads <= 256, "CPU thread count exceeds supported limit (256)");
}

Bytes AppleArmAnalyzer::encode(PcmView audio, const EncodeOptions& options,
                               const EntropyObserver& observer)
{
    const auto header = detail::audio_header(audio, options, limits_);
    auto lease = cache_->take(header, true, threads_);
    auto& session = *lease.session;
    auto& execution = session.execution;
    std::vector<std::uint16_t> tokens;
    {
        const auto pcm = detail::padded_audio(audio, header, limits_);
        const auto variant = detail::audio_variant(header);
        if (!session.analysis)
        {
            session.analysis = apple_arm::DacAnalysis::from_model(
                DacAnalysisModel(directory_, variant), limits_, {&execution});
            session.encoder = apple_arm::RvqEncoder::from_model(
                RvqEncoderModel(directory_, variant, RvqInputLayout::input_major, header.codebooks),
                limits_, {&execution});
        }
        tokens = detail::analyze_audio(pcm, header, *session.analysis, *session.encoder,
                                       detail::audio_chunk_frames(400, limits_), limits_);
    }
    if (header.method == 0)
    {
        return pack_fast(header, tokens, limits_);
    }
    if (!session.transformer)
    {
        session.transformer = apple_arm::CpuTransformer::from_model(
            TransformerModel(directory_, detail::audio_variant(header), header.codebooks), limits_);
    }
    TransformerPredictor predictor(*session.transformer, limits_, &execution);
    return pack_normal(header, tokens, predictor, limits_, detail::completed_symbols(observer));
}

PcmAudio AppleArmAnalyzer::decode(ByteView stream, const EntropyObserver& observer)
{
    const auto header = parse_header(stream, limits_);
    detail::require_probability_assets(header);
    auto lease = cache_->take(header, false, threads_);
    auto& session = *lease.session;
    auto& execution = session.execution;
    std::vector<std::uint16_t> tokens;
    if (header.method == 0)
    {
        tokens = unpack_fast(stream, limits_);
    }
    else
    {
        if (!session.transformer)
        {
            session.transformer = apple_arm::CpuTransformer::from_model(
                TransformerModel(directory_, detail::audio_variant(header), header.codebooks),
                limits_);
        }
        TransformerPredictor predictor(*session.transformer, limits_, &execution);
        tokens = unpack_normal(stream, predictor, limits_, detail::completed_symbols(observer));
        // KV storage is transient; release it before allocating DAC scratch.
        session.transformer->clear_session();
    }
    const auto variant = detail::audio_variant(header);
    if (!session.synthesis)
    {
        session.decoder =
            RvqDecoder::from_model(RvqDecoderModel(directory_, variant, header.codebooks), limits_);
        session.synthesis = apple_arm::DacSynthesis::from_model(
            DacSynthesisModel(directory_, variant), limits_, {&execution});
    }
    return detail::synthesize_audio(tokens, header, *session.decoder, *session.synthesis,
                                    detail::audio_chunk_frames(400, limits_), limits_);
}
} // namespace tsac
