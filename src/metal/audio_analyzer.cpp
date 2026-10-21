#include "tsac/metal_analyzer.hpp"

#include "audio_dac.hpp"
#include "audio_rvq.hpp"
#include "common/audio_pipeline.hpp"
#include "metal/transformer.hpp"

#include <exception>
#include <mutex>
#include <optional>

namespace tsac
{
struct MetalAnalyzer::Cache
{
    struct Session
    {
        ModelVariant variant;
        unsigned books;
        bool encoding, normal;
        std::optional<metal::AudioDac> dac;
        std::optional<metal::AudioRvqEncoder> encoder;
        std::optional<metal::AudioRvqDecoder> decoder;
        std::optional<MetalTransformer> transformer;

        Session(const Header& h, bool encode)
            : variant(detail::audio_variant(h)), books(h.codebooks), encoding(encode),
              normal(h.method == 1)
        {
        }

        bool matches(const Header& h, bool encode) const
        {
            return variant == detail::audio_variant(h) && books == h.codebooks &&
                   encoding == encode && normal == (h.method == 1);
        }

        void release_transformer_state()
        {
            if (transformer)
            {
                // Keep immutable prepared weights, releasing KV caches, scratch,
                // recorded commands and stream state before the next phase.
                transformer->clear_session();
            }
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
            std::lock_guard lock(cache.mutex);
            if (!cache.idle)
            {
                cache.idle = std::move(session);
            }
        }
    };

    Lease take(const Header& header, bool encoding)
    {
        std::unique_ptr<Session> session;
        {
            std::lock_guard lock(mutex);
            session = std::move(idle);
        }
        if (session && !session->matches(header, encoding))
        {
            session.reset();
        }
        if (!session)
        {
            session = std::make_unique<Session>(header, encoding);
        }
        return {*this, std::move(session)};
    }

    std::mutex mutex;
    std::unique_ptr<Session> idle;
};

MetalAnalyzer::MetalAnalyzer(ModelDirectory directory, const Limits& limits)
    : cache_(std::make_shared<Cache>()), directory_(std::move(directory)), limits_(limits)
{
    detail::require(metal::available(),
                    "Metal 4 requires macOS 26 or later and a supported Apple GPU");
}

Bytes MetalAnalyzer::encode(PcmView audio, const EncodeOptions& options,
                            const EntropyObserver& observer)
{
    const auto header = detail::audio_header(audio, options, limits_);
    auto lease = cache_->take(header, true);
    auto& session = *lease.session;
    std::vector<std::uint16_t> tokens;
    {
        const auto pcm = detail::padded_audio(audio, header, limits_);
        const auto variant = detail::audio_variant(header);
        if (!session.dac)
        {
            session.dac =
                metal::AudioDac::from_model(DacAnalysisModel(directory_, variant), limits_);
            session.encoder = metal::AudioRvqEncoder::from_model(
                RvqEncoderModel(directory_, variant, RvqInputLayout::output_major,
                                header.codebooks),
                limits_);
        }
        tokens.resize(header.symbol_count());
        const auto chunk_size = detail::audio_chunk_frames(200, limits_);
        metal::DacGraph graph;
        for (std::size_t group = 0; group < header.groups; ++group)
        {
            for (std::size_t begin = 0; begin < header.frames;)
            {
                const auto chunk = detail::audio_chunk(begin, header.frames, chunk_size, 8, 9);
                const SignalShape shape{header.model_channels, (chunk.right - chunk.left) * 512, 1};
                const auto offset = (group * pcm.shape.frames + chunk.left * 512) * shape.channels;
                graph.begin();
                const auto input = graph.upload(
                    std::span(pcm.values).subspan(offset, shape.elements(limits_)), shape, limits_);
                const auto latent = session.dac->record(graph, input, limits_);
                auto central_header = chunk.header(header);
                central_header.frames = chunk.count;
                const metal::DacTensor central{
                    {1024, chunk.count, 1},
                    latent.buffer.slice((begin - chunk.left) * 1024 * sizeof(float),
                                        chunk.count * 1024 * sizeof(float))};
                const auto selected =
                    session.encoder->record(graph, central_header, central, limits_);
                graph.finish();
                for (std::size_t book = 0; book < selected.size(); ++book)
                {
                    const auto values = selected[book].mapped<std::uint16_t>();
                    for (std::size_t frame = 0; frame < chunk.count; ++frame)
                    {
                        tokens[((begin + frame) * header.groups + group) * header.codebooks +
                               book] = values[frame];
                    }
                }
                begin += chunk.count;
            }
        }
    }
    if (header.method == 0)
    {
        return pack_fast(header, tokens, limits_);
    }
    if (!session.transformer)
    {
        session.transformer = MetalTransformer::from_model(
            TransformerModel(directory_, detail::audio_variant(header), header.codebooks), limits_);
    }
    auto result = session.transformer->encode(header, tokens, detail::completed_symbols(observer));
    session.release_transformer_state();
    return result;
}

PcmAudio MetalAnalyzer::decode(ByteView stream, const EntropyObserver& observer)
{
    const auto header = parse_header(stream, limits_);
    detail::require_probability_assets(header);
    auto lease = cache_->take(header, false);
    auto& session = *lease.session;
    std::vector<std::uint16_t> tokens;
    if (header.method == 0)
    {
        tokens = unpack_fast(stream, limits_);
    }
    else
    {
        if (!session.transformer)
        {
            session.transformer = MetalTransformer::from_model(
                TransformerModel(directory_, detail::audio_variant(header), header.codebooks),
                limits_);
        }
        tokens = session.transformer->decode(stream, detail::completed_symbols(observer));
        session.release_transformer_state();
    }
    const auto variant = detail::audio_variant(header);
    if (!session.dac)
    {
        session.decoder = metal::AudioRvqDecoder::from_model(
            RvqDecoderModel(directory_, variant, header.codebooks), limits_);
        session.dac = metal::AudioDac::from_model(DacSynthesisModel(directory_, variant), limits_);
    }
    auto audio = detail::allocate_pcm(header, limits_);
    const auto chunk_size = detail::audio_chunk_frames(400, limits_);
    metal::DacGraph graph;
    for (std::size_t group = 0; group < header.groups; ++group)
    {
        for (std::size_t begin = 0; begin < header.frames;)
        {
            const auto chunk = detail::audio_chunk(begin, header.frames, chunk_size, 11, 11);
            const auto selected = detail::chunk_tokens(tokens, header, group, chunk);
            graph.begin();
            const auto latent =
                session.decoder->record(graph, chunk.header(header), selected, limits_);
            const auto evaluated = session.dac->record(graph, latent, limits_);
            graph.finish();
            const auto values = evaluated.buffer.mapped<float>();
            for (std::size_t sample = 0; sample < chunk.count * 512; ++sample)
            {
                for (std::size_t channel = 0; channel < header.model_channels; ++channel)
                {
                    audio.samples[(begin * 512 + sample) * audio.channels +
                                  group * header.model_channels + channel] =
                        values[((begin - chunk.left) * 512 + sample) * header.model_channels +
                               channel];
                }
            }
            graph.release(evaluated);
            begin += chunk.count;
        }
    }
    return audio;
}
} // namespace tsac
