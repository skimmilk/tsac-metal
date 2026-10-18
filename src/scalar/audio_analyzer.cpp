#include "tsac/scalar_analyzer.hpp"

#include "common/audio_pipeline.hpp"
#include "common/rvq.hpp"
#include "scalar/pipeline.hpp"

namespace tsac
{
ScalarAnalyzer::ScalarAnalyzer(ModelDirectory directory, const Limits& limits)
    : directory_(std::move(directory)), limits_(limits)
{
}

Bytes ScalarAnalyzer::encode(PcmView audio, const EncodeOptions& options,
                             const EntropyObserver& observer)
{
    const auto header = detail::audio_header(audio, options, limits_);
    std::vector<std::uint16_t> tokens;
    {
        const auto pcm = detail::padded_audio(audio, header, limits_);
        const auto variant = detail::audio_variant(header);
        const auto analysis =
            scalar::DacAnalysis::from_model(DacAnalysisModel(directory_, variant), limits_);
        const auto quantizer = scalar::RvqEncoder::from_model(
            RvqEncoderModel(directory_, variant, RvqInputLayout::input_major, header.codebooks),
            limits_);
        tokens = detail::analyze_audio(pcm, header, analysis, quantizer,
                                       detail::audio_chunk_frames(400, limits_), limits_);
    }
    if (header.method == 0)
    {
        return pack_fast(header, tokens, limits_);
    }
    auto transformer = scalar::CpuTransformer::from_model(
        TransformerModel(directory_, detail::audio_variant(header), header.codebooks), limits_);
    TransformerPredictor predictor(transformer, limits_);
    return pack_normal(header, tokens, predictor, limits_, detail::completed_symbols(observer));
}

PcmAudio ScalarAnalyzer::decode(ByteView stream, const EntropyObserver& observer)
{
    const auto header = parse_header(stream, limits_);
    detail::require_probability_assets(header);
    std::vector<std::uint16_t> tokens;
    if (header.method == 0)
    {
        tokens = unpack_fast(stream, limits_);
    }
    else
    {
        auto transformer = scalar::CpuTransformer::from_model(
            TransformerModel(directory_, detail::audio_variant(header), header.codebooks), limits_);
        TransformerPredictor predictor(transformer, limits_);
        tokens = unpack_normal(stream, predictor, limits_, detail::completed_symbols(observer));
    }
    const auto variant = detail::audio_variant(header);
    const auto quantizer =
        RvqDecoder::from_model(RvqDecoderModel(directory_, variant, header.codebooks), limits_);
    const auto synthesis =
        scalar::DacSynthesis::from_model(DacSynthesisModel(directory_, variant), limits_);
    return detail::synthesize_audio(tokens, header, quantizer, synthesis,
                                    detail::audio_chunk_frames(400, limits_), limits_);
}
} // namespace tsac
