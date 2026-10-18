#pragma once
#include "format/container.hpp"
#include "metal/dac_graph.hpp"
#include "models/rvq_decoder_model.hpp"
#include "models/rvq_encoder_model.hpp"

namespace tsac::metal
{
class AudioRvqEncoder
{
public:
    static AudioRvqEncoder from_model(RvqEncoderModel model, const Limits& limits);
    std::vector<std::uint16_t> encode(const Header& header, std::span<const float> latent,
                                      const Limits& limits) const;
    std::vector<Buffer> record(DacGraph& graph, const Header& header, const DacTensor& latent,
                               const Limits& limits) const;

private:
    struct Quantizer
    {
        MetalConvolution input;
        RvqBuffers output;
    };

    unsigned channels_ = 0;
    std::vector<Quantizer> codebooks_;
};

class AudioRvqDecoder
{
public:
    static AudioRvqDecoder from_model(RvqDecoderModel model, const Limits& limits);
    std::vector<float> decode(const Header& header, std::span<const std::uint16_t> tokens,
                              const Limits& limits) const;
    DacTensor record(DacGraph& graph, const Header& header, std::span<const std::uint16_t> tokens,
                     const Limits& limits) const;

private:
    unsigned channels_ = 0;
    std::vector<RvqBuffers> codebooks_;
};
} // namespace tsac::metal
