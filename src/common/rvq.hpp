#pragma once
#include "common/dsp.hpp"
#include "format/container.hpp"
#include "models/rvq_decoder_model.hpp"
#include "models/rvq_encoder_model.hpp"
#include "weights/normalization.hpp"

namespace tsac
{
// A mathematical primitive for one DAC quantizer, also usable with synthetic
// weights. Inference obtains prepared tensors from the concrete RVQ adapters.
class RvqCodebook
{
public:
    static constexpr std::size_t vector_width = 8, entries = 1024, latent_width = 1024;
    // Shapes: codebook [8,1024], v [8,1,1024], g [1,1,1024], bias [1024].
    RvqCodebook(std::span<const float> codebook, std::span<const float> v, std::span<const float> g,
                std::span<const float> bias, const Limits& limits = {});
    // Captured original CPU normalization and fused eight-product projection.
    static RvqCodebook from_dac_weights(std::span<const float> codebook, std::span<const float> v,
                                        std::span<const float> g, std::span<const float> bias,
                                        const Limits& limits = {});
    static RvqCodebook from_prepared(const TensorView<float>& codebook,
                                     const FloatProjectionWeights& projection,
                                     const Limits& limits = {});
    // Writes exactly 1024 channels. Dot product in increasing input order,
    // followed by bias. Uses the original, unnormalized codebook vector.
    void project(std::uint16_t index, std::span<float> output) const;

private:
    RvqCodebook() = default;
    weights::Array<float> codebook_, weights_, bias_;
    bool fused_ = false;
};

// Residual quantization from DAC analysis latents. All input/output projection
// tensors and search vectors are schema-validated and owned after preparation.
template <class Kernels> class RvqEncoder
{
public:
    static RvqEncoder from_model(RvqEncoderModel model, const Limits& limits = {},
                                 Kernels kernels = {});
    // [1024,frames,groups] -> [time,group,codebook]. Encoding accepts either
    // header method; a normal header's CRC is assigned after token generation.
    std::vector<std::uint16_t> encode(const Header& header, std::span<const float> latent,
                                      const Limits& limits = {}) const;

private:
    struct Quantizer
    {
        typename Kernels::Conv1d input_projection;
        RvqCodebook output_projection;
        weights::Array<float> normalized_vectors;
        std::vector<std::uint16_t> apply(std::span<float> residual, const Limits& limits) const;
    };

    RvqEncoder() = default;
    unsigned model_channels_ = 0;
    std::vector<Quantizer> codebooks_;
};

class RvqDecoder
{
public:
    // Owns only the enabled quantizers; the adapter can be released after preparation.
    static RvqDecoder from_model(RvqDecoderModel model, const Limits& limits = {});
    // Input is [time,group,codebook] as returned by unpack_fast. Output has shape
    // [1024,frames,groups], axis 0 fastest. Normal headers require a valid CRC.
    // tensor_elements limits the complete output before it is allocated.
    std::vector<float> decode(const Header& header, std::span<const std::uint16_t> indices,
                              const Limits& limits = {}) const;

private:
    RvqDecoder() = default;
    unsigned model_channels_ = 0;
    std::vector<RvqCodebook> codebooks_;
};
} // namespace tsac
