#pragma once
#include "common/transformer.hpp"
#include "models/transformer_model.hpp"

#include <memory>

namespace tsac
{
// Shared CPU Transformer pipeline with explicit numerical kernels. Prepared weights are owned; each
// instance owns independent mutable caches. Implements the captured BF16/BF8L arithmetic, including
// the distinct 256-token rollover-prefill contract.
template <class Kernels> class CpuTransformer final : public TransformerBackend
{
public:
    static CpuTransformer from_model(TransformerModel model, const Limits& limits = {});
    ~CpuTransformer();
    CpuTransformer(CpuTransformer&&) noexcept;
    CpuTransformer& operator=(CpuTransformer&&) noexcept;
    bool independent_lanes() const override;
    void clear_session();
    void reset(const Header& header, std::size_t independent_lanes) override;
    void reset_temporal(std::size_t lane) override;
    void temporal_step(std::size_t lane, std::size_t position,
                       std::span<const std::uint16_t> feedback, bool replay) override;
    void reset_codebook(std::size_t lane) override;
    std::span<const float> codebook_step(std::size_t lane, std::size_t book,
                                         std::span<const std::uint16_t> prefix) override;

private:
    struct Impl;
    explicit CpuTransformer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
} // namespace tsac
