#pragma once
#include "metal/convolution.hpp"
#include "metal/runtime.hpp"

namespace tsac::metal
{
struct RvqBuffers
{
    Buffer codebook, weights, bias, normalized;
};

struct DacTensor
{
    SignalShape shape;
    Buffer buffer;
};

// One bounded DAC chunk. Explicit release marks the last recorded consumer;
// later operations can reuse its storage because dispatches are ordered.
class DacGraph
{
public:
    DacGraph();
    ~DacGraph();
    // Reuse the completed chunk's command memory, staging, and scratch buffers.
    void begin();
    // Distinct CPU-visible ranges for the whole chunk, reused after completion.
    Buffer shared_temporary(std::size_t bytes);
    DacTensor upload(std::span<const float> input, SignalShape shape, const Limits& limits);
    DacTensor zeros(SignalShape shape, const Limits& limits);
    DacTensor convolution(const MetalConvolution& conv, const DacTensor& input,
                          const Limits& limits);
    DacTensor snake(const DacTensor& input, const Buffer& alpha, const Limits& limits);
    DacTensor residual(const DacTensor& left, const DacTensor& right, bool subtract,
                       const Limits& limits);
    Buffer search(const DacTensor& input, const RvqBuffers& weights, const Limits& limits,
                  bool normalize = true);
    DacTensor lookup(const Buffer& tokens, const RvqBuffers& weights, const Limits& limits);
    DacTensor project(const Buffer& tokens, const RvqBuffers& weights, const DacTensor& residual,
                      bool subtract, const Limits& limits);
    void finish();
    void release(const DacTensor& tensor);
    Signal download(const DacTensor& tensor, const Limits& limits);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace tsac::metal
