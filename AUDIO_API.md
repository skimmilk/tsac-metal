# Audio analyzer contract

The supported C++ surface consists of eight headers under `include/tsac`:

- `types.hpp`: bytes, errors, and resource limits.
- `pcm.hpp`: borrowed PCM views, owning results, and slicing helpers.
- `observer.hpp`: entropy progress positions and callbacks.
- `audio_analyzer.hpp`: recording options and the abstract analyzer contract.
- `apple_arm_analyzer.hpp`, `scalar_analyzer.hpp`, and `metal_analyzer.hpp`: concrete implementations.
- `model_directory.hpp`: the opaque, copyable model-directory handle.

Choose a concrete implementation when composing the application. Construction
receives a `ModelDirectory` handle and `Limits`; loading and execution inside each
implementation have no CPU/Metal switch. Tensor layouts, mapping leases, model
adapters, operators, schedulers, entropy coders, and file/CLI helpers are private
headers under `src`. Only `include/` is propagated to library consumers. White-box
tests opt into private headers explicitly; the public consumer test does not.
Every public header is also compiled in isolation.

Private sources are grouped under `apple-arm`, `scalar`, `metal`, `common`,
`models`, `weights`, and `format`. Shared pipeline templates, geometry, validation,
scheduling, and portable numerical helpers live under `src/common`. Scalar and
Apple ARM pipelines bind their own kernel types at compile time, with no CPU
backend mode flag. GPU recording/runtime code lives under `src/metal`. See the
[scalar build guide](SCALAR.md) for portable reference and research use.

## PCM views and ownership

`PcmView` is a C++23 `std::mdspan<const float, dextents<size_t, 2>, layout_stride>`.
Its logical indices are `[channel, frame]`; frames are individual PCM samples per
channel, rather than 512-sample codec frames. Samples are normalized float32 at
44.1 kHz. Both dimensions are dynamic, and strides are measured in float elements.

The caller keeps the view's backing allocation alive throughout the synchronous
call. Mapping extents and strides must describe valid accessible memory. Zero
sample frames are supported. Input channels must be 1–127; nonfinite samples are
rejected. Sample-rate conversion, file reading, and WAVE/PCM16 conversion remain
outside the analyzer.

`interleaved_pcm(samples, channels)` validates complete interleaved frames.
`channel_view(audio, channel)` selects one channel while retaining its frame
stride. `frame_view(audio, first, count)` selects a checked frame interval while
preserving both strides. These helpers allocate nothing. A channel of interleaved
PCM remains strided and cannot be represented by a contiguous `std::span<float>`.
They construct `mdspan` mappings directly; `std::submdspan` is not required.

Decoded `PcmAudio` owns interleaved `samples` and a channel count. Its lvalue-only
`view()` exposes a borrowed `PcmView` and prevents borrowing from a temporary.
Mutation or destruction of the owning sample buffer may invalidate borrowed views.

## Recording operations

```cpp
#include <tsac/apple_arm_analyzer.hpp>

tsac::AppleArmAnalyzer analyzer(directory, limits, 4); // Per-instance thread budget.
auto pcm = tsac::interleaved_pcm(samples, 2);
tsac::EncodeOptions options;
options.codebooks = 3;
auto stream = analyzer.encode(pcm, options);
auto reconstructed = analyzer.decode(stream);
auto right = tsac::channel_view(reconstructed.view(), 1);
```

`encode` returns a complete TSAC stream. It pads input with zeros to a multiple
of 512 samples per channel. `codebooks == 0` chooses the model's maximum (9 mono,
12 joint stereo). Two-channel input uses the stereo model unless
`separate_channels` is set; other channel counts use independent mono groups.

`CodingMethod::normal` uses the Transformer and arithmetic coding.
`CodingMethod::fast` packs RVQ tokens directly and does not load Transformer
assets. An absent `entropy_batch` selects automatic normal-stream batching;
a supplied value must be nonzero and is rejected for fast encoding. The stored
entropy batch is format state, independent of internal chunk or submission sizes.

`decode` derives all format choices from the stream, validates framing and
limits, verifies a normal stream's token CRC before synthesis, and returns
interleaved PCM in original channel order. Decoded padding is retained because
the existing TSAC format does not record the original input sample count.

Calls complete before returning, including GPU work. Returned buffers own their
storage. Calls on one analyzer instance must be exclusive; separate instances
support concurrent recordings. `AppleArmAnalyzer` alone accepts a thread budget
as its third constructor argument (maximum 256). The default, zero, runs outer
work serially and retains automatic Accelerate threading; a positive budget bounds
outer workers and uses single-threaded BLAS. Scalar and Metal analyzers have no
thread-count option. Each call has independent recording state, so an analyzer
can be reused after success, validation failure,
inference failure, or an observer exception. Explicitly requesting an unavailable
Apple ARM or Metal implementation fails at construction; no automatic execution fallback occurs.

`AppleArmAnalyzer` retains at most one idle prepared session for sequential reuse.
The cache matches the encoding/decoding role, mono/stereo model, exact codebook
coverage, and fast/normal method. A different key evicts the idle session; an
exception discards the failing session. The session owns its workers so prepared
convolutions never borrow call-local execution resources. Transformer KV storage
is released after entropy decoding, before DAC synthesis, and after encoding.
Prepared weights remain resident until eviction or analyzer destruction, so model
files must remain immutable for that lifetime. Copies share the single idle slot.

The optional `EntropyObserver` runs synchronously in stream entropy order after
the associated work completes, provides per-symbol bit statistics, and propagates
exceptions. Fast coding emits no entropy callbacks. No implementation prints
or performs audio file I/O. Internal layer diagnostics remain CPU operator APIs,
not part of the shared recording interface.

## Implementation boundaries and complexity

Shared helpers own format geometry, limits, padding, channel layout, chunk halos,
and token ordering. Analysis retains eight latent frames on the left and nine on
the right; synthesis retains eleven on each side. Each implementation prepares
and evaluates its own operators. Metal owns GPU DAC/RVQ resources and uses
`MetalTransformer` directly, including resident range decoding. Its DAC/RVQ
stages record into the same Metal 4 graph for each bounded chunk and share latent
buffer views. CPU access to shared outputs occurs after completion; no latent
CPU vector is created between analysis and RVQ or between RVQ and synthesis.

The CPU Transformer contains no GPU resources or GPU batching capabilities.
Its predictor handles CPU lane scheduling and context replay only. The scalar analyzer
provides reference kernels independently of Accelerate/NEON.

Recording methods remain short orchestration functions. CPU convolution separates
source indexing, scalar evaluation, lowering, tile execution and phase scheduling;
RVQ separates nearest-codebook search from residual updates. Packed projections
separate integer dot blocks from per-column BF16 accumulation. CPU attention
separates tiled/row-major score evaluation, value accumulation and streaming
softmax merging. Native probability kernels live under `src/apple-arm`, reference kernels under
`src/scalar`, and reusable arithmetic and pipeline structure under `src/common`. Preserve numerical
reduction and rounding order when changing these helpers. Prefer functions around
40–60 lines and cyclomatic complexity below ten; justified numerical kernels may
need more space.
