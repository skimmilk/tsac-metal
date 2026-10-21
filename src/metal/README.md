# Metal shader library

Each `.metal` file is an independent translation unit. Headers contain shared
types, inline arithmetic, and templates; source files contain kernel entry points
and helpers used only by that operation. Compile the sources in each directory
and link their intermediate objects into a probability or DAC `.metallib`.
Do not include `.metal` files from other source files or compile headers as
standalone sources.

This source organization works with the Metal 4 execution API. The shading
language version, command API, and source file boundaries are separate choices.
The CMake build compiles and embeds both libraries; the runtime loads those bytes
instead of compiling shader source at startup. GPU-specific pipeline creation
still happens on the selected device.

The runtime uses Metal 4 command queues, allocators, argument tables, residency
sets, compute barriers, compiler pipelines, and commit feedback. It requires
macOS 26 or later and a Metal 4-capable GPU. CPU inference keeps its macOS 15
minimum. The [shared parameter ABI](abi/README.md) defines the host/shader contract.

## Unified memory and execution

All inference records through `ExecutionContext`, which retains its own Metal 4
queue and two submission slots. Each slot owns its command allocator, command
buffer, argument table, parameter arena, transient residency, retained pipelines,
and completion feedback. `Program` owns the library and resolves pipelines; it
has no separate submission path. `submit()` commits without waiting and returns
a ticket; `wait()` retires tickets in queue order, and `drain()` completes all
outstanding work. Reusing a busy slot waits for its completion first. Existing
`run()` callers remain synchronous. Session replacement and destruction also
drain the queue before releasing resources.
This follows Apple's [Metal 4 resource and command model](https://developer.apple.com/videos/play/wwdc2025/205/).
The runtime explicitly requires a GPU with unified memory.

`MetalAnalyzer` records analysis and RVQ encoding in one DAC graph per bounded
chunk. The central latent interval is a buffer slice, including when analysis
has context frames on either side. Decoding records RVQ accumulation and DAC
synthesis in the same graph. Latents never pass through CPU vectors between
these stages. Only token results and the retained PCM interval are read by the
CPU after completion. The graph reuses command storage, resolved pipelines,
shared staging, and scratch across chunks and channel groups. The DAC and RVQ
recorders consume their input tensors: they release scratch after its last
recorded consumer, so callers must not schedule another consumer afterward.

CPU inputs and GPU results use shared Metal allocations. `Buffer::mapped<T>()`
returns a read-only span into those allocations without allocating or copying;
it validates storage mode, alignment, and element size. Keep an owning `Buffer`
alive while borrowing a span. Immutable mapped model files are imported with
`newBufferWithBytesNoCopy`, and views retain their mapping ownership. Arbitrary
caller spans still require a copy at the public boundary: their allocation,
alignment, lifetime, and exclusive access are not owned by the backend.

CPU-populated staging stays separate from recycled GPU scratch. Each recorded
CPU input gets a distinct range until the chunk completes, even if the graph
has released its last GPU consumer. Explicit dispatch barriers order GPU writes
and scratch reuse. `Prediction` alternates two CPU staging/error-word slots;
`begin()` waits before reusing a busy slot, while `finish()` remains synchronous.
Borrowed scratch results must be consumed before the next chunk overwrites them.

Resident probability decoding keeps at most two ten-step chunks submitted on the
same queue. While the GPU executes one chunk, the CPU records and commits the
next. The queue-level dispatch barrier preserves shared token/range state, KV
caches, probabilities, and GPU-only scratch dependencies across submissions.
Only CPU-written staging, scalar parameters, and error words need separate
storage. Progress callbacks remain ordered and run after their chunk completes.
Context replay drains pending chunks and remains synchronous; final token reads,
CRC checks, exceptions, and teardown also drain outstanding consumers. This
pipeline overlaps host preparation with GPU execution; it does not reduce the
number of dispatches or change model arithmetic.

With `TSAC_METAL_PROFILE`, `wait` measures time blocked on completion feedback.
Preparation of a later asynchronous submission can overlap earlier GPU time, so
`encode + wait` is not the full decode elapsed time; graph recording and other
host work are outside those counters. `gpu` sums command execution intervals.

The Transformer host range encoder writes softmax output directly to shared
buffers, then converts the mapped BF16 values after completion. Known-token
lookahead retains the shared buffer itself. GPU-only Transformer caches and
activations remain private; no probability staging copy is needed.

Standalone convolution, activation, residual, RVQ, and linear helpers record
through these same graph implementations. The former standalone quantization
shader and one-shot submission API have been removed.

## Building

Normal builds compile each source to `.air`, link `probability.metallib` and
`dac.metallib`, then generate private C++ headers that embed the libraries in
`tsac_cpu`. The executable and tests need no external shader files. All generated
artifacts live under `build/metal/<configuration>/`, with separate directories for
each configuration. Shader and header changes trigger incremental rebuilds.

Build just the libraries and embedded headers with:

```sh
cmake --build build --target tsac-metal-shaders -j 4
```

`tsac-metal-shader-check` remains an alias for the same work. Compiler diagnostics
refer directly to the `.metal` and header files. Xcode's Metal toolchain must be
installed; if its launcher reports a missing component, install it with
`xcodebuild -downloadComponent MetalToolchain`.

Shader options follow the CMake configuration:

- `Debug`: `-O0`, line tables, and embedded shader sources for GPU debugging.
- `Release`: `-O2`, without embedded debugging information.
- `RelWithDebInfo`: `-O2` with line tables and embedded sources.
- `MinSizeRel`: `-Os`, without embedded debugging information.

Every configuration uses MSL 4.0 with a macOS 26 shader deployment target,
`-fno-fast-math`, `-ffp-contract=off`, safe Metal math, and precise FP32 math
functions. Configuration changes affect optimization and debugging information,
not the intended numerical contract.

The active Xcode SDK supplies `metal` and `metallib` through `xcrun`. For a separate
toolchain, set the CMake cache paths `TSAC_METAL_COMPILER` and `TSAC_METAL_LINKER`.

## Probability and entropy

### Shader organization

`probability/` contains the deterministic Transformer and range decoder:

- `parameters.h`: operation-specific wire records shared by the host and shaders.
- `numeric.h`: BF16 conversions, explicit floating-point boundaries, ordered
  reduction, compatibility exponential, and GELU. These inline helpers live in
  `tsac_probability`. The header disables implicit FP contraction for every
  probability translation unit.
- `quantization.h`: shared threadgroup state for quantization and fused embedding
  preparation.
- `elementwise.metal`: BF16 residual addition and exact copying.
- `embedding.metal`: token embeddings and fused resident embedding quantization.
- `quantization.metal`: checked parallel activation quantization.
- `linear.metal`: scalar/SIMD and packed small-batch projections.
- `linear_prepared.metal`: fused normalization/GELU, quantization, and projection.
- `linear_matrix.metal`: tiled matrix projections with optional residuals.
- `attention.metal`: short, codebook, and blockwise attention.
- `cache.metal`: QKV splitting, rotary positions, and cache appending.
- `normalization.metal`: normalization and fused normalization/quantization.
- `softmax.metal`: categorical probabilities and range-decoder subtree masses.
- `range.metal`: serialized resident range decoding and token commitment.

The probability library contains 34 kernel entry points. Kernel names remain global;
bindings use the names in `abi/bindings.h`, with parameters at slot zero.
Shared header functions are inline to permit inclusion by independent
translation units without duplicate helper definitions.

## DAC

`dac/` contains audio convolution, activation, and residual vector quantization:

- `parameters.h`: named convolution, activation, residual, and RVQ wire records.
- `convolution.h`: shared coordinate mapping and convolution tile templates in
  `tsac_dac`.
- `convolution.metal`: generic scalar and four-channel convolution.
- `convolution_tiled.metal`: tiled and pointwise convolution entry points.
- `convolution_specialized.metal`: fixed channel/dilation variants and their
  shared interior/tail selection helper.
- `convolution_transposed.metal`: phase-aware transposed convolution variants.
- `snake.metal`: Snake activation and its sine approximation.
- `rvq.metal`: codebook search, projection, and accumulation.
- `elementwise.metal`: residual arithmetic and nonfinite detection.

The library retains all 29 kernel entry points. DAC arithmetic remains separate
from the probability compatibility primitives because their numerical contracts
differ.
