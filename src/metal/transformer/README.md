# MetalTransformer

`tsac::MetalTransformer` is the GPU-owned probability-model session. Its private
interface accepts models and exposes normal-stream `encode` and resident
`decode`. Each operation resets logical stream state while preserving reusable
execution resources. `new_session()` shares the immutable model and creates an
independent recording owner, queue, caches, and scratch storage.
`clear_session()` releases mutable execution state while retaining prepared
weights; the next encode/decode recreates that state. The analyzer uses it to
avoid retaining probability scratch and KV buffers alongside DAC working memory.

## Implementation boundaries

- `model.hpp` / `model.mm`: GPU-prepared projections, normalization parameters,
  feedback embeddings, learned positions, and rotary coefficients. Canonical
  int8 weights, scales, biases and BF16 constants use no-copy mapping imports.
  R2/R8 projection tiles are interleaved directly into final shared buffers;
  only partial tiles need padding writes. One immutable residency set retains
  every model backing allocation, including imported mapping owners.
- `session.hpp` / `session.mm`: the recording owner, lane caches, logical cache
  positions, uniform cache views, and the budget-aware unequal-lane layout.
  Matching stream geometries reuse caches. Cache residency changes only when
  their allocation geometry changes.
- `execution.mm`: stack evaluation and causal known-token batches. Softmax
  writes directly to shared output buffers for the host range encoder, which
  reads mapped BF16 values after completion without GPU or CPU staging copies.
- `encoding.mm`: private encoding scheduler, bounded known-token lookahead,
  committed history, and context replay. Container traversal and host range
  encoding use the shared entropy implementation.
- `decoding.mm`: resident tokens and serial range state, bounded command chunks,
  context replay, ordered progress, and final token CRC validation.
- `replay.mm`: encoding replay across independent lanes in bounded submissions,
  retaining the last conditioning output without feeding feedback twice.
- `../transformer.mm`: private C++ session interface and implementation ownership.

The implementation has no dependency on `CpuTransformer`, `PackedLinear`,
`TransformerBackend`, `TransformerPredictor`, or CPU lane scheduling. CPU
preprocessing of model data does not retain CPU inference layouts.

## Metal 4 contract

Execution uses the operation-specific records and named resource bindings in
`../abi/` and `../probability/parameters.h`. The shared prediction recorder
records those contracts through a persistent `ExecutionContext`. It owns a
queue, command allocator, command buffer, argument table, growing constant
arena, and transient residency set. Allocator memory and CPU-written staging
are reused only after completion feedback. Each dispatch receives its own
aligned, immutable parameter range. Resident embedding calls carry an explicit
step and initial-feedback flag.

Model and cache residency sets are attached to the session's queue and remain
separate from transient working resources. Only changed transient allocations
update that residency set. Sets retain actual backing allocations, including
allocations reached through buffer views.

CPU-populated weights, constants, parameters, uploads, error words, and host
probability outputs use shared storage. KV caches, conditioning, range trees, and intermediate
activations use private storage. Scratch recycling never rewinds the staging
cursor: earlier recorded commands retain distinct CPU uploads until completion.
Conservative dispatch barriers order scratch reuse and producer/consumer work.

The existing BF16 rounding, quantization, attention reductions, and shader
arithmetic are retained. Cache rollover replays the last 256 complete frames at
1024 positions, with replay-specific attention rounding. Encoding lookahead
uses ordinary causal rounding. Batch and submission sizes are execution policy.

## Integration status

`MetalAnalyzer` uses this class for normal-stream encoding and decoding. The CLI
selects `AppleArmAnalyzer` or `MetalAnalyzer` once per run through `AudioAnalyzer`.
The mixed `CpuTransformer` GPU implementation and generic GPU predictor scheduling
have been removed. Projection, stream parity, replay, bounded-cache and resident
decode tests exercise this implementation directly.
