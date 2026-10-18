# Metal 4 host/shader ABI

`types.h` defines fixed-width scalar types and error bits. `bindings.h` defines
named resource slots. The probability and DAC `parameters.h` headers define the
same records for C++ and MSL; there are no separate host mirrors.
`kernel_contracts.hpp` catalogs each kernel's parameter size and required resource
slots for validation before encoding GPU work.

This internal interface ships with the embedded shader libraries. It is
independent of the TSAC container format and does not change stored streams.

## Wire records

All fields are 32-bit integers or FP32 values, naturally aligned to four bytes.
There are no pointers, `size_t`, native `bool`, vector types, inheritance, packed
structs, or implicit padding. Boolean options use integer zero or one. Shared
size/alignment assertions compile on both host and GPU; host assertions also
check every member offset, standard layout, and trivial copyability.

Names include units: `payload_bytes` is a byte count, while
`key_column_stride_elements` counts BF16 elements. Host size arithmetic and
limits are checked before narrowing to a wire field.

Probability records cover addition, copying, embeddings, resident embeddings,
quantization, projection, prepared projection, normalization, cache appending,
attention, softmax, and range decoding. DAC records cover convolution, Snake,
residual arithmetic, RVQ search/projection/accumulation, and finite checking.

The former overloaded fields have explicit replacements:

- `apply_gelu`, `normalize_input`, and `add_residual` describe independent
  preparation options.
- `round_block_values` preserves attention's BF16 block boundary. The host sets
  it for replay and for histories longer than 64 keys. Blockwise kernels always
  retain that boundary.
- `write_position` is a zero-based cache index; `key_count` includes the newly
  appended key. Key and value column strides are separate fields.
- Nonempty dispatches use their actual column count; zero no longer means one.
- `initial_feedback` selects codebook symbol zero in every prefix book. The step
  field remains an ordinary step index instead of encoding an initial sentinel.
- `RangeState` names the GPU-written range, code, payload offset, and status.
  It is mutable device state, not a parameter block.

Fixed model geometry and mathematical constants remain governed by the codec's
numerical contract. The ABI change preserves computation and launch rules.

## Resource interface

Slot zero always holds the operation's constant parameter record. Remaining slots
have operation-specific names shared by host code and shader attributes. Host
resource lists contain explicit `(slot, Buffer)` bindings; list order is
irrelevant. Every required binding must appear once. Parameter slots, duplicates,
out-of-range slots, missing resources, incorrect parameter sizes, and mismatched
prepared pipelines are rejected on the host.

Keep the corresponding kernel contract current when changing a signature. The
registry contains all 64 exported entry points, including specialized variants.
Prepared projection currently binds valid resources in each declared slot even
when a preparation option is disabled.

## Submission ownership

`ExecutionContext` owns reusable Metal 4 command storage, an argument table,
parameter arena, and a transient residency set. Prediction and DAC graphs retain
this context across submissions, including standalone operator calls. `Program`
only loads libraries and resolves pipelines. Allocators are reset and CPU-written
memory is reused only after commit feedback reports completion, including
submissions that report GPU errors.

Immutable model resources and session caches have separate persistent residency
sets. Transient residency changes only when backing allocations change. Every
distinct backing buffer is retained and made resident, including constants,
scratch, error flags, caches, and allocations reached through buffer views. GPU
addresses alone do not establish ownership.

Each active dispatch receives a separate immutable parameter allocation. The
arena currently aligns allocations to 256 bytes as a conservative runtime policy;
this does not change the records' four-byte ABI alignment. Argument-table
snapshots capture bindings at dispatch time, not referenced parameter bytes.
The CPU must not overwrite an in-flight arena. Shared upload/readback staging
uses a separate cursor from GPU-only scratch, so recycling scratch cannot change
CPU-written values referenced by earlier recorded commands.

Dispatch barriers conservatively order dependencies and scratch reuse. A
queue-stage consumer barrier orders prior compute work before a new submission.
Completion feedback supplies GPU errors and duration metrics before CPU reads or
resource release. The GPU backend has no legacy command API fallback.

## Validation

`metal-abi` checks different parameter records within one submission, named
bindings supplied out of order, buffer view ownership and offsets, producer/
consumer ordering, invalid bindings and parameter blocks, repeated submissions,
and recovery after rejected work. Existing `metal-range` and `metal-optimized`
tests cover numerical and stream compatibility through this ABI, including
persistent contexts, private Transformer scratch, direct shared probability
outputs, shared DAC/RVQ handoffs, and staging reuse across completed chunks.
Typed mapped views are checked for backing identity, CPU/GPU visibility,
misalignment, incomplete elements, and rejection of private storage.
