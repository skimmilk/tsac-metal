# Compact packed weights

The compact profile stores 20 raw `.weights` files totaling **317,259,776 bytes**
(317.26 MB). The implementation is in `src/weights/`.

## Build and packaging

`./build.sh -DTSAC_ENABLE_METAL=OFF` builds the CPU Accelerate/NEON path and stages
`./tsac` with its weights in the project root. Set `TSAC_BUILD_DIR` or
`TSAC_BUILD_CONFIG` to select another build directory/configuration.

The four original inputs are `dac_mono_q8.bin`, `dac_stereo_q8.bin`,
`tsac_mono_q8.bin`, and `tsac_stereo_q8.bin`. Missing inputs are downloaded during
the build by `download-models.sh`, using `curl` and `tar`, from
[Bellard's TSAC distribution](https://bellard.org/tsac/). Supply an existing
directory with `-DTSAC_MODEL_DIR=/absolute/path/to/models`, or disable downloading
with `-DTSAC_DOWNLOAD_MODELS=OFF`. These `.bin` files are build inputs, not runtime
files. Their download cache is separate from the runtime storage total.

`tsac-pack-models` is a standalone C++ host executable with original-model
parsing and pure CPU preparation helpers. It does not link inference, Metal,
Objective-C++, or shaders. `cmake/Weights.cmake` builds it before inference.
`cmake/pack_weights.cmake` hashes the finished executable, the four original
`.bin` inputs, the compact layout version and Transformer inclusion option.
The first 16 SHA-256 hex characters identify a generation. The full canonical
identity is retained in build-only `identity.txt` to detect prefix collisions.

Each generation is written under
`build/weights/<configuration>/generations/<id>/`. Its C++ header and CMake
manifest are published only when all files are complete. The fixed bridge
header is `build/generated/<configuration>/adapter_offsets.hpp`. An incremental
check verifies the presence and sizes of every dynamic weight output, so a
surviving header cannot hide missing weights. No weights are hashed at startup.

The full profile emits ten roles for each of mono/stereo: compact DAC
analysis/synthesis sources, their
prepared shared constants, four prepared RVQ roles, canonical Transformer
source and Transformer shared data. Total weight bytes are 317,259,776.
`-DTSAC_ENABLE_TRANSFORMER=OFF` omits the four Transformer files, producing 16
files totaling 172,916,736 bytes. Normal entropy mode then reports that fast mode
(`-f`) is required. Both production backend layouts of the small RVQ input
projection are included even when Metal
compilation is gated off, keeping the default weight package backend-independent.

Each `.weights` file contains raw payloads and zero alignment padding only.
Tensor starts align to 256 bytes; mappable regions and file ends align to
16,384 bytes. Scalars use little-endian IEEE FP32, BF16 bits, or signed int8.
Compact DAC `weight_v` data retain their original quantization and internal
scale-plane layout; `weight_g` remains FP32. No new lossy quantization is used.

## File contents

Names follow `tsac-<mono|stereo>-<role>-<generation-id>.weights`. Each model has
the following roles:

- `dac-analysis-source`: original convolution `weight_v` payloads and FP32
  normalization gains for the encoder.
- `dac-analysis-shared`: prepared encoder biases and snake activation constants.
- `dac-synthesis-source`: original convolution `weight_v` payloads and FP32
  normalization gains for the decoder, including transposed convolutions.
- `dac-synthesis-shared`: prepared decoder biases and snake activation constants.
- `rvq-shared`: codebook vectors, folded output projection weights, and output
  biases used for quantizer reconstruction.
- `rvq-encode-shared`: normalized search vectors and input projection biases.
- `rvq-encode-apple`: folded input projection weights in the CPU layout.
- `rvq-encode-metal`: folded input projection weights in the Metal layout.
- `transformer-source`: one canonical row-major int8 plane for each projection.
- `transformer-shared`: FP32 projection scales and biases, BF16 normalization,
  positional and rotary constants, and precomputed feedback embeddings.

The two Transformer roles are absent from fast-only packages. Large convolution
execution arrays and backend-specific Transformer tiles are expanded in RAM
instead of duplicated on disk. The generated header contains private filenames, offsets, and byte ranges for
application entrypoints. Shapes and encodings live in the fixed C++ contract;
the payload files themselves contain no metadata.

## Model adapters

Inference uses the concrete adapters in the `tsac-model-adapters` CMake target.
The five concrete classes are `DacAnalysisModel`, `DacSynthesisModel`,
`RvqEncoderModel`, `RvqDecoderModel`, and `TransformerModel`, under
the private `src/models/` directory. They share infrastructure through composition and have
no model base class or generic model interface.

`ModelDirectory` is a public opaque, copyable handle with a shared weak mapping
cache. Private mapping access in `src/weights/model_directory.hpp` accepts only a
`ModelFile` enum, an offset, and a byte length.
It resolves hashed filenames internally and checks that the requested range
exists in the file. `MappedRegion` closes the file descriptor after mapping and
unmaps when its final shared owner disappears. Directory caches do not retain
unused mappings, and concurrent requests for the same range share an owner.

Each adapter owns its directory and the mapping handles for its purpose.
`TensorView<T>` returns an immutable span and contract-defined shape while
retaining the mapping independently. Keep the view alive while borrowing its
span; borrowing from temporary views is disabled. Encoded DAC payloads use
`EncodedTensorView` with the fixed contract's FP32/BF8 interpretation.

```cpp
ModelDirectory directory("/path/to/package");
RvqDecoderModel decoder(directory, ModelVariant::stereo, 4);
auto book = decoder.codebook(0);
std::span<const float> values = book.values();
// book retains its mapping even if decoder and directory are destroyed.
```

Shapes, axis order, encodings, and alignment live in the shared model contract,
not generated metadata. The packer validates its output against that contract.
The private `adapter_offsets.hpp` contains only filenames, offsets, and byte
ranges for consumed entrypoints, plus the package generation identity for
build-tree execution. The generic `Model`, runtime offset bindings, and old
`weight_offsets.hpp` generator have been removed.

The RVQ decoder requests only reconstruction data. The encoder additionally
selects the Accelerate/NEON or Metal input projection layout. RVQ mappings cover
only enabled books. Transformer feedback embeddings similarly map only the
enabled prefix. Adapters perform no inference, decompression, tile packing, or
GPU submission.

Build and test the adapter contract independently:

```sh
cmake --build build --target tsac-model-adapters-test
ctest --test-dir build -R '^model-adapters$' --output-on-failure
```

## Separate input and runtime paths

Only the original `packing::SourceModel` reader used by the packer/tests resolves
source tensor names. It is implemented in `src/weights/source_model.cpp` and built
into the separate `tsac-model-source` library. The packer validates the original
models and emits offsets for the named accessors of the concrete adapters. It
also writes a build-only CMake manifest with filenames and sizes for staging.
Inference does not link the source reader or parse that manifest.

The codec shares one `ModelDirectory` and constructs only the adapters needed by
the current phase: analysis plus RVQ encoder for compression, Transformer for
normal entropy coding, and RVQ decoder plus synthesis for decompression.
`CpuTransformer::from_model` and `MetalTransformer::from_model` take only a
`TransformerModel`; they do not need a DAC model. The channel variant and active
book count come from that adapter.

Preparation reads typed operand groups through explicit accessors such as
`residual_convolution(stage, unit, ResidualConvolution::dilated)` and
`attention_qkv(TransformerStack::temporal, layer)`. There is no generic tensor
lookup, model inheritance, graph traversal, or generated topology. Fixed pipeline
code supplies convolution stride/dilation and execution order. The Metal
`DacGraph` is an execution command recorder, not a stored model or weight lookup.

Residual units and stages carry no string identifiers. Optional diagnostic
observers receive labels constructed from numeric stage/unit positions only
when an observer is attached; ordinary inference creates no such labels.

## Mapping ownership

Constructing a `ModelDirectory` maps nothing. Constructing a concrete adapter maps
only its role's required regions after range, alignment, and policy-limit checks.
File descriptors are closed after mapping. File sizes only need to cover those
ranges; no expected-size metadata is required. Mappings are private and read-only.
Tensor views and Metal imports share RAII owners; the final owner unmaps the
region. The directory's weak cache cannot extend a mapping's lifetime.

Pipeline factories accept their concrete adapter by value. The codec passes a
temporary, so the factory's adapter is destroyed when preparation returns. An
external caller that keeps an adapter copy or tensor view deliberately keeps its
mapping alive. No global release or forced-unmap method can invalidate it.

DAC preparation expands one convolution, folds its captured normalization,
and creates the selected backend's execution array. Ordinary CPU convolutions
move the folded vector into final storage. Expanded temporaries are destroyed
before final phase packing/upload. When the consumed analysis or synthesis
adapter is destroyed after preparation, its quantized source mapping is released.
Mapped biases/alpha views retain only the small shared region.

CPU Transformer construction builds R4 tiles from the canonical row-major
mapping and then releases the source mapping. Scales, biases and BF16 constants
remain mapped. Feedback embeddings map one contiguous prefix containing exactly
the requested number of books, with no unused book mappings.

Metal construction imports canonical row-major regions without copying and
builds R2/R8 tiles in owned GPU buffers. Imports of the same mapping share a
buffer wrapper; residency follows those allocations. The Metal deallocator
retains the mapping independently, so destroying adapters does not unmap
live GPU weights. Immutable imports reject mutable CPU access. Metal 4 execution
and these import/protection details require subsequent GPU validation.

Encoding finishes and releases analysis/RVQ state before entropy-model loading.
Decoding destroys entropy-model state before preparing RVQ reconstruction and
synthesis. Fast mode never opens Transformer weights. RVQ data map only enabled
books; encode-only search/input projections are not requested during decoding.

## Staging, cleanup and installation

`tsac-stage` copies current assets beside `./tsac`, checks actual weight plus
executable sizes against `TSAC_WEIGHT_BUDGET` (400,000,000 bytes by default),
and publishes the executable after its weights. Old-minus-new hashed filenames
are removed after success.
A root-owned file list survives deletion of the build tree. An interrupted
staging transaction retains both file lists so the next stage can reconcile
them. Individual file renames are atomic; the whole set is not one transaction.

`build.sh` runs the CMake cleanup script in an exit trap and preserves failures.
Unused generations and abandoned temporary directories are removed, while the
active build generation and currently staged files are retained on failure.
Explicit `clean` removes staged assets and native build outputs. Ninja/Make
native clean also removes private generated directories; other generators use
the explicit asset-clean target. Cache downloads are preserved separately.
SIGKILL/power loss cannot execute an exit trap; the next invocation reconciles
temporary generations and staging transactions.

Run `./build.sh clean` for explicit cleanup, or `cmake --build build --target
tsac-clean-assets` to remove staged assets and generated weight packages only.

Install records owned filenames and prunes previous weight generations after
installing the executable. Explicit uninstall removes the recorded installed
files, including weights from an interrupted upgrade. These operations never
glob unrelated weight files. Already mapped unlinked files retain their backing
until final owners close them; starting old executable generations during an
upgrade is not a supported asset-retention policy.

`./build.sh uninstall` removes the last installation recorded by that build.
Installed assets load beside the actual executable, including when invoked
through a symlink. Development executables inside their build tree use that
build's immutable generated package.

## CPU verification

`tsac-weights-test` compares compact source bytes and dequantized tensors with
the originals, all 63 NEON projections for each model, prepared norm constants,
RVQ projection output, feedback embeddings, and Accelerate analysis/synthesis.
It checks exact bytes rather than tolerance. Weak-owner probes verify that DAC
and CPU Transformer source mappings expire after preparation while execution
objects remain alive. Two-book embeddings are checked to map only 4 MiB.

Use `-DTSAC_ENABLE_METAL=OFF` to compile and test this path without Objective-C++
or shader compilation. Metal stubs report unavailable and reject accidental GPU
calls. CPU parity does not establish Metal correctness.
