# Portable scalar reference

`tsac::ScalarAnalyzer` is a reference and research implementation. Its numerical
kernels use plain C++23, without Accelerate, NEON intrinsics, Metal, or an external
BLAS library. It supports both fast DAC/RVQ coding and normal Transformer entropy
coding through the same public `AudioAnalyzer` contract as the native analyzers.
It is intentionally slow and is useful for studying algorithms and comparing
numerical behavior, rather than production audio processing.

## Build

Use CMake 3.20 or later and a C++23 compiler **and standard library**. The scalar
kernels are portable; model loading requires POSIX `mmap` and `madvise`, with
fixed 16 KiB Apple Silicon page alignment. Prepared model assets currently use
little-endian IEEE-754 representations.

```sh
cmake -S . -B build-scalar -DCMAKE_BUILD_TYPE=Release \
  -DTSAC_ENABLE_APPLE_ARM=OFF \
  -DTSAC_ENABLE_METAL=OFF \
  -DTSAC_BUILD_CLI=OFF
cmake --build build-scalar --parallel
ctest --test-dir build-scalar --output-on-failure
```

The three switches explicitly select a portable library build even on an Apple
Silicon host. On other hosts they default to `OFF`. The existing command-line
application is Apple-specific; consume the scalar implementation through C++.

The build prepares the same model assets used by the native implementations.
Automatic downloads use `download-models.sh` and require a POSIX shell and its
usual download tools. To build without that dependency, supply the four original
model files described in [WEIGHTS.md](WEIGHTS.md) and additionally configure:

```sh
-DTSAC_DOWNLOAD_MODELS=OFF -DTSAC_MODEL_DIR=/path/to/original/models
```

`-DTSAC_ENABLE_TRANSFORMER=OFF` omits normal entropy assets; use
`CodingMethod::fast` in that configuration. See [WEIGHTS.md](WEIGHTS.md) for
asset generation, packaging, and storage details.

## Consume

Add this project with CMake's `add_subdirectory` and link the public target:

```cmake
add_subdirectory(path/to/tsac-metal tsac-build)
target_link_libraries(research PRIVATE tsac::scalar)
```

`tsac::scalar` aliases the existing `tsac_cpu` library target. Only `include/` is
public; no private source headers are needed.

```cpp
#include <tsac/scalar_analyzer.hpp>

// During development this may be the absolute build-scalar directory; the
// generated adapter resolves its active weight generation. For distribution,
// pass the directory containing the matching prepared .weights files.
tsac::ModelDirectory directory("/absolute/path/to/build-scalar");
tsac::ScalarAnalyzer analyzer(directory);

tsac::EncodeOptions options;
options.method = tsac::CodingMethod::fast;
options.codebooks = 1;
auto stream = analyzer.encode(tsac::interleaved_pcm(samples, 1), options);
auto reconstructed = analyzer.decode(stream);
```

Input samples are normalized float32 PCM at 44.1 kHz. The analyzer owns its output
and retains padding to a multiple of 512 samples. See [AUDIO_API.md](AUDIO_API.md)
for limits, strided PCM views, observers, and reuse guarantees.

Different floating-point reduction orders can change RVQ decisions and decoded
samples. Do not assume that scalar and accelerated encoders produce identical
streams, or that all compiler/host combinations reproduce identical research
results. Avoid fast-math options when comparing numerical behavior.
