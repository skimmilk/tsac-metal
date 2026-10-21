# TSAC on Apple Silicon

A native C++23 implementation of Fabrice Bellard's TSAC codec for Apple M1 or
later. CPU inference requires **macOS 15 or later**; the Metal 4 backend requires
**macOS 26 or later**. Both support DAC encoding and
decoding and deterministic normal-mode probability inference.

A [reference scalar option](SCALAR.md) is also available for research. It uses
plain C++ and should compile on any machine with a C++23 compiler and standard
library, without Apple frameworks or NEON extensions.

Build the CPU Accelerate/NEON path with CMake 3.20 or later and Xcode 15.3 or later
(Metal 4 builds need a newer SDK as described below):

```sh
./build.sh -DTSAC_ENABLE_METAL=OFF
ctest --test-dir build --output-on-failure
./tsac -T 1 -q 3 c input.wav encoded.tsac
./tsac d encoded.tsac decoded.wav
```

`build.sh` configures and builds under `build/`, then stages `./tsac` in the
project root. Set `TSAC_BUILD_DIR` or `TSAC_BUILD_CONFIG` to select another build
directory/configuration. See [WEIGHTS.md](WEIGHTS.md) for model downloads,
weight storage, packaging options, mapping lifetimes, and asset cleanup.

```sh
cmake --install build --prefix /your/prefix
cmake --build build --target uninstall
```

Installation includes the executable, runtime assets, and documentation with
the original notice.

CPU is the default; `--metal` enables GPU execution.
The Metal backend may be faster or slower depending on your hardware. On this M1,
Metal took 8–9% longer than CPU in the benchmark below.

Commands are `c` (compress), `d` (decompress), and `t` (compress then decompress).
Original CLI options retained:
`-h`/`--help`, `-T`, `-q`/`--n_codebooks`, `-s`/`--separate_channels`,
`-c`/`--channels`, `-f`/`--fast`, `--batch_size`, and `-v`.
Quality is 1–12 for joint stereo and 1–9 otherwise; 0 selects the maximum.
Model-path options are excluded. Run `./tsac --help` for
the option descriptions.

Benchmark measurements on an Apple M1
(8 physical CPU cores), macOS 27.0.1, Release build, using approximately
30 seconds of music at 44.1 kHz in joint stereo, maximum quality (12 codebooks),
entropy batch size 3, and 8 CPU workers:

- Encode: CPU **9.39 s**; Metal **10.21 s** (1.09× the CPU time).
- Decode: CPU **13.54 s**; Metal **14.60 s** (1.08× the CPU time).

Native input is 44.1 kHz PCM16 WAVE or signed little-endian raw `.sw`. Other
formats, sample-rate conversion, and explicit WAVE channel conversion use
optional FFmpeg. Ordinary commands are quiet; `-v` reports statistics.

Original TSAC distribution notice:

> Copyright (c) 2023-2024 Fabrice Bellard
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
> THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
> THE SOFTWARE.
