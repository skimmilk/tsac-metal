#!/bin/sh
set -eu

cd "$(dirname "$0")"

formatter=${CLANG_FORMAT:-clang-format}
if [ -z "${CLANG_FORMAT:-}" ] && ! command -v "$formatter" >/dev/null 2>&1; then
    for candidate in \
        /opt/homebrew/opt/llvm/bin/clang-format \
        /usr/local/opt/llvm/bin/clang-format \
        /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang-format \
        /Library/Developer/CommandLineTools/usr/bin/clang-format; do
        if [ -x "$candidate" ]; then
            formatter=$candidate
            break
        fi
    done
fi

if ! command -v "$formatter" >/dev/null 2>&1; then
    printf 'Error: %s was not found. Install clang-format or set CLANG_FORMAT.\n' "$formatter" >&2
    exit 1
fi

# Format project sources, including Metal shaders, using .clang-format.
find include src tests -type f \( \
    -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o \
    -name '*.h' -o -name '*.hh' -o -name '*.hpp' -o -name '*.hxx' -o -name '*.tpp' -o \
    -name '*.m' -o -name '*.mm' -o -name '*.metal' \
\) -exec "$formatter" --style=file -i -- {} +
