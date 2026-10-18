#!/bin/sh
set -eu
cd "$(dirname "$0")"
build_dir=${TSAC_BUILD_DIR:-build}
case "$build_dir" in
    /*) ;;
    *) build_dir="$PWD/$build_dir" ;;
esac
configuration=${TSAC_BUILD_CONFIG:-Release}
cleanup() {
    result=$?
    trap - EXIT
    cmake -DDESTINATION="$PWD" -DROOT="$build_dir/weights/$configuration" \
        -P cmake/clean_weights.cmake || true
    exit "$result"
}
trap cleanup EXIT
if [ "${1:-}" = clean ]; then
    cmake -DDESTINATION="$PWD" -DROOT="$build_dir/weights/$configuration" -DALL=ON \
        -P cmake/clean_weights.cmake
    if [ -f "$build_dir/CMakeCache.txt" ]; then
        cmake --build "$build_dir" --config "$configuration" --target clean
    fi
    exit 0
fi
if [ "${1:-}" = uninstall ]; then
    cmake --build "$build_dir" --config "$configuration" --target uninstall
    exit 0
fi
cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE="$configuration" "$@"
cmake --build "$build_dir" --config "$configuration" --parallel --target tsac-stage
