#!/bin/sh
set -eu

# The official distribution bundles all four quantized models.
archive_name=tsac-2024-04-08
archive_url="https://bellard.org/tsac/${archive_name}.tar.gz"
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
model_dir=${1:-"$repo_dir/models"}
model_files='dac_mono_q8.bin dac_stereo_q8.bin tsac_mono_q8.bin tsac_stereo_q8.bin'

missing=0
for model in $model_files; do
    if [ ! -s "$model_dir/$model" ]; then
        missing=1
    fi
done
if [ "$missing" -eq 0 ]; then
    printf 'All four model files already exist in %s\n' "$model_dir"
    exit 0
fi

for tool in curl tar; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$tool" >&2
        exit 1
    fi
done

mkdir -p "$model_dir"
temp_dir=$(mktemp -d "$model_dir/.download-XXXXXX")
trap 'rm -rf "$temp_dir"' 0
trap 'exit 1' HUP INT TERM

printf 'Downloading models from %s\n' "$archive_url"
curl --fail --location --retry 3 --connect-timeout 15 \
    --output "$temp_dir/models.tar.gz" "$archive_url"
tar -xzf "$temp_dir/models.tar.gz" -C "$temp_dir" \
    "$archive_name/dac_mono_q8.bin" \
    "$archive_name/dac_stereo_q8.bin" \
    "$archive_name/tsac_mono_q8.bin" \
    "$archive_name/tsac_stereo_q8.bin"

# Validate the extraction before installing any files.
for model in $model_files; do
    if [ ! -s "$temp_dir/$archive_name/$model" ]; then
        printf 'Missing or empty model in download: %s\n' "$model" >&2
        exit 1
    fi
done
for model in $model_files; do
    if [ -s "$model_dir/$model" ]; then
        printf 'Keeping existing %s\n' "$model"
    else
        mv -f "$temp_dir/$archive_name/$model" "$model_dir/$model"
        printf 'Installed %s\n' "$model"
    fi
done
