#!/usr/bin/env bash
set -euo pipefail

: "${PI0246_T_RUN_CARD:?set the sealed Corelab child-card path}"
: "${PI0246_T_RUNTIME_ROOT:?set the bound runtime root}"
: "${PI0246_T_RUNTIME_WRAPPER:?set this wrapper path}"
: "${PI0246_T_RUNTIME_BINDING:?set the runtime binding path}"
: "${PI0246_T_RUNTIME_BINDING_SHA256:?set the sealed runtime-binding SHA256}"
: "${PI0246_T_TRAINER:?set the staged trainer path}"
: "${PI0246_T_PYTHON:?set the exact Stage10 Python path}"

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ffmpeg_root=$(jq -er '.paths.ffmpeg6_root' "$PI0246_T_RUN_CARD")
ffmpeg_manifest=$(jq -er '.paths.ffmpeg6_manifest' "$PI0246_T_RUN_CARD")
ffmpeg_manifest_sha=$(jq -er '.paths.ffmpeg6_manifest_sha256' "$PI0246_T_RUN_CARD")
ffmpeg_record=$(jq -er '.paths.ffmpeg6_bootstrap_record' "$PI0246_T_RUN_CARD")
ffmpeg_record_sha=$(jq -er '.paths.ffmpeg6_bootstrap_record_sha256' "$PI0246_T_RUN_CARD")
capsule_sha=$(jq -er '.paths.ffmpeg6_capsule_sha256' "$PI0246_T_RUN_CARD")
torchcodec_root=$(jq -er '.paths.torchcodec_library_root' "$PI0246_T_RUN_CARD")
torch_library_root=$(jq -er '.paths.torch_library_root' "$PI0246_T_RUN_CARD")

test -d "$ffmpeg_root/lib" && test -d "$ffmpeg_root/compat"
test "$(sha256sum "$ffmpeg_manifest" | cut -d ' ' -f 1)" = "$ffmpeg_manifest_sha"
test "$(sha256sum "$ffmpeg_record" | cut -d ' ' -f 1)" = "$ffmpeg_record_sha"
test "$(jq -er '.status' "$ffmpeg_record")" = PASS
test "$(jq -er '.source_runtime_id' "$ffmpeg_record")" = 9243a7880b06ac3984ddf4a1a667b685441ec0f34d6d529fa0f96c3c894e10cf
test "$(jq -er '.source_manifest_sha256' "$ffmpeg_record")" = "$ffmpeg_manifest_sha"
test "$(jq -er '.capsule_sha256' "$ffmpeg_record")" = "$capsule_sha"
test -d "$torchcodec_root" && test -d "$torch_library_root"

export LD_LIBRARY_PATH="$ffmpeg_root/lib:$ffmpeg_root/compat:$torchcodec_root:$torch_library_root:${LD_LIBRARY_PATH:-}"

mode=${1:---train}
if [[ $# -gt 1 ]]; then
  echo "usage: $0 [--functional-preflight|--train]" >&2
  exit 2
fi
case "$mode" in
  --functional-preflight|--train) ;;
  *) echo "unsupported mode: $mode" >&2; exit 2 ;;
esac

exec "$PI0246_T_PYTHON" "$PI0246_T_TRAINER" "$mode" --run-card "$PI0246_T_RUN_CARD"
