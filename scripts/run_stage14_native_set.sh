#!/usr/bin/env bash
set -euo pipefail

# Native IsaacGym execution wrapper. All private inputs are arguments; this
# repository contains no lab source, checkpoint, or dataset.
SOURCE_ROOT=${SOURCE_ROOT:?set SOURCE_ROOT to the external delivered source}
ARM_CHECKPOINT=${ARM_CHECKPOINT:?set ARM_CHECKPOINT}
DOG_CHECKPOINT=${DOG_CHECKPOINT:?set DOG_CHECKPOINT}
E1_TRAJECTORY_ROOT=${E1_TRAJECTORY_ROOT:?set E1_TRAJECTORY_ROOT containing E1 trajectories}
E2_TRAJECTORY_ROOT=${E2_TRAJECTORY_ROOT:?set E2_TRAJECTORY_ROOT containing E2 trajectories}
OUTPUT_ROOT=${OUTPUT_ROOT:?set node-local output root}
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
NATIVE_PYTHON=${NATIVE_PYTHON:?set NATIVE_PYTHON to the compatible Python 3.8 runtime}
ISAACGYM_PYTHON=${ISAACGYM_PYTHON:?set ISAACGYM_PYTHON}
NATIVE_COMPAT=${NATIVE_COMPAT:?set NATIVE_COMPAT for the isolated compatibility shim}
RUNTIME_LIB=${RUNTIME_LIB:?set RUNTIME_LIB containing libpython/libtorch dependencies}
TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:?set TORCH_EXTENSIONS_DIR to a node-local compiled-extension cache}
SIM_DEVICE=${SIM_DEVICE:-cuda:0}
GPU_VISIBLE=${GPU_VISIBLE:-1}
SECONDS_PER_ROLLOUT=${SECONDS_PER_ROLLOUT:-2}
SEED=${SEED:-13}
REFERENCE_MODE=${REFERENCE_MODE:-BENCHMARK_CONTINUOUS}

mkdir -p "$OUTPUT_ROOT"
export PATH="$(dirname "$NATIVE_PYTHON"):$PATH"
export PYTHONPATH="$ISAACGYM_PYTHON:$NATIVE_COMPAT:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$RUNTIME_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6
export TORCH_EXTENSIONS_DIR
export CUDA_VISIBLE_DEVICES="$GPU_VISIBLE"
export WBC_ARM_CHECKPOINT_SHA256="$(sha256sum "$ARM_CHECKPOINT" | awk '{print $1}')"
export WBC_DOG_CHECKPOINT_SHA256="$(sha256sum "$DOG_CHECKPOINT" | awk '{print $1}')"
export WBC_RUNTIME_VERSION="IsaacGym native; Python 3.8.20; PyTorch 2.3.1; CUDA 12.1; Gym 0.23.1"

CONFIG_PATH="$REPO_ROOT/configs/stage14_native_baseline.yaml"
CONFIG_SHA256="$(sha256sum "$CONFIG_PATH" | awk '{print $1}')"

if [[ -n "${STAGE14_IDS:-}" ]]; then
  read -r -a ids <<<"$STAGE14_IDS"
else
  ids=(
    E1_W0_00
    E1_W0_01
    E1_ang_0_00
    E1_curvature_1_00
    E1_coupling_1_00
    E2_static_0_0
    E2_cont_0_0
  )
fi

for id in "${ids[@]}"; do
  case "$id" in
    E2_*) trajectory="$E2_TRAJECTORY_ROOT/$id.npz" ;;
    *) trajectory="$E1_TRAJECTORY_ROOT/$id.npz" ;;
  esac
  output="$OUTPUT_ROOT/${id}_seed${SEED}"
  test -f "$trajectory"
  mkdir -p "$output"
  set +e
  (
    cd "$SOURCE_ROOT"
    timeout --signal=TERM 180s "$NATIVE_PYTHON" \
      "$REPO_ROOT/scripts/native_wbc_benchmark.py" \
      --source-root "$SOURCE_ROOT" \
      --arm "$ARM_CHECKPOINT" \
      --dog "$DOG_CHECKPOINT" \
      --trajectory "$trajectory" \
      --reference-mode "$REFERENCE_MODE" \
      --output-dir "$output" \
      --seconds "$SECONDS_PER_ROLLOUT" \
      --seed "$SEED" \
      --sim-device "$SIM_DEVICE"
  ) >"$output/run.log" 2>&1
  rc=$?
  set -e
  python3 "$REPO_ROOT/scripts/run_manifest.py" \
    --repo "$REPO_ROOT" \
    --output "$output/run_manifest.json" \
    --trajectory-sha "$(sha256sum "$trajectory" | awk '{print $1}')" \
    --config-sha "$CONFIG_SHA256" \
    --reference-mode "$REFERENCE_MODE" \
    --seed "$SEED"
  if [[ "$rc" -ne 0 ]]; then
    exit "$rc"
  fi
done
