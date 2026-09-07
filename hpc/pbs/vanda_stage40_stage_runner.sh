#!/usr/bin/env bash
# Shared Vanda A40 Stage40 runner for the authorized V3-V6 ladder.
# The small PBS wrappers select the stage; this file owns common
# provenance, runtime, worker, and result handling.

set -euo pipefail

STAGE="${VANDA_STAGE40_STAGE_RUN:?VANDA_STAGE40_STAGE_RUN is required}"
REPO="${VANDA_STAGE40_REPO-}"
if [ -z "$REPO" ]; then
  REPO=/home/svu/leonliu/projects/go2-arx5-hierarchical-vla
fi
PACKAGE_ROOT="${VANDA_STAGE40_PACKAGE_ROOT-}"
if [ -z "$PACKAGE_ROOT" ]; then
  PACKAGE_ROOT=/scratch/leonliu/leon-storage/stage40_frozen_inputs_20260907_v1/packages
fi
ARTIFACT_ROOT="${VANDA_STAGE40_ARTIFACT_ROOT-}"
if [ -z "$ARTIFACT_ROOT" ]; then
  ARTIFACT_ROOT=/scratch/leonliu/leon-storage/stage40_frozen_inputs_20260907_v1/materialized
fi
BUNDLE="${VANDA_STAGE40_BUNDLE-}"
if [ -z "$BUNDLE" ]; then
  BUNDLE="$REPO/manifests/VANDA_STAGE40_A40_EXECUTION_BUNDLE_20260908_V2.json"
fi
V2_LATEST="${VANDA_STAGE40_V2_LATEST-}"
if [ -z "$V2_LATEST" ]; then
  V2_LATEST=/scratch/leonliu/leon-storage/runs/vanda_stage40_v2_latest.json
fi
OUTPUT_BASE="${VANDA_STAGE40_OUTPUT_BASE-}"
if [ -z "$OUTPUT_BASE" ]; then
  OUTPUT_BASE=/scratch/leonliu/leon-storage/runs
fi
APPTAINER="${VANDA_STAGE40_APPTAINER_BIN-}"
if [ -z "$APPTAINER" ]; then
  APPTAINER=/cm/shared/apps/apptainer/1.4.1/bin/apptainer
fi
SIF="${VANDA_STAGE40_ISAAC_SIF-}"
if [ -z "$SIF" ]; then
  SIF="$PACKAGE_ROOT/ubuntu-desktop_24.04-cu12.6.2-isaac.sif"
fi

case "$STAGE" in
  v3_q0)
    RUN_LABEL=vanda_stage40_v3_q0
    NATIVE_MODE=c0
    RESULT_FILE=stage40_c0_result.json
    ;;
  v4_c0_dev)
    RUN_LABEL=vanda_stage40_v4_c0_dev
    NATIVE_MODE=v4
    RESULT_FILE=stage40_v4_result.json
    ;;
  v5_one_handoff)
    RUN_LABEL=vanda_stage40_v5_one_handoff
    NATIVE_MODE=live-seed
    RESULT_FILE=stage40_live_result.json
    ;;
  v6_3h)
    RUN_LABEL=vanda_stage40_v6_3h
    NATIVE_MODE=v6
    RESULT_FILE=stage40_v6_result.json
    ;;
  *)
    echo "UNKNOWN_VANDA_STAGE40_STAGE=$STAGE" >&2
    exit 64
    ;;
esac

RUN_ROOT="$OUTPUT_BASE/${RUN_LABEL}_$PBS_JOBID"
if [ -e "$RUN_ROOT" ]; then
  echo "OUTPUT_ALREADY_EXISTS=$RUN_ROOT" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT/runtime/home" "$RUN_ROOT/runtime/tmp" "$RUN_ROOT/runtime/cache" \
  "$RUN_ROOT/mailbox/requests" "$RUN_ROOT/mailbox/responses"
exec > >(tee -a "$RUN_ROOT/${RUN_LABEL}.log") 2>&1

cd "$REPO"
test -d "$REPO"
test -d "$ARTIFACT_ROOT/wbc/WBCTrackingShare"
test -f "$BUNDLE"
test -f "$V2_LATEST"
test -x "$APPTAINER"
test -f "$SIF"
test -d "$PACKAGE_ROOT"
test -z "$(git status --porcelain --untracked-files=all)"

EXPECTED_EXECUTABLE_HEAD="$(python3 - "$BUNDLE" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["source"]["executable_source_head"])
PY
)"
test -n "$EXPECTED_EXECUTABLE_HEAD"
git merge-base --is-ancestor "$EXPECTED_EXECUTABLE_HEAD" HEAD
git diff --quiet "$EXPECTED_EXECUTABLE_HEAD" -- hpc/pbs scripts src

python3 - "$V2_LATEST" "$EXPECTED_EXECUTABLE_HEAD" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

latest = Path(sys.argv[1])
expected_head = sys.argv[2]
payload = latest.read_bytes()
digest = hashlib.sha256(payload).hexdigest()
sidecar = Path(str(latest) + ".sha256").read_text(encoding="utf-8").split()[0]
if digest != sidecar:
    raise SystemExit("V2_LATEST_SHA256_MISMATCH")
result = json.loads(payload)
if result.get("status") != "PASS":
    raise SystemExit("V2_PREDECESSOR_NOT_PASS")
if int(result.get("forward_count", -1)) != 0 or int(result.get("wbc_forward_count", -1)) != 0:
    raise SystemExit("V2_PREDECESSOR_HAS_EXECUTION")
if result.get("repository_head") != expected_head:
    raise SystemExit("V2_PREDECESSOR_SOURCE_HEAD_MISMATCH")
PY

AUTH_ROOT="$ARTIFACT_ROOT/wbc"
RUNTIME_ROOT="$ARTIFACT_ROOT/stage22_runtime"
SITE="$RUNTIME_ROOT/runtime/venv/lib/python3.12/site-packages"
GROOT_SRC="$RUNTIME_ROOT/src/isaac-gr00t"
STRUCTURED_ROOT="$ARTIFACT_ROOT/structured_checkpoint_1000/checkpoint-1000"
BASE_MODEL="$ARTIFACT_ROOT/groot_base_model/GR00T-N1.7-3B"
TINY_ROOT="$ARTIFACT_ROOT/tiny1_h2stats_lerobot_v2/tiny1_h2stats_lerobot_v2"
COSMOS_ROOT="$ARTIFACT_ROOT/cosmos_reason2_2b_hf_cache/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561"
MODALITY="$ARTIFACT_ROOT/modality/go2_arx5_modality.py"
STRUCTURED_STATS="$PACKAGE_ROOT/structured_action_statistics.json"
ARM="$AUTH_ROOT/WBCTrackingShare/runs/2026-08-05/stage2_v3_trajtracking_rl_e2e_gait_wo_postproc_192018/checkpoints_arm/ac_weights_039999.pt"
DOG="$AUTH_ROOT/WBCTrackingShare/runs/2026-08-05/stage2_v3_trajtracking_rl_e2e_gait_wo_postproc_192018/checkpoints_dog/ac_weights_039999.pt"
for path in "$RUNTIME_ROOT" "$SITE" "$GROOT_SRC" "$STRUCTURED_ROOT" "$BASE_MODEL" \
  "$TINY_ROOT/meta" "$COSMOS_ROOT" "$MODALITY" "$STRUCTURED_STATS" "$ARM" "$DOG"; do
  test -e "$path"
done

export HOME="$RUN_ROOT/runtime/home"
export TMPDIR="$RUN_ROOT/runtime/tmp"
export TMP="$RUN_ROOT/runtime/tmp"
export TEMP="$RUN_ROOT/runtime/tmp"
export XDG_CACHE_HOME="$RUN_ROOT/runtime/cache"
export XDG_CONFIG_HOME="$RUN_ROOT/runtime/cache/config"
export XDG_DATA_HOME="$RUN_ROOT/runtime/cache/data"
export HF_HOME="$RUN_ROOT/runtime/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0

echo "=== VANDA STAGE40 $STAGE ==="
date
hostname
whoami
echo "PBS_JOBID=$PBS_JOBID"
echo "PBS_QUEUE=$PBS_QUEUE"
echo "PBS_O_QUEUE=$PBS_O_QUEUE"
echo "REPO_HEAD=$(git rev-parse HEAD)"
echo "EXECUTABLE_SOURCE_HEAD=$EXPECTED_EXECUTABLE_HEAD"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version,compute_cap --format=csv,noheader
python3 "$REPO/scripts/vanda_stage40_telemetry.py" --phase "${STAGE}_baseline" \
  --output-dir "$RUN_ROOT" --job-id "$PBS_JOBID" --gpu-index 0 \
  --runtime-root "$RUNTIME_ROOT" --cache-root "$RUN_ROOT/runtime/cache"

"$APPTAINER" exec --cleanenv --bind "$RUN_ROOT:/run" "$SIF" env \
  HOME=/run/runtime/home TMPDIR=/run/runtime/tmp TMP=/run/runtime/tmp TEMP=/run/runtime/tmp \
  PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/envs/isaacgym/bin/python - /run/trajectory.npz <<'PY'
import sys
import numpy as np

path = sys.argv[1]
timestamp = np.linspace(0.0, 5.0, 101, dtype=np.float64)
position = np.stack([
    np.full_like(timestamp, 0.30),
    np.full_like(timestamp, 0.10),
    np.full_like(timestamp, 0.58),
], axis=-1)
quaternion = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (len(timestamp), 1))
np.savez(path, timestamp=timestamp, position=position, quaternion=quaternion)
PY

NATIVE_LD="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/opt/miniconda3/envs/isaacgym/lib:/.singularity.d/libs:/lib/x86_64-linux-gnu"
GROOT_LD="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cusparse/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/curand/lib:$SITE/nvidia/cuda_nvrtc/lib"

"$APPTAINER" exec --nv --cleanenv \
  --bind "$REPO:/repo:ro" --bind "$ARTIFACT_ROOT:/artifacts:ro" \
  --bind "$PACKAGE_ROOT:/packages:ro" --bind "$RUN_ROOT:/run" "$SIF" env \
  HOME=/run/runtime/home TMPDIR=/run/runtime/tmp TMP=/run/runtime/tmp TEMP=/run/runtime/tmp \
  HF_HOME=/run/runtime/cache/huggingface HF_HUB_CACHE=/run/runtime/cache/huggingface/hub \
  TRANSFORMERS_CACHE=/run/runtime/cache/huggingface/hub HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$GROOT_LD" \
  PYTHONPATH="/repo:/repo/src:/repo/scripts:/artifacts/stage22_runtime/src/isaac-gr00t" \
  /usr/bin/python3.12 -B /repo/scripts/stage21_same_scene_groot_worker.py \
  --stage10-root /artifacts/stage22_runtime \
  --base-model-path /artifacts/groot_base_model/GR00T-N1.7-3B \
  --model-path /artifacts/structured_checkpoint_1000/checkpoint-1000 \
  --dataset-dir /artifacts/tiny1_h2stats_lerobot_v2/tiny1_h2stats_lerobot_v2 \
  --modality-config /artifacts/modality/go2_arx5_modality.py \
  --cosmos-snapshot /artifacts/cosmos_reason2_2b_hf_cache/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561 \
  --mailbox /run/mailbox --device cuda:0 --local-files-only \
  --action-contract STRUCTURED_ACTION_V1 >"$RUN_ROOT/groot_worker.log" 2>&1 &
WORKER_PID=$!
READY=0
cleanup() {
  touch "$RUN_ROOT/mailbox/STOP" 2>/dev/null || true
  if [ "${READY}" != 1 ] && kill -0 "$WORKER_PID" 2>/dev/null; then
    kill "$WORKER_PID" 2>/dev/null || true
  fi
  wait "$WORKER_PID" 2>/dev/null || true
}
trap cleanup EXIT
for _ in $(seq 1 "${VANDA_STAGE40_WORKER_READY_TIMEOUT_S-1200}"); do
  if grep -q '"status": "READY"' "$RUN_ROOT/groot_worker.log" 2>/dev/null; then
    READY=1
    break
  fi
  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done
test "$READY" = 1
python3 "$REPO/scripts/vanda_stage40_telemetry.py" --phase "${STAGE}_runtime_ready" \
  --output-dir "$RUN_ROOT" --job-id "$PBS_JOBID" --gpu-index 0 \
  --runtime-root "$RUNTIME_ROOT" --cache-root "$RUN_ROOT/runtime/cache"

ARM_REL="${ARM#"$AUTH_ROOT/"}"
DOG_REL="${DOG#"$AUTH_ROOT/"}"
NATIVE_ARGS=(
  /repo/scripts/stage40_live_one_handoff.py
  --mode "$NATIVE_MODE"
  --source-root /authority/WBCTrackingShare
  --trajectory /run/trajectory.npz
  --arm "/authority/$ARM_REL"
  --dog "/authority/$DOG_REL"
  --structured-stats /packages/structured_action_statistics.json
  --mailbox /run/mailbox
  --output-dir /run
  --sim-device cuda:0
  --seed 4242
  --response-timeout-s "${VANDA_STAGE40_RESPONSE_TIMEOUT_S-300}"
)
if [ "$NATIVE_MODE" = v6 ]; then
  NATIVE_ARGS=(
    /repo/scripts/stage40_receding_horizon_3h.py
    --source-root /repo
    --trajectory /run/trajectory.npz
    --arm "/authority/$ARM_REL"
    --dog "/authority/$DOG_REL"
    --mailbox /run/mailbox
    --output-dir /run
    --structured-stats /packages/structured_action_statistics.json
    --execution-bundle /repo/manifests/VANDA_STAGE40_A40_EXECUTION_BUNDLE_20260908_V2.json
    --sim-device cuda:0
    --seed 4242
    --response-timeout-s "${VANDA_STAGE40_RESPONSE_TIMEOUT_S-300}"
  )
fi
set +e
"$APPTAINER" exec --nv --cleanenv \
  --bind "$REPO:/repo:ro" --bind "$AUTH_ROOT:/authority:ro" --bind "$PACKAGE_ROOT:/packages:ro" \
  --bind "$RUN_ROOT:/run" "$SIF" env \
  HOME=/run/runtime/home TMPDIR=/run/runtime/tmp TMP=/run/runtime/tmp TEMP=/run/runtime/tmp \
  XDG_CACHE_HOME=/run/runtime/cache XDG_CONFIG_HOME=/run/runtime/cache/config \
  XDG_DATA_HOME=/run/runtime/cache/data PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  CUDA_VISIBLE_DEVICES=0 PATH="/usr/local/nvidia/bin:$PATH" LD_LIBRARY_PATH="$NATIVE_LD" \
  PYTHONPATH="/repo:/repo/src:/repo/scripts:/authority/WBCTrackingShare" \
  /opt/miniconda3/envs/isaacgym/bin/python -B "${NATIVE_ARGS[@]}"
NATIVE_RC=$?
set -e
echo "NATIVE_STAGE_RC=$NATIVE_RC"
test -f "$RUN_ROOT/$RESULT_FILE"
python3 - "$RUN_ROOT/$RESULT_FILE" "$STAGE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
stage = sys.argv[2]
data = json.loads(path.read_text(encoding="utf-8"))
if stage == "v3_q0":
    ok = data.get("status") == "PASS" and int(data.get("gr00t_forwards", -1)) == 1 and int(data.get("wbc_forwards", -1)) == 0
elif stage == "v4_c0_dev":
    ok = data.get("status") == "PASS" and int(data.get("gr00t_forwards", -1)) == 1 and int(data.get("native_wbc_forward_count", -1)) == 0
elif stage == "v5_one_handoff":
    ok = data.get("status") == "PASS"
else:
    ok = data.get("status") == "PASS" and int(data.get("forward_count", -1)) == 4 and int(data.get("reference_update_count", -1)) == 3
if not ok:
    raise SystemExit(f"{stage.upper()}_RESULT_NOT_PASS")
PY
if [ "$STAGE" = v5_one_handoff ]; then
  python3 "$REPO/scripts/vanda_stage40_v5_result.py" \
    --live-result "$RUN_ROOT/stage40_live_result.json" \
    --bundle "$BUNDLE" --repo "$REPO" --telemetry-root "$RUN_ROOT" \
    --job-id "$PBS_JOBID" --output "$RUN_ROOT/stage40_v5_result.json"
  python3 "$REPO/scripts/vanda_stage40_telemetry.py" \
    --seal-result "$RUN_ROOT/stage40_v5_result.json" \
    --seal-output "$RUN_ROOT/stage40_v5_result.seal.json"
fi
if [ "$STAGE" = v6_3h ]; then
  python3 "$REPO/scripts/vanda_stage40_telemetry.py" \
    --seal-result "$RUN_ROOT/stage40_v6_result.json" \
    --seal-output "$RUN_ROOT/stage40_v6_result.seal.json"
fi
python3 "$REPO/scripts/vanda_stage40_telemetry.py" --phase "${STAGE}_complete" \
  --output-dir "$RUN_ROOT" --job-id "$PBS_JOBID" --gpu-index 0 \
  --runtime-root "$RUNTIME_ROOT" --cache-root "$RUN_ROOT/runtime/cache"
if [ "$NATIVE_RC" -ne 0 ]; then
  exit "$NATIVE_RC"
fi
echo "VANDA_STAGE40_${STAGE^^}_PASS"
