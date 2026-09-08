#!/usr/bin/env bash
# Direct Corelab RTX CT3 runtime qualification (not a PBS submission).
#
# This launcher is intentionally inert until a one-line Vanda cutover token is
# supplied. Once authorized, it starts exactly two Stage40-owned processes on
# one explicitly selected GPU: a native constructor and a GR00T constructor.
# Both processes stop at a barrier; neither is allowed to forward, step WBC,
# or update a reference during CT3.

set -euo pipefail

: "${STAGE40_CORELAB_REPO:?set the pushed Corelab checkout path}"
: "${STAGE40_CT3_OUTPUT_ROOT:?set a new mutable CT3 output root}"
: "${STAGE40_CT3_BUNDLE:?set the frozen RTX-runtime execution bundle}"
: "${STAGE40_CT3_WBC_SOURCE_ROOT:?set the current WBC source root}"
: "${STAGE40_CT3_STAGE10_ROOT:?set the Stage10 GR00T root}"
: "${STAGE40_CT3_BASE_MODEL_PATH:?set the immutable GR00T base model path}"
: "${STAGE40_CT3_MODEL_PATH:?set the immutable STRUCTURED@1000 model path}"
: "${STAGE40_CT3_DATASET_DIR:?set the immutable TINY1 dataset path}"
: "${STAGE40_CT3_MODALITY_CONFIG:?set the immutable modality module path}"
: "${STAGE40_CT3_COSMOS_SNAPSHOT:?set the exact Cosmos snapshot path}"
: "${STAGE40_CT3_NATIVE_PYTHON:?set the native Python 3.8 executable}"
: "${STAGE40_CT3_MODEL_PYTHON:?set the GR00T Python executable}"
: "${STAGE40_CT3_NATIVE_NINJA:?set the native ninja executable}"
: "${STAGE40_CT3_ISAACGYM_PYTHON:?set the IsaacGym Python path}"
: "${STAGE40_CT3_NATIVE_COMPAT_ROOT:?set the native compatibility root}"
: "${STAGE40_CT3_NATIVE_LD_LIBRARY_PATH:?set the native runtime library path}"
: "${STAGE40_CT3_GPU_UUID:?set the exact RTX GPU UUID}"
: "${STAGE40_CUTOVER_TOKEN_FILE:?set a file containing the exact Vanda cutover token}"

REPO="$(cd "${STAGE40_CORELAB_REPO}" && pwd -P)"
OUTPUT_ROOT="$(cd "${STAGE40_CT3_OUTPUT_ROOT}" && pwd -P)"
BUNDLE="$(cd "$(dirname "${STAGE40_CT3_BUNDLE}")" && pwd -P)/$(basename "${STAGE40_CT3_BUNDLE}")"
WBC_SOURCE_ROOT="$(cd "${STAGE40_CT3_WBC_SOURCE_ROOT}" && pwd -P)"
STAGE10_ROOT="$(cd "${STAGE40_CT3_STAGE10_ROOT}" && pwd -P)"
BASE_MODEL_PATH="$(cd "${STAGE40_CT3_BASE_MODEL_PATH}" && pwd -P)"
MODEL_PATH="$(cd "${STAGE40_CT3_MODEL_PATH}" && pwd -P)"
DATASET_DIR="$(cd "${STAGE40_CT3_DATASET_DIR}" && pwd -P)"
MODALITY_CONFIG="$(cd "$(dirname "${STAGE40_CT3_MODALITY_CONFIG}")" && pwd -P)/$(basename "${STAGE40_CT3_MODALITY_CONFIG}")"
COSMOS_SNAPSHOT="$(cd "${STAGE40_CT3_COSMOS_SNAPSHOT}" && pwd -P)"
NATIVE_PYTHON="$(cd "$(dirname "${STAGE40_CT3_NATIVE_PYTHON}")" && pwd -P)/$(basename "${STAGE40_CT3_NATIVE_PYTHON}")"
MODEL_PYTHON="$(cd "$(dirname "${STAGE40_CT3_MODEL_PYTHON}")" && pwd -P)/$(basename "${STAGE40_CT3_MODEL_PYTHON}")"
NATIVE_NINJA="$(cd "$(dirname "${STAGE40_CT3_NATIVE_NINJA}")" && pwd -P)/$(basename "${STAGE40_CT3_NATIVE_NINJA}")"
ISAACGYM_PYTHON="$(cd "${STAGE40_CT3_ISAACGYM_PYTHON}" && pwd -P)"
NATIVE_COMPAT_ROOT="$(cd "${STAGE40_CT3_NATIVE_COMPAT_ROOT}" && pwd -P)"
TOKEN_FILE="$(cd "$(dirname "${STAGE40_CUTOVER_TOKEN_FILE}")" && pwd -P)/$(basename "${STAGE40_CUTOVER_TOKEN_FILE}")"
GPU_UUID="${STAGE40_CT3_GPU_UUID}"

CUTOVER_TOKEN="VANDA_STAGE40_V2_QUEUE_RELEASED_FOR_CORELAB_TAKEOVER"
RELEASE_TOKEN="CORELAB_STAGE40_CT3_RELEASE"
DEV_SEED=4242

test -d "${OUTPUT_ROOT}"
test -f "${BUNDLE}"
test -d "${WBC_SOURCE_ROOT}"
test -d "${STAGE10_ROOT}"
test -d "${BASE_MODEL_PATH}"
test -d "${MODEL_PATH}"
test -d "${DATASET_DIR}"
test -f "${MODALITY_CONFIG}"
test -d "${COSMOS_SNAPSHOT}"
test -x "${NATIVE_PYTHON}"
test -x "${MODEL_PYTHON}"
test -x "${NATIVE_NINJA}"
test -d "${ISAACGYM_PYTHON}"
test -d "${NATIVE_COMPAT_ROOT}"
test -f "${TOKEN_FILE}"
test ! -L "${TOKEN_FILE}"

TOKEN_OBSERVED="$(tr -d '\r\n' < "${TOKEN_FILE}")"
test "${TOKEN_OBSERVED}" = "${CUTOVER_TOKEN}"

# Validate the frozen bundle before touching CUDA. This imports only the
# standard-library evidence module; it does not construct a model or simulator.
"${MODEL_PYTHON}" - "${REPO}" "${BUNDLE}" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1])
bundle_path = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("stage40_corelab_evidence", repo / "scripts" / "stage40_corelab_evidence.py")
if spec is None or spec.loader is None:
    raise SystemExit("cannot load Corelab evidence verifier")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
module.validate_bundle(bundle, expected_stage="RTX-runtime")
PY

RUN_ID="${STAGE40_CT3_RUN_ID:-ct3_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_ID}"
test ! -e "${RUN_ROOT}"
mkdir -p "${RUN_ROOT}/native" "${RUN_ROOT}/model" "${RUN_ROOT}/runtime"
exec > >(tee -a "${RUN_ROOT}/stage40_ct3_runtime.log") 2>&1

echo "STAGE40_PHASE=RTX-runtime"
echo "STAGE40_PI_GATE=PI_GATE_STATE_0065"
echo "STAGE40_BUNDLE=${BUNDLE}"
echo "STAGE40_GPU_UUID=${GPU_UUID}"
echo "STAGE40_CUTOVER_TOKEN=${CUTOVER_TOKEN}"
echo "STAGE40_DEV_SEED=${DEV_SEED}"

# nvidia-smi is intentionally the first GPU-facing operation. No Stage40
# process is started until the UUID and the preferred 32-GiB card are visible.
GPU_BASELINE="${RUN_ROOT}/runtime/gpu_baseline.csv"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv,noheader,nounits | tee "${GPU_BASELINE}"
GPU_ROW="$(awk -F',' -v wanted="${GPU_UUID}" '
  { uuid=$2; gsub(/^[[:space:]]+|[[:space:]]+$/, "", uuid); if (uuid == wanted) { print; exit } }
' "${GPU_BASELINE}")"
test -n "${GPU_ROW}"
GPU_TOTAL_MIB="$(printf '%s\n' "${GPU_ROW}" | awk -F',' '{v=$4; gsub(/^[[:space:]]+|[[:space:]]+$/, "", v); print v}')"
test "${GPU_TOTAL_MIB}" -ge 30000

export CUDA_VISIBLE_DEVICES="${GPU_UUID}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HOME="${RUN_ROOT}/runtime/home" TMPDIR="${RUN_ROOT}/runtime/tmp" TMP="${RUN_ROOT}/runtime/tmp" TEMP="${RUN_ROOT}/runtime/tmp"
export XDG_CACHE_HOME="${RUN_ROOT}/runtime/cache" XDG_CONFIG_HOME="${RUN_ROOT}/runtime/config" XDG_DATA_HOME="${RUN_ROOT}/runtime/data"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="${RUN_ROOT}/runtime/hf" HF_HUB_CACHE="${RUN_ROOT}/runtime/hf/hub" TRANSFORMERS_CACHE="${RUN_ROOT}/runtime/hf/hub"
mkdir -p "${HOME}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_DATA_HOME}" "${HF_HUB_CACHE}"

ABORT_MARKER="${RUN_ROOT}/ABORT"
RELEASE_MARKER="${RUN_ROOT}/RELEASE"
NATIVE_READY="${RUN_ROOT}/native.ready.json"
MODEL_READY="${RUN_ROOT}/model.ready.json"
NATIVE_LOG="${RUN_ROOT}/native/native.log"
MODEL_LOG="${RUN_ROOT}/model/model.log"

NATIVE_PYTHONPATH="${REPO}/scripts:${REPO}/src:${WBC_SOURCE_ROOT}:${ISAACGYM_PYTHON}:${NATIVE_COMPAT_ROOT}"
MODEL_PYTHONPATH="${REPO}/scripts:${REPO}/src:${STAGE10_ROOT}/src/isaac-gr00t"
NATIVE_PATH="$(dirname "${NATIVE_NINJA}"):${PATH}"

cleanup() {
  active=0
  for pid in "${NATIVE_PID:-}" "${MODEL_PID:-}"; do
    if test -n "${pid}" && kill -0 "${pid}" 2>/dev/null; then
      active=1
      kill "${pid}" 2>/dev/null || true
    fi
  done
  if test "${active}" -eq 1; then
    : > "${ABORT_MARKER}" 2>/dev/null || true
  fi
  wait "${NATIVE_PID:-}" 2>/dev/null || true
  wait "${MODEL_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT

env \
  CUDA_VISIBLE_DEVICES="${GPU_UUID}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  HOME="${HOME}" TMPDIR="${TMPDIR}" TMP="${TMP}" TEMP="${TEMP}" \
  XDG_CACHE_HOME="${XDG_CACHE_HOME}" XDG_CONFIG_HOME="${XDG_CONFIG_HOME}" XDG_DATA_HOME="${XDG_DATA_HOME}" \
  PATH="${NATIVE_PATH}" LD_LIBRARY_PATH="${STAGE40_CT3_NATIVE_LD_LIBRARY_PATH}" \
  PYTHONPATH="${NATIVE_PYTHONPATH}" STAGE40_NATIVE_PYTHON="${NATIVE_PYTHON}" \
  "${NATIVE_PYTHON}" -B "${REPO}/scripts/stage40_corelab_native_hold.py" \
  --repo-root "${REPO}" --source-root "${WBC_SOURCE_ROOT}" \
  --output-dir "${RUN_ROOT}/native" --ready-marker "${NATIVE_READY}" \
  --release-marker "${RELEASE_MARKER}" --abort-marker "${ABORT_MARKER}" \
  --cutover-token-file "${TOKEN_FILE}" --gpu-uuid "${GPU_UUID}" \
  --sim-device cuda:0 --seed "${DEV_SEED}" \
  >"${NATIVE_LOG}" 2>&1 &
NATIVE_PID=$!

env \
  CUDA_VISIBLE_DEVICES="${GPU_UUID}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  HOME="${HOME}" TMPDIR="${TMPDIR}" TMP="${TMP}" TEMP="${TEMP}" \
  XDG_CACHE_HOME="${XDG_CACHE_HOME}" XDG_CONFIG_HOME="${XDG_CONFIG_HOME}" XDG_DATA_HOME="${XDG_DATA_HOME}" \
  HF_HOME="${HF_HOME}" HF_HUB_CACHE="${HF_HUB_CACHE}" TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTHONPATH="${MODEL_PYTHONPATH}" \
  "${MODEL_PYTHON}" -B "${REPO}/scripts/stage40_corelab_model_hold.py" \
  --repo-root "${REPO}" --stage10-root "${STAGE10_ROOT}" \
  --base-model-path "${BASE_MODEL_PATH}" --model-path "${MODEL_PATH}" \
  --dataset-dir "${DATASET_DIR}" --modality-config "${MODALITY_CONFIG}" \
  --cosmos-snapshot "${COSMOS_SNAPSHOT}" --output-dir "${RUN_ROOT}/model" \
  --ready-marker "${MODEL_READY}" --release-marker "${RELEASE_MARKER}" \
  --abort-marker "${ABORT_MARKER}" --cutover-token-file "${TOKEN_FILE}" \
  --gpu-uuid "${GPU_UUID}" --device cuda:0 --seed "${DEV_SEED}" \
  >"${MODEL_LOG}" 2>&1 &
MODEL_PID=$!

READY_TIMEOUT_S="${STAGE40_CT3_READY_TIMEOUT_S:-1800}"
[[ "${READY_TIMEOUT_S}" =~ ^[1-9][0-9]*$ ]]
READY=0
for _ in $(seq 1 "${READY_TIMEOUT_S}"); do
  if test -f "${NATIVE_READY}" && grep -q '"status": "ERROR"' "${NATIVE_READY}"; then
    echo "CT3 native constructor reported an error" >&2
    exit 2
  fi
  if test -f "${MODEL_READY}" && grep -q '"status": "ERROR"' "${MODEL_READY}"; then
    echo "CT3 model constructor reported an error" >&2
    exit 2
  fi
  if test -f "${NATIVE_READY}" && test -f "${MODEL_READY}" \
      && grep -q '"status": "READY"' "${NATIVE_READY}" \
      && grep -q '"status": "READY"' "${MODEL_READY}"; then
    READY=1
    break
  fi
  if ! kill -0 "${NATIVE_PID}" 2>/dev/null || ! kill -0 "${MODEL_PID}" 2>/dev/null; then
    echo "CT3 constructor exited before the dual-runtime barrier" >&2
    exit 2
  fi
  sleep 1
done
test "${READY}" = 1

GPU_TELEMETRY="${RUN_ROOT}/runtime/gpu_telemetry.csv"
printf '%s\n' 'timestamp_utc,index,uuid,name,memory.total_mib,memory.used_mib' > "${GPU_TELEMETRY}"
COEXIST_HOLD_S="${STAGE40_CT3_COEXIST_HOLD_S:-5}"
[[ "${COEXIST_HOLD_S}" =~ ^[1-9][0-9]*$ ]]
for _ in $(seq 1 "${COEXIST_HOLD_S}"); do
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv,noheader,nounits \
    | awk -v ts="${timestamp}" '{gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print ts "," $0}' >> "${GPU_TELEMETRY}"
  sleep 1
done

printf '%s\n' '{"schema":"CORELAB_STAGE40_CT3_RELEASE_V1","release_token":"CORELAB_STAGE40_CT3_RELEASE","gpu_uuid":"'"${GPU_UUID}"'"}' > "${RELEASE_MARKER}"

set +e
wait "${NATIVE_PID}"
NATIVE_RC=$?
wait "${MODEL_PID}"
MODEL_RC=$?
set -e
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv,noheader,nounits > "${RUN_ROOT}/runtime/gpu_after.csv"

if test "${NATIVE_RC}" -ne 0 || test "${MODEL_RC}" -ne 0; then
  echo "CT3 blocked: native_rc=${NATIVE_RC} model_rc=${MODEL_RC}" >&2
  exit 2
fi

# Combine the two independently written runtime counters without inventing a
# forward/step/update. The evidence verifier applies the exact zero budget.
"${MODEL_PYTHON}" - "${REPO}" "${RUN_ROOT}" "${BUNDLE}" "${GPU_UUID}" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1])
run_root = Path(sys.argv[2])
bundle_path = Path(sys.argv[3])
gpu_uuid = sys.argv[4]
spec = importlib.util.spec_from_file_location("stage40_corelab_evidence", repo / "scripts" / "stage40_corelab_evidence.py")
if spec is None or spec.loader is None:
    raise SystemExit("cannot load Corelab evidence module")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
native = json.loads((run_root / "native" / "native_runtime.json").read_text(encoding="utf-8"))
model = json.loads((run_root / "model" / "model_runtime.json").read_text(encoding="utf-8"))
module._require(native.get("status") == "PASS", "native runtime did not finish PASS")
module._require(model.get("status") == "PASS", "model runtime did not finish PASS")
names = module.COUNTER_NAMES
sources = {"native": native["counters"], "model": model["counters"]}
before = {name: 0 for name in names}
after = {}
for name in names:
    values = []
    for source, counters in sources.items():
        row = counters[name]
        module._require(row["before"] == 0 and row["delta"] == row["after"], f"invalid {source} counter: {name}")
        values.append(int(row["after"]))
    after[name] = max(values)
    module._require(after[name] == 0, f"CT3 nonzero counter: {name}")
observation = module.build_counter_observation(before, after)
observation["gpu_uuid"] = gpu_uuid
observation["source_observations"] = sources
(run_root / "counter_observations.json").write_text(
    json.dumps(observation, sort_keys=True, indent=2) + "\n", encoding="utf-8"
)
metadata = {
    "schema": "CORELAB_STAGE40_CT3_RUNTIME_RUN_V1",
    "status": "PASS",
    "stage": "RTX-runtime",
    "execution_class": "CORELAB_STAGE40_RTX_RUNTIME_QUALIFICATION_V1",
    "bundle": str(bundle_path),
    "bundle_sha256": module.sha256_file(bundle_path),
    "gpu_uuid": gpu_uuid,
    "native_return_code": 0,
    "model_return_code": 0,
    "model_forward_count": 0,
    "wbc_step_count": 0,
    "reference_update_count": 0,
    "training": False,
    "scientific_rows": 0,
    "forbidden_operations": module.expected_forbidden_operations(),
}
(run_root / "run_metadata.json").write_text(
    json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8"
)
PY

printf '%s\n' '{"observation_body":"native/observation_body.npz","observation_wrist":"native/observation_wrist.npz","state_v1":"native/state_v1.json","native_runtime":"native/native_runtime.json","model_runtime":"model/model_runtime.json","counter_observations":"counter_observations.json","logs":"stage40_ct3_runtime.log","telemetry":"runtime/gpu_telemetry.csv"}' > "${RUN_ROOT}/roles.json"
"${MODEL_PYTHON}" "${REPO}/scripts/stage40_corelab_evidence.py" manifest \
  --stage RTX-runtime --evidence-root "${RUN_ROOT}" --roles-json "${RUN_ROOT}/roles.json" \
  --output "${RUN_ROOT}/evidence_manifest.json"
"${MODEL_PYTHON}" "${REPO}/scripts/stage40_corelab_evidence.py" build-result \
  --bundle "${BUNDLE}" --manifest "${RUN_ROOT}/evidence_manifest.json" \
  --evidence-root "${RUN_ROOT}" --output "${RUN_ROOT}/RTX-runtime_result.json"
"${MODEL_PYTHON}" "${REPO}/scripts/stage40_corelab_evidence.py" seal \
  --result "${RUN_ROOT}/RTX-runtime_result.json" --bundle "${BUNDLE}" \
  --evidence-root "${RUN_ROOT}" --output "${RUN_ROOT}/RTX-runtime_result.seal.json"
echo "CORELAB_STAGE40_RTX_RUNTIME_PASS=${RUN_ROOT}/RTX-runtime_result.seal.json"
