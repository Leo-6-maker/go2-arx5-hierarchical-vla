#!/bin/bash
# Stage30 import-only provenance closure. This is a login-node CPU/container task, not PBS.

set -euo pipefail

: "${STAGE30_REPO:?Set the clean Stage30 checkout}"
: "${STAGE30_OUTPUT_ROOT:?Set a new scratch-backed Stage30 output root}"
: "${STAGE30_GROOT_SOURCE:?Set the pinned GR00T source checkout}"
: "${STAGE10_ROOT:?Set the pinned Stage29 runtime root}"
: "${STAGE30_EXPECTED_PROJECT_SHA:?Set exact committed Stage30 HEAD}"
: "${STAGE30_EXPECTED_RUNNER_SHA256:?Set exact P0 runner SHA256}"
: "${STAGE30_EXPECTED_WRAPPER_SHA256:?Set exact P0 wrapper SHA256}"
: "${STAGE30_PI_GATE_ID:?Set PI_GATE_STATE_0010}"

test "${STAGE30_PI_GATE_ID}" = "PI_GATE_STATE_0010"
test ! -e "${STAGE30_OUTPUT_ROOT}" || {
  echo "FAIL_CLOSED: Stage30 output already exists: ${STAGE30_OUTPUT_ROOT}" >&2
  exit 2
}
mkdir -p "${STAGE30_OUTPUT_ROOT}" "${STAGE30_OUTPUT_ROOT}/runtime/tmp" \
  "${STAGE30_OUTPUT_ROOT}/runtime/cache" "${STAGE30_OUTPUT_ROOT}/runtime/config" \
  "${STAGE30_OUTPUT_ROOT}/runtime/data"

cd "${STAGE30_REPO}"
test "$(git rev-parse HEAD)" = "${STAGE30_EXPECTED_PROJECT_SHA}"
test -z "$(git status --short)"

RUNNER="scripts/stage30_runtime_import_origin.py"
WRAPPER="hpc/stage30_p0_import_origin.sh"
test "$(sha256sum "${RUNNER}" | awk '{print $1}')" = "${STAGE30_EXPECTED_RUNNER_SHA256}"
test "$(sha256sum "${WRAPPER}" | awk '{print $1}')" = "${STAGE30_EXPECTED_WRAPPER_SHA256}"

. /etc/profile.d/modules.sh 2>/dev/null || true
module load singularity

IMAGE=/app1/common/singularity-img/hopper/pytorch/pytorch_25.12-py3.sif
FFMPEG_BUNDLE=/scratch/e1192548/stage22_h200_ffmpeg4_bundle_20260902
CONTAINER_SITE="${STAGE10_ROOT}/runtime/venv/lib/python3.12/site-packages"
test -f "${IMAGE}"
test -d "${FFMPEG_BUNDLE}"
test -d "${CONTAINER_SITE}"

FFMPEG_LD="/.singularity.d/libs:${FFMPEG_BUNDLE}:${FFMPEG_BUNDLE}/lib/x86_64-linux-gnu:${FFMPEG_BUNDLE}/usr/lib/x86_64-linux-gnu:${FFMPEG_BUNDLE}/usr/lib/x86_64-linux-gnu/pulseaudio:${CONTAINER_SITE}/torch/lib"
STAGE29_EXACT_PYTHONPATH="${STAGE30_REPO}:${STAGE30_REPO}/src:${CONTAINER_SITE}:${STAGE30_GROOT_SOURCE}"

singularity exec --bind /app1:/app1 --bind /scratch:/scratch "${IMAGE}" env \
  PYTHONPATH="${STAGE29_EXACT_PYTHONPATH}" \
  LD_LIBRARY_PATH="${FFMPEG_LD}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  GROOT_HF_LOCAL_FIRST=1 GROOT_PATCH_MISTRAL=1 NO_ALBUMENTATIONS_UPDATE=1 \
  XDG_CACHE_HOME="${STAGE30_OUTPUT_ROOT}/runtime/cache" \
  XDG_CONFIG_HOME="${STAGE30_OUTPUT_ROOT}/runtime/config" \
  XDG_DATA_HOME="${STAGE30_OUTPUT_ROOT}/runtime/data" \
  TMPDIR="${STAGE30_OUTPUT_ROOT}/runtime/tmp" \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 "${RUNNER}" \
    --project-repo "${STAGE30_REPO}" \
    --expected-project-sha "${STAGE30_EXPECTED_PROJECT_SHA}" \
    --pinned-groot-source "${STAGE30_GROOT_SOURCE}" \
    --container-image "${IMAGE}" \
    --wrapper "${STAGE30_REPO}/${WRAPPER}" \
    --expected-runner-sha256 "${STAGE30_EXPECTED_RUNNER_SHA256}" \
    --expected-wrapper-sha256 "${STAGE30_EXPECTED_WRAPPER_SHA256}" \
    --authorization-token "${STAGE30_PI_GATE_ID}" \
    --output "${STAGE30_OUTPUT_ROOT}/stage30_p0_runtime_import_origin.json"
