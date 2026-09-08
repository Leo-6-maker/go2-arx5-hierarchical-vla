# Stage45B H200 handoff

This document is the direct-execution handoff for `PI_GATE_STATE_0073`. It
does not authorize Corelab to train; the Corelab executor has no H200 network
route in this session. An H200 operator may execute only the single
`B_FRESH_BASE_FULL24` intervention after checking the B1 manifest.

## Frozen package

```text
branch: stage45b/full24-fresh-base-execution
manifest: manifests/STAGE45B_FULL24_FRESH_BASE_EXECUTION_MANIFEST_V1.json
runner: scripts/stage45b_full24_fresh_base_train.py
PBS: hpc/pbs/stage45b_full24_fresh_base.pbs
intervention: B_FRESH_BASE_FULL24
seed: 2222
steps: 2000
checkpoints: 1000, 2000 (full trainer state)
```

The bundle binds the exact FULL24 tree, B0 StructuredAction statistics,
BASE model, GR00T/Cosmos/modality identities, StructuredActionV1 contract,
and the Stage38 optimizer recipe. It explicitly forbids warm-start Option A,
new data, validation/holdout reads, WBC/simulator/hardware, and any retry after
an optimizer step.

## Suggested Hopper path mapping

The locations below follow the existing Stage22 handoff/runtime layout. If a
path has moved, copy or bind the same immutable bytes and verify every hash in
the manifest before launch; do not substitute a checkpoint or regenerate stats.

```text
STAGE45B_BASE_MODEL_PATH=/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/GR00T-N1.7-3B
STAGE45B_DATASET_DIR=/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/lerobot_train
STAGE45B_GROOT_SOURCE=/scratch/e1192548/stage22_h200_runtime_deb920a_20260902/src/isaac-gr00t
STAGE45B_COSMOS_ROOT=/scratch/e1192548/stage22_h200_runtime_deb920a_20260902/hf_home/hub/models--nvidia--Cosmos-Reason2-2B
STAGE45B_MODALITY_CONFIG=/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/modality/go2_arx5_modality.py
STAGE45B_CONFIG_PATH=/scratch/e1192548/stage28_h2_20260904/stage28_h2_600695.hopper-m-02/h2_training/experiment_cfg/config.yaml
STAGE45B_STATS_ARTIFACT=/scratch/e1192548/stage45b_full24_20260908/STAGE45B_FULL24_STRUCTURED_STATS.json
```

The B0 artifact currently sealed on Corelab is also available at
`/var/cache/users/liuyu-nfs/go2-arx5-vla-stage45b/b0_20260908_v2/` with SHA256
`49c7139dff09d5795e31205c7e72cb11e5801ce5f176572ebc1ec8d91f310c3d`.
Transfer it byte-for-byte if the shared path is not visible from Hopper.

## Required H200 procedure

1. Check out the branch at or after the manifest's `source_head`, with a clean
   worktree. Fill `PROJECT_ID_REQUIRED` only after normal project discovery.
2. Verify one visible H200, the BASE/FULL24/Cosmos/source/modality/config
   identities, and the B0 artifact hash. The runner repeats these checks
   before importing GR00T.
3. Run the PBS template exactly once. Do not add `--resume`, `--eval`, a
   second `qsub`, precision/offload/quantization changes, or an alternate
   starting checkpoint.
4. Preserve the output root, scheduler stdout, `stage45b_runtime_preflight.json`,
   `stage45b_events.jsonl`, `stage45b_training_result.json`, both checkpoints,
   and all failure evidence. Return their SHA256 inventories to Corelab.

If the run fails after any optimizer step, stop and report to PI; do not resume
or rerun. A pre-step failure is still reported first and can be retried only
as the identical bundle after proving zero optimizer exposure.

## Boundary

The expected successful token is
`HOPPER_STAGE45B_FULL24_FRESH_BASE_TRAINING_COMPLETE`. This is training-only
evidence; it is not a task-success, generalization, robustness, or WBC result.
After returning the immutable artifacts, stop for the B3 offline evaluation
gate. Do not run B3 from this handoff unless the next PI gate explicitly
admits it.
