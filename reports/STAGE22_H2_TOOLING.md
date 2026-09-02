# Stage22 H2 Corelab Tooling

Date: 2026-09-02 (Asia/Singapore)

## Scope

This Corelab work prepares the authoritative H2 comparison while the actual
R0/R1 training remains reserved for one matched H200/runtime. No H2 training,
checkpoint selection, validation inference, final-test read, WBC execution, or
hardware action was performed here.

## Frozen tools

| Tool | Purpose | Scientific boundary |
|---|---|---|
| `scripts/stage22_h2_paired_comparison.py` | Pair R0/R1 profile manifests by `(input_id, inference_seed)` and emit R1-minus-R0 deltas | no loss-based selection; no trajectory correction |
| `scripts/stage22_aggregate_profile_seeds.py` | Aggregate one recipe across the locked validation diffusion seeds | binds the actual snapshot-lock file; exact seed/snapshot/overlap set required |
| `scripts/stage22_audit_h2_run_integrity.py` | Verify matched base, dataset tree, split lock, recipe flags, trainable count, runtime, and GPU identity | failed audit blocks H2 claims |
| `scripts/stage22_audit_h2_direct_launch.py` | Audit two direct frozen-launch run manifests plus H2 execution provenance and requires-grad inventories | does not rerun training or select a checkpoint |
| `scripts/stage22_capture_gpu_evidence.py` | Capture a read-only, SHA256-bound `nvidia-smi` identity artifact per recipe | no process control or workload inspection |
| `scripts/stage22_probe_trainable_modules.py` | Construct the frozen Stage20S model and inventory live `requires_grad` state without a forward/backward/optimizer step | model construction only; no training or inference |
| `scripts/stage22_checkpoint_provenance.py` | Hash-seal a future candidate checkpoint and verify a transferred copy | no transfer or model selection |

The paired comparison is restricted to the locked validation split and requires
the Git-tracked `manifests/STAGE22_VALIDATION_SNAPSHOT_LOCK_V1.json`. It rejects
anything other than all three inference seeds `2223/2224/2225`, and every seed
must contain exactly the six frozen `red_target_holdout_e` snapshots and five
absolute-time overlap rows. Multiple per-seed manifests are accepted only when
every `(snapshot, inference_seed)` and every absolute-time overlap key is
present exactly once on both sides.

## Required paired quantities

The output contains per-recipe summaries and `delta_r1_minus_r0` for position
and rotation errors, endpoint errors, path/speed/roughness deviation from 1,
initial discontinuity, common-future overlap errors, tangent error, and the
old-prediction-versus-new-first handoff error. Negative deltas mean a lower
error for R1. SafetyGate decisions and finite-output status are reported
separately.

Training loss is intentionally absent from the selection basis. The output is
an attribution record; it does not promote R0 or R1.

## Direct-launch compatibility

The authoritative H2 runner is the frozen Stage20S launcher at
`deb920a070141e6a32a88cd0da6719828647e1e6`, invoked directly once for R0 and
once for R1. `scripts/stage22_audit_h2_direct_launch.py` consumes those two
`STAGE20S_GR00T_TRAINING_RUN_V1` manifests plus an H2 execution-provenance
manifest; it does not require or rerun `stage22_train_recipes.py`. The newer
wrapper remains optional non-authoritative tooling only. The repository PBS
template has been aligned to this direct path and checks the frozen launcher
file hash before either run.

The direct audit now requires exact canonical identities for the GR00T
revision, base tree, train fixture tree, modality config, and split lock. It
also requires distinct SHA256-bound `nvidia-smi` artifacts for R0 and R1 and
inventory files emitted by the read-only live model-construction probe. These
are integrity gates; they do not establish profile quality or checkpoint
selection.

## Provenance P0 hardening

The live module probe now requires the actual recipe run manifest and records
its SHA256 in `source_run_manifest_sha256`; it also checks the frozen action
timing, recipe flags, shared paths, and base-model index hash before model
construction. This prevents a manually assembled inventory from being
attached to a different run.

GPU evidence has two deliberately separate paths. A future PBS run may use
`STAGE22_GPU_EVIDENCE_V1` only when its capture is bound to the authoritative
PBS job window. A completed H2 run must use
`STAGE22_GPU_EVIDENCE_FROM_IMMUTABLE_H2_LOG_V1`, recovered from the training
log and completed PBS accounting sidecar. A present-time `nvidia-smi` query
cannot certify the already-finished H2 allocation. If the immutable log does
not recover one exact UUID, the correct result is
`GPU_EVIDENCE_RETROACTIVE_INSUFFICIENT`.

The direct auditor also requires `holdout_f_accessed=false` and
`contains_credentials=false` in the execution provenance, verifies historical
source/accounting/run-manifest hashes, and accepts the direct frozen-launch
manifests without requiring the newer Stage22 wrapper.

## 4080S use

The 4080S may run train-only or synthetic CPU/GPU smoke checks for these tools.
Such smoke output is non-authoritative and must not read validation `e` or
locked final-test `f`. The previously corrected 32 GiB R1 one-step preflight
still failed at Adam `optimizer.step()` and therefore does not unlock full
training on Corelab.

## Current status

```text
H2_TOOLING: FROZEN
H1_H200: PASS
H2_H200_TRAINING: RUNNING_OR_DISPATCHED
CORELAB_H2_TOOLING: HARDENED_DIRECT_LAUNCH_AUDIT; AWAITING_H2_ARTIFACT_AUDIT
CORELAB_AUTHORITATIVE_POSTPROCESSING_GATE: NOT_YET_QUALIFIED_UNTIL_DIRECT_H2_AUDIT_PASS
CHECKPOINT_SELECTION: NOT_STARTED
TRANSFER: NOT_STARTED
```

No authoritative H2 bundle was present in the Corelab node-local intake at
the time of this update. The next action is immutable-bundle intake and audit;
H2 must not be rerun merely to manufacture post-hoc GPU evidence.
