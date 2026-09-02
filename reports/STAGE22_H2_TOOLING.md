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
| `scripts/stage22_audit_h2_run_integrity.py` | Verify matched base, dataset tree, split lock, recipe flags, trainable count, runtime, and GPU identity | failed audit blocks H2 claims |
| `scripts/stage22_checkpoint_provenance.py` | Hash-seal a future candidate checkpoint and verify a transferred copy | no transfer or model selection |

The paired comparison requires an explicit `--expected-split` of `validation`
or `final_test`. The final-test split cannot be read accidentally by a
validation-selection invocation. Multiple per-seed manifests are accepted only
when every `(snapshot, inference_seed)` and every absolute-time overlap key is
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

## 4080S use

The 4080S may run train-only or synthetic CPU/GPU smoke checks for these tools.
Such smoke output is non-authoritative and must not read validation `e` or
locked final-test `f`. The previously corrected 32 GiB R1 one-step preflight
still failed at Adam `optimizer.step()` and therefore does not unlock full
training on Corelab.

## Current status

```text
H2_TOOLING: FROZEN
H2_H200_TRAINING: PENDING_H1_PASS_ON_H200
CHECKPOINT_SELECTION: NOT_STARTED
TRANSFER: NOT_STARTED
```
