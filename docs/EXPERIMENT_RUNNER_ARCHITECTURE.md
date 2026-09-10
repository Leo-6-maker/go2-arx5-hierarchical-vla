# Reusable train/eval infrastructure

This document defines the future-facing runner boundary introduced after the
PI0100 Stage45C execution.  It does not alter or regenerate any Stage45B or
Stage45C evidence.

## Ownership layers

```text
experiment manifest
        |
        +--> generic manifest validator / runner
        |          |
        |          +--> inference admission -> groot_h200_inference_backend_v1
        |          |       -> raw handoff -> raw seal -> accepted CPU decode
        |          |       -> decoded evidence -> metrics -> inventory/seal
        |          +--> training admission -> groot_h200_training_backend_v1
        |                  -> full trainer-state checkpoint + seals
        |
        +--> stage-specific frozen wrapper
                   |
                   +--> optional model/runtime work performed by a future gate
```

The generic layer is deliberately runtime-independent.  It does not import
Torch, GR00T, CUDA, PBS, WBC, the simulator, or hardware-control modules.
Historical wrappers remain the authority for their historical evidence; the
new entry points are the canonical path for future manifests.

## Evaluation entry point

`scripts/eval_checkpoint_cohort.py` has two modes:

```bash
python scripts/eval_checkpoint_cohort.py \
  --manifest path/to/evaluation-manifest.json --plan

python scripts/eval_checkpoint_cohort.py \
  --manifest path/to/evaluation-manifest.json \
  --raw-predictions path/to/raw-predictions.jsonl \
  --output-dir path/to/fresh-output
```

The manifest must explicitly provide:

* ordered windows, target IDs, anchors, targets, and an explicit timestamp
  profile.  Generic windows require only finite, H-length, strictly increasing
  timestamps.  A fixed cadence is admitted only when a named timing profile
  explicitly requires it;
* ordered checkpoint labels, checkpoint content identities, runtime mappings,
  statistics identities, and a cohort/dataset identity;
* raw `[1,40,132]` and decoded `[40,10]` ABIs;
* `body_rgb`, `wrist_rgb`, and `state`, with numeric target XYZ excluded;
* same-noise pairing and a named versioned sampler profile.  The preserved
  four-step contract is registered as
  `GROOT_N1D7_SOURCE_MAPPED_EULER4_V1`; its timesteps, step size, and call
  count are profile semantics rather than anonymous generic constants.  The
  timesteps and call count must be JSON integers, and `dt` must be a finite
  JSON number; booleans, strings, and coercible fractional values are rejected;
* offline/local runtime identity, fresh output root, and locked
  Validation-E/Holdout-F/WBC/simulator/hardware boundaries.

The stage-specific runtime supplies a JSON/JSONL record for each explicit
checkpoint/window pair.  Canonical execution requires a concrete manifest
path and unconditionally verifies every manifest-bound checkpoint, runtime
mapping, statistics, cohort, and noise-bank identity before the first decoder
call; there is no CLI opt-out.  Each record carries a raw `.npy` path and both
file and array SHA256 values.  The generic runner copies and fsyncs the raw
file to the fresh output root and records that handoff before calling the
accepted Stage40 decoder.  For each successful window it persists the decoded `[H,10]`
`.npy` array, appends the complete metric row to `per_window.jsonl`, flushes,
and fsyncs before continuing.  A late decoder/metric failure therefore leaves
prior decoded arrays and rows usable, together with a durable failure seal.  A
completed run also writes `per_window.csv`, checkpoint aggregates, exact
runtime and statistics identities, and an SHA256 artifact inventory.

The generic runner never calls a model.  This makes the boundary testable on
CPU and prevents a future evaluator from silently growing a second model,
decode, or metric implementation.  A stage wrapper may produce the raw
handoff after its own model admission; it must not replace the decoder or
metric semantics.

## Pure analyzer

`scripts/analyze_checkpoint_eval.py` reads `per_window.jsonl` (or a directory
containing it) and writes, without changing the source:

* `per_window.json` and `per_window.csv`;
* deterministic per-target summaries;
* cohort distributions;
* paired per-window delta tables and distributions;
* `report.md` and `analysis_manifest.json`.  The analysis output root must be
  new or already empty; a reused/non-empty root is rejected before any artifact
  is overwritten.

Pair deltas use `right - left`.  The experiment manifest may name primary and
secondary pairs.  If no pairs are supplied, the analyzer performs descriptive
per-window, per-target, and cohort analysis only; it never guesses a
comparison from checkpoint labels.  No threshold or pass/fail rule is
embedded in the analyzer.

## Training entry point

`scripts/train_adaptation.py` validates a
`GO2ARX5_REUSABLE_ADAPTATION_TRAINING_MANIFEST_V1` and emits a zero-side-effect
handoff plan for `groot_h200_training_backend_v1`.  The checked-in training
template is intentionally non-executable: every recipe value is
`__REQUIRED__`.  Actual manifests must provide optimizer, scheduler, learning
rate, warmup, precision, trainable components, seed, batching, fresh/resume
mode, and save steps explicitly.  The validator requires `max_steps` in
`save_steps` and `save_only_model=false` so a future backend can seal full
trainer state.  It refuses to infer resume or fresh semantics, and requires
`learning_rate` and `weight_decay` to be finite JSON numbers rather than
coercible strings or booleans.

The plan contains base checkpoint, dataset manifest/statistics, provenance,
runtime, output, recipe, backend-boundary, and late-bound runtime identities.  A
stage-specific wrapper is the only layer allowed to cross into model
construction and training after its gate admission; its execution admission
must reject a plan whose `verified_artifacts` is missing or marked `skipped`.
This generic command itself cannot start either.

## H200 inference-admission preflight

`src/go2arx5_vla/evaluation/h200_admission.py` and
`scripts/validate_h200_inference_admission.py` define a static, stage-specific
preflight immediately before a future H200 inference adapter.  The resolved
admission document must exactly cross-bind the repository execution HEAD,
materialized checkpoint paths/content, checkpoint-specific statistics, the
cohort/dataset path and identity, fresh output root, offline mode, runtime
admission identity, sampler/noise identity, and every locked boundary.  The
preflight has no model import/construction/forward side effect; a future H200
adapter must build this document from observed/materialized runtime state and
run it immediately before model import/construction, rather than copying the
manifest into a second JSON file.  A separate PI review is still required
before a canary.

## Historical boundary

The following remain frozen and are not rewritten by this infrastructure:

* Stage45B fresh-BASE training wrappers, recipe, checkpoints, and evidence;
* Stage45C PI0100 evaluator/runtime wrappers and the successful H200 evidence;
* existing Stage45C protocol, mapping, Golden Release, checkpoint, sampler,
  decoder, and metric artifacts.

No new H200 job, training, Stage45C evaluation, Validation-E, Holdout-F, WBC,
simulator, or hardware execution is part of this branch.

## Status vocabulary and handoff boundaries

These labels describe different facts and must not be conflated:

| Status | Meaning |
|---|---|
| `historical frozen` | An existing wrapper, protocol, or evidence bundle whose bytes and scientific meaning are preserved. |
| `generic CPU` | Manifest admission, raw-handoff consumption, deterministic decode/metrics, or pure analysis; no model construction or inference. |
| `H200 backend implementation` | The named, late-bound interface that a future stage wrapper may implement after a separate gate; the generic planner has zero model/GPU side effects. |
| `H200 qualification` | A separately authorized canary or bounded execution proving the backend/runtime boundary; it is not established by CPU tests. |
| `scientific execution` | A gated model training or inference run whose result may be interpreted under its approved protocol. |

This branch is a `REUSABLE_H200_V1_FREEZE_CANDIDATE`: it implements and tests
the admission and backend boundaries, but it is not `REUSABLE_H200_V1_FROZEN`
and does not claim H200 qualification.

## Contract invariants

The generic layer keeps the established interfaces rather than reinterpreting
them:

| Boundary | Invariant |
|---|---|
| raw model output | `[1,40,132]`, retained and hashed before decode |
| structured decode | accepted Stage40 inverse normalization, K16/T16 |
| physical action | `[40,10]`, EEF position/rot6d/gripper |
| model input | `body_rgb`, `wrist_rgb`, `state`; target XYZ outside input |
| timestamps | generic finite/strictly-increasing H-vector; fixed cadence only via a named timing profile |
| sampler | named versioned profile; strict JSON numeric ABI; same noise per window |
| safety/actuation | no corrective SafetyGate, WBC, simulator, or hardware action |
| analysis | read-only, paired, descriptive, no hardcoded threshold |
