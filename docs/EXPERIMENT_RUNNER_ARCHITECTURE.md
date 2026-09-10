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
        |          +--> raw prediction handoff -> raw seal -> accepted decode -> metrics
        |          +--> explicit training plan -> stage-specific backend handoff
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
  --output-dir path/to/fresh-output \
  --verify-artifacts
```

The manifest must explicitly provide:

* ordered windows, target IDs, anchors, targets, and H40/T16 timestamps;
* ordered checkpoint labels, checkpoint content identities, runtime mappings,
  and statistics identities;
* raw `[1,40,132]` and decoded `[40,10]` ABIs;
* `body_rgb`, `wrist_rgb`, and `state`, with numeric target XYZ excluded;
* same-noise pairing and the fixed four sampler timesteps `[0,250,500,750]`
  with `dt=0.25`;
* offline/local runtime identity, fresh output root, and locked
  Validation-E/Holdout-F/WBC/simulator/hardware boundaries.

The stage-specific runtime supplies a JSON/JSONL record for each explicit
checkpoint/window pair.  Each record carries a raw `.npy` path and both file
and array SHA256 values.  The generic runner copies and fsyncs the raw file to
the fresh output root and records that handoff before calling the accepted
Stage40 decoder.  It then writes per-window rows, checkpoint aggregates, and a
run manifest.  A decoder failure seals `failure.json` while leaving already
persisted raw evidence intact.

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
* `report.md` and `analysis_manifest.json`.

Pair deltas use `right - left`.  The experiment manifest may name primary and
secondary pairs; the conventional Stage45C pair names are recognized only as
a fallback when those labels are present.  No threshold or pass/fail rule is
embedded in the analyzer.

## Training entry point

`scripts/train_adaptation.py` validates a
`GO2ARX5_REUSABLE_ADAPTATION_TRAINING_MANIFEST_V1` and emits a zero-side-effect
backend handoff plan.  All recipe values are required, including optimizer,
scheduler, learning rate, warmup, precision, trainable components, seed,
batching, fresh/resume mode, and save steps.  The validator requires
`max_steps` in `save_steps` and `save_only_model=false` so a future backend can
seal full trainer state.  It refuses to infer resume or fresh semantics.

The plan contains base checkpoint, dataset manifest/statistics, provenance,
runtime, output, and recipe identities.  A stage-specific wrapper is the only
layer allowed to cross into model construction and training after its gate
admission; this generic command itself cannot start either.

## Historical boundary

The following remain frozen and are not rewritten by this infrastructure:

* Stage45B fresh-BASE training wrappers, recipe, checkpoints, and evidence;
* Stage45C PI0100 evaluator/runtime wrappers and the successful H200 evidence;
* existing Stage45C protocol, mapping, Golden Release, checkpoint, sampler,
  decoder, and metric artifacts.

No new H200 job, training, Stage45C evaluation, Validation-E, Holdout-F, WBC,
simulator, or hardware execution is part of this branch.

## Contract invariants

The generic layer keeps the established interfaces rather than reinterpreting
them:

| Boundary | Invariant |
|---|---|
| raw model output | `[1,40,132]`, retained and hashed before decode |
| structured decode | accepted Stage40 inverse normalization, K16/T16 |
| physical action | `[40,10]`, EEF position/rot6d/gripper |
| model input | `body_rgb`, `wrist_rgb`, `state`; target XYZ outside input |
| sampler | four steps `[0,250,500,750]`, `dt=0.25`, same noise per window |
| safety/actuation | no corrective SafetyGate, WBC, simulator, or hardware action |
| analysis | read-only, paired, descriptive, no hardcoded threshold |
