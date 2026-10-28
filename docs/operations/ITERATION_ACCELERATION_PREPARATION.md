# Iteration Acceleration Preparation

Status: **PREPARED_NOT_AUTHORIZED**. User chose script preparation only on
2026-10-13 while discussing the next science task card. No added collection,
GPU/model service, PBS submission or physical benchmark is authorized here.

## Implemented

- `scripts.iteration.plan`: pure CPU planners for the next collection bank and
  Hopper loader comparison; no launch or submission capability.
- `scripts.iteration.queue`: durable single-coordinator reservation state, unique
  pre-registered first-attempt IDs, isolated episode output, bounded outstanding
  CPU/steps/RGB/bytes, no automatic retries. Unknown workers retain reservations.
- `scripts.iteration.fixture`: two spawned synthetic CPU workers with separate
  outputs and durable receipts. No simulator, robot or model imports.
- `scripts.iteration.rgb_cache`: bounded decoded-RGB LRU prototype. Misses check
  original SHA and RGB dimensions; hits check device/inode/size/mtime/ctime.
  Returned arrays are copies. No numeric labels, normalization or augmentation
  changes. Not connected to the scientific NativeDataset.

```bash
python -m scripts.iteration.plan --kind corelab --input NEXT_TASK_DRAFT.json
python -m scripts.iteration.plan --kind hopper --input FROZEN_INPUT_IDENTITIES.json
python -m scripts.iteration.fixture --card SYNTHETIC_CARD.json --root NEW_FIXTURE_OUTPUT
python -m pytest -q tests/test_iteration_preparation.py
```

The fixture card uses `status=NOT_AUTHORIZED`, unique episodes with `id`, `split`
and `seed`, `workers=2`, `cores_per_worker=1`, `max_concurrent_cores=4`, plus
`per_episode` and `limits` for `physics_steps`, `rgb_pairs`, `cpu_s`, `output_bytes`.
These are reservation-contract fixtures, not real physics counts. The coordinator
and future auditor each reserve one core; renderer thread counts must be included
in the actual next GO. No claim of twofold speedup has been measured.

## Required Before Live Workers

The native parallel adapter is **NOT_IMPLEMENTED**. PI0292's existing
`PHYSICS_OWNER.lock` and exclusive Budget are deliberately unchanged; launching
two copies of that collector is invalid. A new task must bind canonical
`enter.sh 4080-simulation`, standard RUN_MANIFEST, fixed source/entry release,
host registry, live lease, original deadline, and output role before dispatch.
Workers must use coordinator-issued tickets and worker-local budgets rather than
edit the shared ledger. The coordinator alone reserves and reconciles resources;
progress messages require attempt identity, sequence and monotonic counts.

Start with a serial canary, then a pre-registered 1-vs-2-worker benchmark. Each
process owns its own MuJoCo model/data/renderer. Keep dt, integration counts,
safety checks, camera observations, teacher and score semantics unchanged. Audit
and export only terminal immutable episodes, with bounded queues and backpressure.
Retain all failed first attempts. Quality/safety failure stops new dispatch;
crashes remain UNKNOWN until terminal PID/start evidence and costs are reconciled.
Reserve CPUs for S6 relay; pause the next-bank producer at a verified boundary.

## Hopper Acceleration

Only use an owned legal H200 allocation with project membership, personal quota,
job identity and original science limits freshly checked. Keep neural work off
CoreLab and login nodes. The planner requires fixed HF revision, dataset manifest,
statistics, processor and source seal, and TRAIN-only optimization.

PI0292 already uses two persistent loader workers and prefetch factor two.
Compare 0/2/4 workers in a separately admitted next run, retaining the consumed
sample cursor rather than the prefetch cursor, isolated loader RNG and native
processor/collator. Measure load wait, update time, RSS, cache hits, scratch and
full-checkpoint cost. The LRU is a prototype: it needs a full input hash audit,
immutable snapshot/lease, serial/cache/worker/resume byte equivalence, and actual
loss/gradient/update/RNG equivalence within allocation before use in training.
Metadata checks detect ordinary changes, not adversarial same-metadata rewrites.

No VLM feature cache, compile, attention replacement, DDP, LoRA, microbatch change
or checkpoint thinning is enabled. Dataset v2 and its statistics remain frozen;
new data enters v3, never a live v2 loader. Engineering CPU acceptance is not
model qualification or a scientific GO. Server publication/deployment/fixture
execution are separate receipts; none implies a real speedup.
