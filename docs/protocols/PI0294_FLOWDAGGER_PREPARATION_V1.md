# FlowDAgger Preparation V1

2026-10-16. Status: **CODE_PREPARED_ONLY / NO_NEW_SCIENTIFIC_GO**.
The user requested code preparation while the adopted E0 GPU job waits.
This document does not change E0, create a scientific window or permit FD runs.

## Fixed Inputs

- Upstream [Microsoft FlowDAgger](https://github.com/microsoft/FlowDAgger/tree/3426f246510fd67a9e39c3c08bd5b02d586ad688), MIT; twelve inspected upstream/native file blobs are locked in
  `third_party/flowdagger/UPSTREAM_SOURCE_LOCK.json`.
- Current official GR00T: `51d4c89f72fda44cbf77285c6a8114b52676b8a1`.
  Upstream example uses `9c7e746b2cd37a810070a98ef41d290a07e806c2`;
  compatibility must be tested, not inferred from the N1.7 name.
- Actual read-only M2048 metadata: horizon40, padded dimension132, four native
  Euler steps, effective EEF10; original v3.0 statistics and processor.
- E0 protocol `22e5f0baeed0af662a7cf1c97edae1f4060947fe`; running capsule
  `9a699d28c6a585df1abb5aa906e41e8d8c025bdc` stays immutable. No FlowDAgger
  module is imported into it, no training, WBC/S1 or v3.1/BASE change.

## What Is Implemented

`src/go2arx5_vla/flowdagger/` contains prepared library components, not a collector
or approved remote launcher:

- `contracts.py`: same-observation student/expert H40, hash-bound original
  reference/S1 receipts, pure safety arbitration, and separate proposal versus
  actual prefix/ACK evidence. A native failure cannot trigger expert rescue.
- `aggregation.py`: bounded label index shared by action-space DAgger and
  FlowDAgger, pinned model/statistics, all-analysis denominator, no duplicates,
  silent eviction, DEV-to-TRAIN conversion or mutation of the old BC bank.
- `codec.py`: original processor absolute EEF10 -> sole relative conversion ->
  normalization -> full model tensor -> original decode. Structural model
  padding is not BASE supervision. Short H8 or EEF13 inputs are rejected.
- `inversion.py`: adapted upstream forward Euler and per-step fixed-point
  inverse, same forward time grid, charge-before-call budget and reconstruction
  error evidence. Finite inversion output alone is never marked qualified.
- `native.py`: original observation-only preparation, one cached backbone/state
  context without a discarded policy draw, frozen/eval checks, original
  alternate-VL masks/position embedding and native velocity calculation.
- `steering.py`: detached observation embeddings -> MLP initial latent; latent
  MSE detaches expert targets and never differentiates into base features.

The local native bridge still requires imports from the pinned repository root
and official runtime. The library has **no CLI, PBS submission, model loader,
physics engine, optimizer or automatic online-learning loop**. Optional tensor
components require the existing Torch environment; no dependency was installed.

## Data And Safety Semantics

The official GR00T trainer defaults to per-block `intervention_probability=1.0`,
not the generic Handler's beta-decay/disagreement logic. Its expert branch
executes 16 live feedback actions, while the student replans at prefix8.
Do not import either behavior as our 0.4s/H40 semantics.

Our arbitration takes an explicit intervention decision; a calibrated beta or
XYZ/SO3/gripper disagreement policy is **NOT_IMPLEMENTED**. No mixed-unit
4-dimensional L2 threshold is copied. Both candidates unsafe -> STOP;
invalid/unavailable expert -> no expert label. S1 stays privileged reject-only,
without clipping, replacement actions or resampling.

An expert label may be unexecuted and still useful for correction learning, but
is never called an actually committed successful demonstration. The current
`NativeDataset` requires `ABSOLUTE_COMMITTED_EEF10`; do not put counterfactual
expert labels in that old export field. The new pool records a separate
`COUNTERFACTUAL_EXPERT_CORRECTION_EEF10` identity and needs future audited
training ingestion, not an implicit reuse of the success-BC loader.

"Student visited" requires a previous actual student-prefix ACK bound to this
observation/tick. Reset/expert-only trajectories cannot acquire that label from
a boolean alone. Safety and execution receipt hashes remain references to
private originals; the library does not authenticate their external origin.
The future collector must verify those originals through the existing relay.

Teacher receives privileged task state separately. Student/steering receives
only original actual RGB, language and observable EEF/gripper state. Native
context rejects extra Teacher/object fields before backbone/processor work.
Expert H40 must be valid at this same state for the entire horizon; eight
known actions plus a fabricated tail, or 40 future live-feedback actions
misrepresented as one current open-loop plan, are not valid inversion targets.

## Future Qualification, Not Authorized Here

1. FD0: inspect actual E0 outcomes and safe states after actual student prefixes.
   E0 is still queued at this preparation snapshot. Zero student advancement
   means no student-visited states, not a synthetic DAgger pool. Use original
   TRAIN lineage for optimization; E0 DEV can remain diagnostic/evaluation only.
2. FD1: new bounded source-D0 Recovery Teacher qualification on off-nominal
   student states. Existing reset Teacher/S1 qualification is not this proof.
   Report valid/unrecoverable corrections and native safety without WBC tuning.
3. FD2: under a separately adopted Hopper allocation, first prove custom-latent
   forward parity with the original sampler, then expert action round trip.
   Keep original processor/numerical mode. Record full/prefix8 XYZ, SO3,
   gripper error/events, original reference gate and S1 outcomes, and costs.
4. FD3: matched model, expert queries, correction pool, training/compute and
   independent evaluation for M2048, added-demo BC, action-space DAgger and
   frozen-base FlowDAgger. Hyperparameters/thresholds and budgets stay UNSET
   until an actual task card is adopted. No best-checkpoint selection from DEV.
5. FD4: fullbody transfer only after independently qualified Teacher/data and
   actual BASE labels; latent steering does not create untrained BASE3 skills.

With four steps and fp_per_step10, inverse work is44 velocity evaluations;
one forward reconstruction adds4, total48, plus observation features. This is
12 times the normal four-step velocity-call count, **not a measured 12x wall
cost**. Padding extension targets zero inactive channels, matching the upstream
model-space strategy; these channels are not observed action labels. Evaluate
active physical errors and expressibility, not just full-padded latent MSE.

Real processor round-trip, M2048 sampler parity, expert recoverability,
inversion quality and steering task performance are all **NOT_RUN**. Failure
must remain in the denominator; no thresholds or recovery success claims are
invented by the passing synthetic tests. selected/deployment remain NONE.

Every future collector/evaluator/trainer must bind the existing standard
RUN_MANIFEST, fixed source/inputs, role/profile, active lease, new explicit GO
and separate budget before dispatch. CoreLab owns physics/Teacher; Hopper owns
all GR00T neural work only inside legal PBS. E0 resources are not repurposed.
