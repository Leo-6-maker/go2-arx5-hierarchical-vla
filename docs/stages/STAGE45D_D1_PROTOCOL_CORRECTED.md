# Stage45D D1 corrected teacher-forced/local-flow protocol

Status: **CORRECTED_FOR_PI_REVIEW_NOT_EXECUTED**

This is a correction layer over the D0 D1 draft.  The original D0 artifacts
and their hashes remain unchanged.  The machine-readable contract is
`manifests/STAGE45D_D1_PROTOCOL_CORRECTED_V1.json`; the CPU-only verifier is
`scripts/stage45d_d1_protocol_static.py`.

Authority is PI_GATE_STATE_0104 (Issue #11 comment `5617025466`) with the
Issue #50 execution pointer `5617028048`.  Only protocol/static hardening is
authorized.  No model, checkpoint, GPU, scheduler, training, Validation-E,
Holdout-F, WBC, simulator, or hardware operation is part of this pass.

## Corrections

### StructuredActionV1 supervision mask

The protocol is bound directly to
`go2arx5_vla.structured_action.structured_action_mask("T16")` with shape
`[40,132]` and exactly 190 active scalar entries:

- geometry: 15 future controls × 9 scalars = 135;
- timing: 15 interval-logit scalars = 15;
- gripper: 40 dense H40 scalars = 40;
- every remaining entry is zero.

The verifier compares the declared contract with the repository implementation
and includes a negative test that rejects a dense first-ten-channel mask.  The
old Stage26 mask topology is not used or described as the D1 supervision law.

### Metric namespaces

D1 keeps two non-interchangeable namespaces.

1. **NORMALIZED_LOCAL_FLOW** is the training-objective space.  It reports
   masked local-flow RMSE/cosine and separate geometry, rotation, timing, and
   gripper blocks.  A block-macro summary is retained so scalar block size
   cannot silently determine the interpretation.
2. **DECODED_PHYSICAL_ACTION** is the interpretation space.  Where valid, the
   implied action is inverse-normalized with the bound checkpoint statistics
   and decoded through the frozen StructuredActionV1 path.  EEF, timing, and
   gripper diagnostics remain separate.  A normalized error is never promoted
   to a physical-accuracy claim, and no WBC or task-success quantity is
   fabricated.

The canonical Stage44R decode and trajectory metric semantics are preserved.

### Hypothesis and taxonomy language

`H_OPT=SUPPORTED` is deliberately narrow: continued optimization along the
existing 1000→2000 schedule improves sealed TRAIN24 behavior.  It does not
claim convergence, pure exposure causality, or monotonic improvement after
2000.  `H_SAMPLER=NOT_TESTED` and `H_POLICY=NOT_TESTED`; D0 had no direct
current-checkpoint teacher-forced local-field evidence.  The prior data claim
is split into descriptive distribution coverage (`H_DATA_DISTRIBUTION`) and a
merely plausible, untested normalization effect (`H_NORMALIZATION_EFFECT`).

Taxonomy fields use descriptive progress signs and coverage/exposure delta
labels.  They do not encode task success, causal help/hurt, or general
robustness.

The causal boundary is also explicit: teacher-forced local-field good plus
free-running bad places the dominant failure in the rollout/off-manifold/
sampler-integration class.  It does not prove Euler integration is the cause;
that requires a later D2 gate.

## Frozen execution plan

The future D1 plan retains the existing TRAIN24 cohort, fixed seeds
`2223,2224,2225`, fixed flow times and fixed CPU noise reuse.  Its estimate is
648 action-head calls plus 72 conditioning passes (720 direct model-side
calls), but this branch records **zero** construction, forward, GPU, PBS,
training, Validation-E, Holdout-F, WBC, simulator, and hardware exposure.

The corrected protocol remains held for PI review:

`STAGE45D_D1_PROTOCOL_CORRECTED_READY_FOR_PI_REVIEW`
