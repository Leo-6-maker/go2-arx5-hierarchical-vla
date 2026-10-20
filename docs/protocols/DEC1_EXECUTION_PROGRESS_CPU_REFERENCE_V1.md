# DEC1 Execution Progress CPU Reference V1

Status: **CPU_IMPLEMENTED_NATIVE_COMMISSIONING_REQUIRED**. Authority remains
`a7a7ecd9a6be62a7221af25a5e32a0009b076a84`; this document adds no budget,
owner, clock, task authorization or relaxation of native safety.

## Shared Reference And Intervention

`ExecutionProgressAdapter` is a simulator-independent, task-blind reference
generator. Both modes use the same frame-corrected `GR00TToWBCAdapterV2`,
native row-packed EEF10 conversion boundary, quaternion SLERP, nine spatial
preview offsets and SE3 metric. Historical V1 bytes are unchanged. The only
between-mode intervention is phase estimation:

- I_TIME interpolates the original absolute sample clock.
- I_PROGRESS projects actual TCP pose onto a bounded forward-only local SE3
  window. Initial search is at most eight waypoint intervals; subsequent
  search is at most two. Gripper residual disambiguates zero-SE3 segments.

Phase is a waypoint index, not normalized progress copied from the old chunk.
Each new chunk is reanchored to fresh feedback, with unique chunk identity.
No object state, Teacher stage, language, task ID, contact or score enters the
estimator. Ambiguous self-intersections are not generally identifiable: the
local search conservatively chooses the earliest equal-distance match rather
than a global nearest-point jump. This is not a learned progress predictor.

## Stationary And Gripper Rules

Repeated SE3 waypoints retain their original local duration while actual TCP
is within the fixed source-only tolerances. At a stationary segment, elapsed
feedback time permits bounded advancement. This clock rule is not proof of
contact, grasp or release. Pure rotation uses the same SE3 metric and stable
quaternion-sign-invariant geodesic.

Spatial-only legacy preview can look through a finite repeated-pose dwell.
The new prospective I_TIME and I_PROGRESS both stop spatial lookahead at the
next not-yet-elapsed stationary segment. This shared original-pose duration
interpretation is **not an I_PROGRESS-only improvement**. It must be included
in source commissioning and disclosed in comparisons. Old B0 results are not
substitutes for the new paired I_TIME cells; standalone historical V1/V2 and
their old results are not rewritten.

The gripper amplitude is the original predicted sample at floor(phase).
Progress cannot skip multiple predicted transitions in one issued command;
it stops before the second change. A late new chunk or I_TIME clock gap that
would skip an unissued transition is rejected, not repaired with a task
script. No tolerance is used to erase small model-generated changes.

## Safety And Actual Feedback

The complete original H40 is checked before frame or phase mapping. The
PI0294 raw wire helper explicitly checks the complete original row-packed
EEF10 before converting rotation rows to a quaternion; it must not use the
canonical column-packed `from_rot6d` path for native rows.

All three adapter safety callbacks are mandatory. Each must raise on reject
and return None only on acceptance. False or uninterpreted decision objects
are not silent acceptance. Native bindings must explicitly translate their
unchanged native S1 decision into this contract, preserving original checks,
accounting and diagnostic evidence. Synthetic callbacks do not qualify S1.

Every proposal includes the immediate nine-point preview plus forty mapped
virtual previews and the mapped full H40 (each preview's first point, with
original 20Hz and predicted gripper sample order). These are forecasts for
full-reference and Prefix8 legality, not forty real observations or actions.
Native full and prefix validators must accept before a proposal can be
issued. Original raw rejection cannot be bypassed by a safe mapped prefix.

Progress commits only after a matching execution ACK with a positive bounded
execution interval. The dispatched preview, full forecast, gripper and clocks
are covered by a mutation digest. A known identical ACK readback is idempotent;
unknown/mismatching ACKs stop the adapter. The next actual observation must
follow the previous execution end, name that predecessor ACK, and pass age,
frame and TCP checks. A new model chunk must itself use a post-action
observation. Stale/NaN/frame/identity/reject/unknown errors latch a stop; the
native caller must actually stop the executor and retain unknown-effect
evidence. This library has no reset or transport replay function.

## Qualification Boundary

The accompanying JSON fixes CPU algorithm scalars without target-result
tuning. It is **not a native-qualified source-only interface freeze**:
executor-specific frame/TCP, the original stale rule, native raw/full/prefix
safety bindings, shadow-state cloning, dispatch/ACK transport and source
commissioning remain required. Do not install placeholder frame names or
permissive callbacks. Shadow checks must stay within the original tick budget
and cannot replace target safety with a source-specific model.

No new GPU/PBS/model load/query/update, physics, episode, WBC execution or
LOCKBOX was used to develop these CPU fixtures. Unit equivalence and synthetic
lag cases are not evidence of task improvement or cross-executor transfer.
The 64-cell panel is not sealed or run. Target task ABI remains undelivered;
source training cold binding and legal GPU admission are separate unfinished
work, not blocked by the independent completed CR2 bank.
