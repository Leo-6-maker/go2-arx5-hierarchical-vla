# PI0256 P0: frozen preparation comparison

Authority: `fcfd37bf2ced86ae3c3ce9ee43c2a0b9ec810450`, Issue #11 canonical
5887791858. This packet precedes new physics. PI0254 U remains quality FAIL,
R remains zero-update integrity BLOCKED, and original PI0255 remains 36/36
first-valid preparation FAIL, with no task-phase or GR00T evidence.

## Readback and old evidence

`runtime_readback.json` records fresh complete-file hashes of the actual four
node-local reference-317a helpers and nine native components. All match the
PI0255 archived source receipt. Repository names were not used as runtime proof.
`legacy_six_family_timelines.json` verifies original raw hashes and indexes
construction, binds, actor outputs, clean prefixes, gates and terminal events.
Original actor-input/history bytes remain unavailable; they are not rebuilt.

Six representatives comprise gate failures for families 000, 001 and 004;
native termination for 002, 003 and 005. Recorded pre-autoreset terminal events
identify trajectory timing aborts, rather than a clean fall-pose measurement.
Post-step poses after these autoresets do not describe the terminating pose.
The implicit reset step ends at approximately 0.02 seconds and is counted.

Actual source establishes three differences worth testing as a PACKAGE:

1. The legacy constructor loads nominal scene coordinates with identity
   orientation and adds the native reach-table translation. The subsequent
   bind uses the cached measured pose, under that same anchor.
2. Actual `_stage21_load_chunk` replaces canonical trajectory and goal fields
   without calling `_update_ee_task_space_error`. Actual arm observation reads
   those cached goal-error/body-target fields as well as a fresh trajectory
   preview. This establishes a possible mixed-reference observation; it does
   not establish a unique historical cause or reconstruct its missing bytes.
3. Legacy preparation repeatedly loads a 5 mm curve and resets local reference
   progress. Native timing termination scales with trajectory length. The
   observations do not establish support for that hold/preparation primitive.

Native cached TCP versus independent URDF/root/DOF FK is recorded prospectively.
No unproved TCP alias or frame hypothesis is repaired by modifying native WBC.

## Single prospective package

The source is PI0252 `pre_full_sim/controls/A_attempt2`'s actual complete C0
reference: 97 finite physical ticks, reference complete, no autoreset, and its
original task-specific PASS. That run alone does NOT meet or prove the new
0.5-second preparation predicate. It includes previously recorded soft joint
margin violations and cannot serve as retrospective preparation qualification.

B/C rigidly transport this complete, nondegenerate numerical curve ONCE to the
reset-time TCP from the actual URDF, root pose and arm DOFs, preserving the
measured entry orientation. Positions rotate by R_entry R_source_first^T and
translate by p_entry-p_source_first after rotation. The source's constant
orientation is transported to the entry orientation. The original 0..1.95 s
curve is sampled every 0.05 s and its terminal pose continues until 5 seconds.
This is a continuous native expert preparation asset (101 points), not a
change to the future policy's H40 ABI. Actual generic trajectory contract,
original gate and native canonical implementation accept this asset on CPU.
No minimum-arc workaround, repeated reanchor or idle actuator is introduced.

Construction uses one immutable world-aligned translation anchor. After the
implicit reset step, the existing derived goal-error cache is refreshed once
before the first actor, with root/DOF/clock/episode/reset and history invariants.
There is no extra observation getter, history clearing, hidden step or physics
write. Actual arm -> body plan -> dog -> native step ordering is reused.
Native source, weights, gains, reset distribution, dynamics and gate stay fixed.

`PREPARATION_V2_CONTRACT.json` seals numerical formula, source/template hashes,
all thresholds and all 18 IDs. B receives the six old conditions; C receives
six fresh disjoint conditions and is conditional on B 6/6. A retains original
preparation behavior, including its absolute-clock deadline, with new recording.
All groups additionally report the new predicate. Matched pre-intervention
root/DOF/RNG/scene identities are checked independently of altered references.
Post-intervention histories and references are explicitly package-dependent.

The frozen joint predicate is the original native telemetry violation over
soft bounds. Actual hard URDF margins are separate diagnostics; results cannot
be improved by switching bounds after observation. The five-second deadline
counts the implicit reset step. Qualification needs every fixed condition for
0.5 continuous seconds; task time begins only after qualification.

## Tests, recording and resource envelope

CPU negatives cover stale native goal-error cache, wrong quaternion order,
double anchor, elapsed clock offset, immutable source reference, physical-state
mutation, nonfinite values, angular motion despite small XYZ speed, and distinct
pre-/post-autoreset poses. The cache test compiles the actual native function
for a position-cache fixture; it is not a controller rollout or full history
parity claim. Physical instrumentation retains actual current observations and
histories, full arrays, actor outputs, raw/canonical reference loads, pre/post
step state and the existing read-only pre-autoreset hook from the first attempt.

Native-import crashes occurred during no-physics CPU preflight. The executable
now imports the native binary before scientific libraries; its native CPU path
passed. No model, physics start or optimizer update was spent on these imports.
The failed import logs remain engineering evidence, not scientific outcomes.

Bounded finite runner only: 24 preparation / 384 total native starts and 480000
actual control-step caps are checked before execution, including implicit steps.
Each preparation case has a 128 MiB evidence guard (planned 18 <=2.25 GiB).
Current live GPU0 has about 25 GiB free; other workloads remain untouched.
Shared filesystem free is about 381 GiB, node-local about 949 GiB; user quota
is UNKNOWN, not a quota PASS. No large model/dataset write is released at P0/P1.
No Hopper jobs, new model queries, task pilots, mail or monitor are started.

B and C each require 6/6. A/B measure the package effect, not a unique internal
cause. Valid scientific failure stops that dependency; the six reserved checks
require a source-proven implementation correction, not a different pose/profile.
All original and new failures remain in their namespaces. Both selected and
deployment checkpoints remain NONE.
