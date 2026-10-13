# PI0291 Fixed-Base ARX5 Teacher Data

Status: `PARTIAL`. The original queue is terminal:56 TRAIN attempted and
owner-admitted;8 TRAIN and16 DEV not attempted. All56 independent raw audits
passed. Frozen export contains33,656 TRAIN H40 windows and0 DEV windows.
These are overlapping windows, not independent trials. No complete-bank or
first-training-ready claim is made. See terminal `DATASET_ADMISSION.json`.

## Scope

Real native ARX5 six-joint geometry and finite joint torque actuation in
MuJoCo 3.2.7, attached to a fixed Go2 base. Two independently actuated opposed
prismatic fingers make actual simulated contacts with a 4 cm red cube.
Contact pad geometry and friction are disclosed SIM_ONLY assumptions, not
hardware calibration. Domain: `FIXED_BASE_ARX5_TASK_SIM`.

The sole task is grasp, lift, carry, release, withdraw and hold the cube in a
16 cm target region. Formal source/goal jitter is bounded to +/-8 mm and yaw
to +/-0.1 rad. This narrow, single-task dataset does not establish task,
language, object, mobile-base, frozen-WBC or real-robot generalization.

## Observations And Labels

Each complete 32 s episode has 641 actual, synchronized pairs of body/wrist
256x256 RGB and actual TCP/gripper feedback at 20 Hz. The body camera is a
fixed world rig; the wrist camera follows the arm. There are 640 committed
absolute world EEF10 commands and 601 complete H40 windows. The final
observation has no command. Commands are actual installed teacher references,
not achieved future states, and are converted relative to the anchor's
actual TCP exactly once by the fixed official processor.

Student input is restricted to actual RGB, original language and actual
EEF/gripper. Object state, contact forces, joint state, teacher stage and
scoring are privileged evidence and are excluded from the policy input.

## Admission And Splits

All 64 TRAIN and 16 DEV IDs, seeds and layout groups were registered before
the first pilot. IDs are attempted once in TRAIN-then-DEV order, with no
replacement after failure. Unattempted IDs remain in the planned denominator.
Commissioning, pilot and PI0290 legacy episodes are diagnostics only and
cannot enter formal BC or TRAIN statistics. Only complete successful formal
episodes with independent raw audit and valid data may be admitted.

Split is by episode, layout group and seed before window construction. The
601 overlapping windows are not 601 independent trials. DEV qualifies this
data version and is not a blind final model TEST. Original T1 instruction is
unchanged; variable-text CPU collator diagnostics are not extra language data.

The exact original policy instruction is "Pick up the red cube, place it
inside the blue square, release it, and move the gripper away." The frozen
export's shorter `task_map.json` T1 text is a descriptive metadata label only.
The actual loader takes the verbatim NPZ instruction from original raw
observations, not that label. A future Trainer adapter must preserve this
distinction. Authoritative uint64 seeds are compared losslessly in Python;
sanitized report seeds use decimal strings, not JavaScript floating numbers.

## Statistics And Interface

Frozen rule: `TRAIN_ONLY_ALL_FULL_H40_EQUAL_WINDOW_TIMESTEP_WEIGHT_MINMAX_V1`.
All admitted TRAIN state anchors are counted once; every H40 window action
timestep has equal weight, so interior committed commands intentionally
repeat. DEV, pilot, failures and legacy data never fit statistics. Percentile
normalization and input outlier clipping are disabled; stock inverse clipping
is measured, with no DEV refit or processor monkeypatch.

Official source is `51d4c89f72fda44cbf77285c6a8114b52676b8a1`.
Actual formal CPU DatasetLoader/processor/collator and all-window codec
passed:all33,656 TRAIN H40 windows, real dual RGB, mixed two-episode/stage
batches, variable-text diagnostic, deterministic0/2-worker equality and
five oracle/causality negatives. Actual CPU618.029364s including children.
Normalized float32 official decode maxima:XYZ8.5735081e-9m,
SO3 2.1073424e-8rad, gripper0. Pre-float32 action preclip35 components;
stock float32 inverse clip0, state preclip0, encoder clipping0. DEV has
zero windows and is NOT_TESTED, not a zero-error validation success.
The immutable exporter manifest retains its at-export NOT_RUN field;
the actual final qualification authority is `evidence/DATALOADER_SMOKE.json`.
Fixed-revision HF readback is pending at upload time. The existing
official training launcher does not directly consume this NPZ+PNG wrapper:
see `TRAINER_INTERFACE_READONLY_REVIEW.json`. CPU data qualification must not
be misrepresented as completed Trainer integration.

## Provenance And Access

Physical source: `57a02b4923e5001df5fd4b3b5958c372b9cff541`.
Run: `20261013T023717-localpc-arx5`.
Separate CPU-only closeout window:2026-10-13T01:12:42Z to02:12:42Z,
new CPU allowance14400s, original total output cap40GiB. The original
physical clock/counters are not reset and no missing DEV is collected.
Seal SHA256: `f4919dd5b4ad4f978f182cb0663e6cbdb3b50be34a84d19f56f2d7aa98398f9d`.
The authoritative immutable private HF repository, prefix, revisions and blob
hashes will be recorded in the actual transfer/readback receipts, not inferred
from a repository name or mutable HEAD.

No private WBC source, robot meshes, neural weights, tokenizer cache or
credentials are included in this data handoff. Physics replay depends on
the separately held private native robot asset identities; asset-free replay
and third-party mesh redistribution permission are not claimed. Do not
republish original assets without a separate rights review.

GR00T model load, forward, training, new Hopper PBS, original WBC execution
and hardware actions are all zero. `selected=NONE; deployment=NONE`.
