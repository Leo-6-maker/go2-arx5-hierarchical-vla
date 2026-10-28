# FD2 Dynamic Expert Clearance Pilot V1

Authority: the human's 2026-10-19 SGT `允许推进`, directly following their
review of FD1 ba40c713b483a02be5387b5254d045174dc6493b and the existing owner's
bounded eight-episode pilot proposal. This adopts that one package, not MP1,
N1, FD1 balances, another experiment or a later training GO.

Status: AUTHORIZED_CONDITIONAL_ON_FRESH_NATIVE_BINDING. Git publication alone
is not an execution receipt. One existing local-PC owner, no second owner.

## Question And Design

Measure whether the ordinary, unchanged source-domain T2 Teacher can safely
complete pushing on new TRAIN-only geometry, and whether changing only its
nominal pad-table reference clearance from 0.004 to 0.012 m changes dynamic
clearance and task feasibility. Higher is not presumed better: it may lose
object contact. This does not establish a neural training cause or benefit.

Four fixed new groups: +Y twice and -Y twice, fixed appearance. Exact cards
come from `scripts/pi0294/fd2.py:pilot_plan`; no random candidate search.
All old actual scene cards plus the legacy exclusion population are checked
with the existing 2mm position / 2degree yaw near-duplicate rule, ignoring
appearance/family. Any conflict blocks rather than replacing the layout.
No CONFIRM40 states, variants or appearance-only siblings become new TRAIN.

Eight cells, one first attempt each. Each pair has identical scene XML,
qpos/qvel, actual initial TCP and initial camera pixels. Order is fixed and
counterbalanced: pair0 4/12, pair1 12/4, pair2 4/12, pair3 12/4 mm. No seed,
clearance or successful-cell selection; engineering partials remain spent.

The adapter invokes the original PushTeacher, adding exactly 0.008 m only
to source/goal reference Z in the 12mm arm. The 4mm arm adds nothing. All
other targets, stage rules, feedback, H40/prefix8 and gripper commands stay
original. Static qualification uses the corresponding same reference arm.
No change to robot, WBC, legs, weights, gains, solver, plant, friction,
gripper PD, S1, physical safety, scorer, observation schema or task timing.
This is FIXED_BASE_ARX5_TASK_SIM, not Go2/WBC or real-robot qualification.

## Execution And Evidence

New manifest, registry role, output, lease, source capsule and CPU ledger;
fresh approved `4080-simulation` entry, own UID/host, original 26 asset
hashes, protected numerical hashes and N1/FD1 closure checks before dispatch.
No mutable running source update. Old safety STOPs and raw evidence remain.

Use original finite 32s Teacher episode, actual two-camera Mesa CPU frames,
500Hz dynamics and original S1 prediction/copy/bind/1e-10 parity. Count every
live and shadow tick, engine constructor and clone, including failures and
negative-control clones. Rejected prefixes have zero live execution.
All eight cells remain in the planned denominator if a hard stop intervenes.
No replacement, extra smoke rollout, static seed search, recovery or retry.
Routine same-semantics engineering fixes apply only to unattempted cells
after exact terminal/no-effect proof and a new cold source identity.

Passive measurements reuse existing snapshot returns, adding no forward,
snapshot or integration call. Record actual pad vertical box gap, finite
pad/table signed geom distances, force/contact, stage, object pose, actual
TCP versus active reference XYZ/SO3/gripper errors and actual 500Hz timing.
Measurements must leave the original S1 fingerprint unchanged. Distinguish
live history, predicted rejected shadow and nominal reference throughout.

Saved-state independent audit checks all raw member hashes, all image
hashes/counts, actual labels, full highrate coverage, S1 parity/zero-live
rejections, exact original score plus independently implemented push
predicates. Generate paired-camera video from actual frames, no padding,
interpolation or action replay. Retain failures and partial evidence.

Report per cell: actual initial identity, static result, duration, Teacher
events, minimum dynamic gap/distance, contact, push displacement, highrate
no-lift, separation/withdrawal and original terminal41 subpredicates, score,
S1/native failures, audit and video index. Aggregate four pairs, not eight
independent layouts. Four layouts cannot establish population reliability.

## Finite New Resource Window

The machine card is `reports/PI0294/design/FD2_DYNAMIC_EXPERT_CLEARANCE_SCOPE_V1.json`.
Register T0 once, six hours inclusive of preparation and closeout from actual
adoption; stop new cells at T0+5h, hard close at T0+6h. Account prebinding
engineering explicitly inside this package, never against a closed budget.

CoreLab <=8 CPU core-hours; local-PC <=2 CPU core-hours; at most2 cores/site.
Eight Teacher task episodes, all live+shadow <=300000 physical ticks,
<=700 engine starts including clones, <=8GiB scratch, <=2GiB transfer.
GPU/PBS/neural generation/optimizer updates/new training datasets=0.
No Hopper/Vanda compute, WBC execution or hardware. No deletes/cleanup.

Global stop on actual native safety fault, S1 integrity/parity failure,
unknown side effects, stale/wrong authority/source/lease, budget/deadline or
input corruption. Ordinary original-S1 rejection and safe task failure are
results, not a reason to change gates or rerun. Save terminal state before
closeout; no unbounded daemon or automatic continuation.

## Decision

Report a feasibility table, not a posthoc optimum. A larger margin that is
safe but cannot contact/push/withdraw/hold is not a qualified demonstration
generator. If original4mm is dynamically robust on all four fresh groups,
expert fragility is weakened here; student-state/action error remains open.
No result automatically authorizes dataset expansion, BC, DAgger, new model
selection or another evaluation population. T1 release and T2 no-lift/hold
are distinct failures. selected/deployment=NONE, TF-R3 STOP preserved.
