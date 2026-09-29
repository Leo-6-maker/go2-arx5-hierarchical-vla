# PI0256 — Preparation repair to a real multitask EEF generation pilot

Authorization: **PI0256_PREPARATION_REPAIR_TO_MULTITASK_PILOT_GO_V1**

Issued 2026-09-29 under the user's request to review the PI0254/PI0255 closeout,
issue the next scientific goal, and continue multitask EEF generation.

This is a NEW, conditional execution authorization, not a reopening of PI0254
or a retry of the original PI0255 failures. It authorizes the complete chain:
source-backed preparation repair -> bounded expert data -> real processor and
resume qualification -> matched MT-Base/MT-G training -> fixed offline evaluation.
Passing the explicit gates below releases the next step without routine PI
reapproval. A material scientific change outside this document still requires
an amendment. Publication of this goal is not evidence that execution occurred.

## 0. Authority, source identity, and facts that must remain unchanged

Repository: `Leo-6-maker/go2-arx5-hierarchical-vla`.
Scientific ledger: Issue #11. PR #77 receives pointers only; no merge.
Source base: `757706bc3e93f91496927a000d3d3c49c41364a1`.
Branch: `pi0256/preparation-repair-to-multitask-pilot-v1`.

Read these immutable sources before implementation:

- Issue #11 closeout comment `5887330960` (2026-09-29 09:20:50 UTC).
- PI0254 `b3e2bd41ed42e85c09c400494b6d4ae75dbf3e67`,
  `reports/PI0254/U_FINAL/REPORT.md` and its raw evidence index.
- At the source base: `reports/PI0255/COMPLETION_ADMISSION_AUDIT.md`,
  `PILOT_RESULT.md`, `PILOT/pilot_stage_summary.json`, `CAPABILITY_MATRIX.md`,
  `INITIALIZER_COMPARISON.md`, `FUTURE_EVALUATION_DRAFT.md`, and model drafts.
- At that base: `scripts/pi0255_collect.py`, `pi0255_motion3.py`,
  `pi0255_pilot_draft.py`, and their actual pinned runtime dependencies.
- Original PI0255 authority `eac535e7c4016df39821798b111fae7ad79e8253`,
  canonical `5884624474`, and durable negative-evidence canonical `5886399186`.

### Accepted closeout interpretation

U4608 is a valid quality FAIL under the original AND endpoint. E64 relative
support improvement is not absolute task qualification; P64 is previously
exposed TRAIN/development support, not a blind generalization test. O64 and
old16 fail retained XYZ requirements. The two O64 speed rejects remain real.

R has zero forward/optimizer updates in its final blocked attempt. There is
no completed equal-budget rehearsal comparison and no evidence that rehearsal
itself failed. Do not restart R, transfer its unused budget, select U, or claim
a unique cause of its processed-input mismatch from the RNG audit alone.

PI0255 has 36 first-valid end-to-end preparation FAILs, 18 gate and 18 physical,
across six reset families. Every task has 0/12 under the original protocol,
but task-phase execution, successful demonstrations and eligible windows are
all zero. These outcomes do not measure task-phase feasibility or VLA ability.
Same-reset task/command siblings are not independent reliability replications.

The original 24 episodes lack full handoff arrays; only the later 12 have the
complete recorded arrays for gate replay. Never reconstruct missing bytes or
convert incomplete historical evidence into a full-parity PASS. Terminal
post-step observations may be post-autoreset; they are not clean failure poses.

PI0255 raw evidence is fixed at
`leoyuer/pi0246-gate-pass-campaign@c6bc2a041323788a4b19b7b8ef11227f9950a0a6`,
prefix `evidence/PI0255/fixed-pilot-audit-v1-20260929` (789 members).
PI0254 final raw evidence is fixed at revision
`9aec61d5e4f5b9858059bcb3b45ac8956c448f71` of that repository.
Reuse these receipts; do not archive or relabel old runs as new experiments.

## 1. Mainline decision and explicit amendments

The objective is a shared language-conditioned EEF generator, not endless
repair of single-task O64 fit. PI0254 PASS, old-G physical success and perfect
O64 retention are NOT prerequisites for this new multitask study.

This GO explicitly authorizes five prospective changes:

1. A versioned, source-coherent external preparation/reference lifecycle,
   with measured entry state and separate preparation/task clocks.
2. M3-v2 directed OPEN-contour traversal, replacing the unobserved single-lap
   stopping obligation of M3-v1. M1 and M2 retain their registered semantics.
3. A deterministic, view-preserving common processor without stochastic crop,
   horizontal flip or color jitter, and a TRAIN-only affine C_MT binding.
4. Conditional real Base/G pilot training, previously only a PI0255 draft.
5. A minimum of TWO qualified behavior families for that shared-model pilot.
   All three remain the preferred target; unsupported families stay visible.

The complete admitted set is determined mechanically after the fixed pilots,
collection and data QA finish: include EVERY family satisfying the gates.
K=3 gives MT3; K=2 gives MT2 with the exact task names in every result. K<2
blocks training. Do not choose an easier pair based on model scores, run an
MT1 fallback, call MT2 Motion3 READY, or erase an excluded family's failures.

The external ABI stays two RGB streams + measured StateV1 + natural language
-> H40 XYZ/row-packed rot6d/parallel gripper, using the same frozen WBC.
No privileged task ID, target XYZ, expert future, clock or hidden phase enters
the policy. No gripper/contact manipulation capability is presumed.

## 2. P0 — Narrow causal audit and executable preparation contract

Use an isolated worktree/runtime/root. Refresh actual jobs, GPU occupancy and
storage before use. A GitHub status snapshot is not live SSH evidence. Match
the node-local `PI0250/runtime/reference-317a` helper hashes against the
collector's archived source readback; repository helper names alone do not
establish runtime parity. No library/framework/model upgrade is authorized.

Inspect the existing six-family traces before new physics. Produce one causal
timeline per family: reset, implicit first physics, reference construction,
first bind, actor observation/history read, arm/body/dog action, subsequent
reference update, gate reject or native termination.

The following are hypotheses to discriminate, NOT asserted root causes:

- construction reference at nominal scene coordinates with identity rotation
  versus subsequent reference bound to the measured reset pose;
- frame/TCP/quaternion conversion or canonical trajectory mismatch;
- stale reference-dependent observations/history across the first bind or
  later handoffs; inspection of the actual helper is required;
- a tiny nonzero reference not representing a supported native hold/preparation
  primitive, or the required pose lying outside the controller's useful support;
- real native fall/joint/contact termination versus post-autoreset telemetry.

The archived example fails at approximately 0.22 s with rotation discontinuity
1.576489 rad despite a constant registered desired orientation. Its clean
prefix reports about 0.112 m XYZ RMS and 1.252 rad SO3 RMS. A threshold change
from 1.5 to a slightly larger number would not establish controlled settling.

### Preparation-v2 implementation scope

Implement `RESET -> PREPARE -> SETTLED -> TASK -> TERMINAL` with explicit
actual simulation timestamps. The preparation deadline is five seconds of
ELAPSED preparation time, not `absolute_sim_clock < 5`. Retain and count any
implicit reset step. Task time starts only after preparation qualifies.

A single source-supported preparation reference/primitive must be resolved
and sealed before new physics. Prefer an existing physically evidenced native
expert entry procedure, not a constructor-only camera/hold smoke. A nominal
FK pose or a passing trajectory gate alone is not a physical positive control.
Record exactly which evidence supports the chosen entry pose/reference.

Changes are limited to project-owned preparation reference construction,
initialization order, reference-dependent observation synchronization, and
non-invasive recording. A proposed fix must state its invariant and CPU test.
Preserve native action ordering, observation semantics, controller history
semantics after initialization, weights, gains, robot dynamics and reset-state
distribution. Rebuilding a current observation must not clear learned history
at each handoff or advance hidden physics. Do not invent an idle controller.

Never continually reanchor the desired trajectory to the drifting achieved
pose to hide tracking error. Never use a 5 mm motion merely to circumvent a
minimum-arc rejection. A zero-motion/hold path requires demonstrated support
in the existing native interface; otherwise report that specific capability
boundary. No gate bypass, teleport-to-settled pose, state transplantation,
external stabilizing force, dynamics/gain tuning, or reset after a task failure.

### Frozen preparation success predicate

For a continuous 0.5-second interval within that five-second deadline require:
finite fresh state; no native termination/reset/fall/contact failure; no joint
limit violation; EEF speed <=0.02 m/s; EEF angular speed <=0.20 rad/s; and
same-tick achieved/reference error <=0.035 m and <=0.10 rad. Retain native
whole-body safety predicates and record base pose/velocity and joint margins.
These are NEW preparation-study criteria, not retrospective PI0255 thresholds.
Do not relax them after observations. A gate PASS alone is insufficient.

### Required actual traces and CPU negatives

Record complete raw reference arrays, timestamps, canonical/native reference,
measured pose, immutable full SE(3), actor input/history hashes and small
reconstructable slices, arm/body/dog outputs, pre/post-step state, all native
termination flags and reference-load events from the FIRST attempt onward.
Record world-aligned translation-only native framing only when actual source
proves it; unit tests must still cover a nonzero rotation and translation.

Add a read-only hook before native autoreset to preserve the terminating state
and flags. Keep normal reset behavior intact. When that hook is unavailable,
mark terminal pose unavailable, preserve last clean state, and do not infer
pre-reset tracking from a reset image. Missing decisive evidence blocks causal
claims, not the preservation of the observed FAIL.

CPU tests must cover stale observation, wrong quaternion convention, double
anchor, elapsed-clock offset, reference/physics mutation, nonfinite state,
angular motion with small XYZ speed, and pre-/post-reset record separation.
Commit source, exact numeric reference, hashes, predicates and case registry
before any authorized new physical attempt. This materialization needs no
additional routine GO.

## 3. P1 — Small shared-preparation experiment before task pilots

New diagnostic IDs are separate from PI0255 episode IDs. The original failures
remain immutable. Reusing their reset conditions here is an explicit new
mechanism comparison, not a retry used to increase old success counts.

First register 18 preparation-only attempts:

- A: six old reset families, original preparation behavior, complete new
  instrumentation. Reproduce the legacy behavior, not its missing records.
- B: the same six conditions with the single sealed preparation-v2 package.
- C: six fresh, disjoint preparation-confirmation conditions using B, executed
  only if all six B conditions qualify.

Pair intervention-independent initial state/RNG/scene identities. When a
preparation change affects the post-reset reference/history, disclose that as
part of the intervention; do not falsely claim those post-intervention bytes
are identical. A versus B identifies the PACKAGE effect, not necessarily one
unique internal cause. No task instruction, task trajectory or VLA is executed.

B must qualify 6/6 and C 6/6 before releasing task pilots. These small cohorts
are development admission evidence, not a broad reliability claim. Keep A
results whatever their sign; an unexpected non-reproduction requires an
identity review rather than silently dropping the comparator.

Up to six ADDITIONAL preparation attempts are reserved only to verify a
source-proven implementation correction within the already sealed B contract.
Archive the proof, patch, version and pre-execution verification cells first.
They cannot fund searching new poses, gains, reference profiles or thresholds.
Any previous valid failure remains a failure of its version. A scientific
failure with no demonstrated implementation discrepancy stops this dependency.

Thus preparation diagnostics have an absolute cap of 24 starts. Do not spend
36 task pilots when the shared preparation gate is already known to fail.
Independent CPU/data-interface work can still finish under a blocked report.

## 4. P2 — Observable, versioned behavior registry

Reuse the three-behavior common-scene design: visible pose targets, grey
obstacle/orange goal, and contour coexist, so background does not identify
the task. Keep M1 pose approach and M2 specified-side routing semantics and
existing success thresholds. Keep the existing robot, camera streams, TCP,
physics timing, WBC and SafetyGate unchanged.

### M3-v2: directed open contour, not hidden lap counting

Replace the closed contour with one visible 270-degree open arc at the
registered center/radius. End A is black at angle -3*pi/4; end B is white at
+3*pi/4, viewed from world +Z. Command 0 approaches A then traces the arc
counterclockwise to B; command 1 approaches B then traces clockwise to A.
The missing 90-degree section remains visibly absent. Both markers and the
requested direction/endpoints are stated in natural language.

Use the existing four-second approach and twelve-second traversal durations,
followed by the registered terminal hold, with a continuous desired pose path.
Retain the 12 ordered-progress sectors, 0.035 m cross-track RMS limit, and
existing backtrack/phase-jump tolerances; replace full-circle coverage by full
OPEN-arc coverage and the correct terminal endpoint within 0.035 m for 0.5 s.
The approach must meet the arc only at the commanded starting endpoint.
Direction, ordered intermediate coverage and endpoint are conjunctive; going
straight to the end, the wrong way, or through the gap is a task FAIL.

This is an intentional task amendment. M3-v2 success must never be reported as
M3-v1 one-lap stopping success. It removes the known identical-start/end lap
ambiguity without adding a hidden counter or changing model state dimensions.

Before physics, test all task experts for shared-prefix semantics, continuous
position/SO3/time, path clearance and visibility through the actual cameras.
Analytic projections and actual captures must show the command-relevant cues;
metadata saying an object exists is not enough. Visual-only marker size or
contrast corrections may be sealed before task physics; no per-outcome scene
or camera tuning, pixel-encoded target coordinates, or virtual object welding.

Check exact input aliases and near-neighbor conflicting futures, including
approach/traverse/terminal phases. Same instruction and identical allowed
input cannot be assigned incompatible required futures without explanation.
Time-driven expert generation is audit metadata, not a permitted policy input.
The absence of detected aliases in a finite dataset is not proof of complete
observability. A remaining structural phase ambiguity blocks that family.

### New splits and fixed attempts

Use a PI0256 seed/ID namespace. Quarantine all old PI0255 and new preparation
conditions as development data; they cannot enter new TRAIN or MT-VAL.
Create six fresh pilot families, forty TRAIN families and ten MT-VAL families,
with all three tasks and both commands per family. Seal additional composition
and geometry TEST manifests only; no test rendering, expert, model or statistics.
Detect cross-version duplicate scene/reset tuples, not just different filenames.

Each task gets 12 fixed pilot attempts (6 families x 2 commands). At least
10/12 first-valid end-to-end successes admits that task's fixed 80 TRAIN plus
20 MT-VAL collection attempts. A preparation FAIL in a task episode counts in
its denominator. Do not replace failed seeds, stop on the first ten successes,
or turn the other task's unused budget into more retries.

For a data-qualified task require at least 64 usable successful TRAIN episodes,
16 usable successful MT-VAL episodes, 24 complete TRAIN command pairs and six
complete MT-VAL command pairs. Report scenario counts as well as episode and
window counts. Observe and verify actual paired preparation state/history,
RGB and RNG identities; copying an image from the other command is prohibited.

## 5. P3 — Real data, common input contract and normalization

Capture actual body/wrist RGB, sensor/state timestamps, measured state,
desired versus achieved poses, frame/TCP/anchor, reference arrays, termination
and gate records. Preserve every attempt. Physically failed runs remain audit
data, not successful imitation demonstrations. Gripper remains unexecuted
neutral 0.5, with the limitation explicit in model/data cards.

Split whole scenario families BEFORE H40 windowing. Seal the actual source IDs
and actual observation-to-future pairing. Enumerate active-generation windows
and terminal-hold/audit-only windows by the versioned runtime phase rule,
not by whether their errors look easy. A local H40 window need not complete
the entire reach, route or contour task.

For EVERY enumerated window perform raw label -> original chunk/gate and
physical label -> actual processor -> actual inverse decode -> original gate
checks, with real StateV1 and timestamps. Save failures and clipping/support
statistics by task, episode, command and motion-phase quartile. Constant-only
terminal windows are not silently changed into motion to satisfy an arc gate.

A usable episode needs >=95% qualified active-generation windows and at least
one qualified active window in every motion-phase quartile. A missing hard
initial quarter cannot be excused by many easy late windows. Structural
identity/time/frame/shape errors require correction from original bytes or
block admission; a success flag does not make corrupted data eligible.

### Explicit C_MT_V2 processor experiment

Keep original action-relative conversion, row-packed rotation, slot10 mapping,
units, state/action dimensions, padding and masks. Use deterministic resize
that preserves the full field of view and the pinned tokenizer. Disable random
crop, horizontal flip and color jitter in BOTH arms for this first pilot.
Record this as an experimental setting, not proof of augmentation invariance.

Resolve TRAIN-only q01/q99 (NumPy linear quantiles) once from successful TRAIN
active windows that pass raw provenance/geometry/time checks, BEFORE any
codec-based eligibility filtering. Never recompute statistics after excluding
hard windows, reading MT-VAL or seeing model results. For coordinates already
normalized by the source processor, use the original affine convention and
an explicit range floor of 1e-6 in that coordinate's units; apply NO percentile
clamp to otherwise valid finite values. Inverse transformation uses the same
sealed values. Constant coordinates and masks require an explicit round-trip
test; no newly invented trainable embodiment rows or silent mask changes.

This is a new named processor identity, not a patch to historical C_T. Record
out-of-range rates under old C_T and new C_MT for diagnosis. Preserve unmodified
old receipts. Require float32 CPU round-trip position error <=1e-5 m and SO3
error <=1e-4 rad on applicable, non-padding samples, plus exact timestamp,
mask, dimension and gripper-convention checks. Do not clamp or smooth decoded
model trajectories to manufacture gate PASS.

Verify the actual collator, not a mock: only body_rgb, wrist_rgb, state_v1 and
language may reach the model. Negative tests inject target coordinates, task
IDs, future actions, success labels and clocks and must fail closed.

For every planned source/visit save the actual processed-input identity, model
noise seed/tensor identity and source mapping. Exercise the real tokenizer and
image processor twice in separate processes and across an iterator resume.
Torch RNG alone is insufficient: audit Python, NumPy, Torch CPU/CUDA and worker
state even when augmentation is disabled. Worker/init/iterator effects must
not change paired examples. Complete this before spending model-canary budget.

## 6. P4 — Conditional model admission and real resume canary

Release this stage only when at least two tasks have completed their fixed
data program and passed P3, and all cards/inputs/metrics are sealed. Do not
stop merely to ask for another training GO when these conditions are met.

Initializers are fixed:

- MT-Base: `nvidia/GR00T-N1.7-3B@2fc962b973bccdd5d8ce4f67cc63b264d6886495`.
- MT-G: `leoyuer/pi0246-gate-pass-campaign@efecea485ae77000599d19171b02491f8d514f42`,
  prefix `arms/PI0252/r1-speed-margin-20260928/G/checkpoint-2000`.
- Common Cosmos tokenizer metadata:
  `9ce19a195e423419c349abfc86fd07178b230561`.

Reuse PI0255's verified key/shape/tied-alias evidence. Base keeps original
slot10 tensors; G keeps adapted slot10 tensors. The single documented tied
LM-head alias is not permission to tolerate arbitrary missing keys. Actual
loads, parameter dtypes, processor mapping and weights must now be verified.
Do not initialize from U, R, a newest-model download or a partially saved run.

Use the existing qualified full-S4 implementation, mode restoration and
frozen-parameter checks. Per initializer compare a continuous eight-update
canary with a fresh process executing four updates, saving the FULL Trainer,
then restoring and executing updates five through eight. Both use the same
first eight rows and original 6000-step LR cursor, not a new short scheduler.
That is 16 canary optimizer updates per initializer, 32 total, all recorded and
DISCARDED as research initializers. Main training starts fresh from Base/G.

Require equal source/visit and noise bytes; matching model, optimizer,
scheduler/cursor and RNG/processor states; finite nonzero qualified gradients;
all four S4 state gradients; frozen tensors unchanged; and actual matching
continued updates. State identities and FP32 deterministic comparisons must
be fixed in the pre-execution card. A numerical nondeterminism exception must
be diagnosed, not accepted by widening a tolerance after seeing the mismatch.
This is real-model recovery evidence, unlike the old scalar CPU fixture.

## 7. P5 — Matched multitask training, no loss/seed sweep

Train exactly two primary endpoints, MT-Base and MT-G, for 6000 NEW optimizer
updates each. K=3 gives 2000 updates per task; K=2 gives 3000 per task. Use one
shared model per arm for all admitted tasks, not per-task specialists.

Inherit the PI0255 proposed recipe with these now explicit bindings:

- AdamW, LR 1e-5, betas (0.9,0.999), eps 1e-8, weight decay 1e-5,
  max grad norm 1, batch1, accumulation1, fresh optimizer, warmup10,
  cosine schedule over6000. Preserve the zero-LR cursor0 accounting.
- BF16 frozen backbone; original G action-head FP32 and freeze flags;
  projector/diffusion trainable, LLM/visual/vlln and other frozen scopes unchanged.
- Original native objective plus full pure-noise S4 physical losses:
  position0.4, increment0.1, rotation0.1, speed-margin0.1; speed margin1.8m/s.
- Same sealed C_MT_V2, data eligibility, source/visit/noise schedule and precision
  for both arms. No loss search, extra adapter, history input, curriculum switch,
  test-time adjustment, old-R continuation, or automatic full-model unfreezing.

Construct the full 6000-row schedule before training using a local RNG seeded
25603: balanced task blocks, alternate commands within each task, uniformly
sample complete-pair TRAIN scenario families, then episodes and motion-phase
quartiles and eligible windows. Both arms use the IDENTICAL schedule and
processed tensors. List successful unpaired episodes separately; do not claim
all collected windows were optimized. Noise is keyed to stage/source/visit,
not the arm, process ID or incidental random draws.

Seal all optimizer source/visit records and actual parameter-update evidence.
Preserve full Trainer checkpoints at1000,3000,6000 for recovery/archive; only
step0 and final6000 receive the offline quality screen below. An intermediate
checkpoint is not a selectable rescue endpoint. No extension after a valid
quality FAIL. G's historical adaptation cost is additional, not equal total
compute to Base.

An integrity failure in one arm must not be counted as that arm's quality
failure or automatically prevent the other independent arm from finishing.
A two-arm causal comparison requires two valid endpoints; report a partial
pilot explicitly otherwise. Never start a third scientific arm.

## 8. P6 — Fixed offline generation and semantic evaluation

This GO does NOT authorize learned-policy WBC episodes. Evaluate predicted
references offline after actual model generation; controller/task success
remains unmeasured for these learned models. No TEST or old DEV/F exposure.

Before first model quality query, freeze exact source IDs and two saved noise
tensors per input. Screen each initializer at step0 under common C_MT_V2 and
each corresponding final6000: four model states, two noises, no best-of-noise.
A shared input/noise identity is reused across Base/G and step0/final.

For EACH admitted task choose by fixed SHA order, independent of scores:

- 32 TRAIN windows from distinct episodes, balanced across four motion quarters.
- 16 MT-VAL windows from distinct episodes, likewise phase balanced.
- 12 MT-VAL semantic-start inputs from six complete command pairs: measured
  identical initial RGB/state within each pair, different requested behaviors.
- One reserved, meaning-equivalent paraphrase for each of those12 inputs.
- Empty-language ablation once per each of the six shared scene observations.

Also screen old O64 (64 inputs) as regression-only, under the SAME new processor.
Disclose that this is retention relative to each arm's common-processor step0,
not a re-run of historical G/C_T qualification. Reuse old archived outputs
read-only to show context; do not change their results or allocate extra queries.

Total maximum planned queries = 8 * (78*K +64):
K=3 ->2384 queries/9536 S4 inference heads;
K=2 ->1760 queries/7040 heads.
No query to choose samples or paraphrases is free. Existing identical valid
queries may be reused without spending another call; unused capacity cannot
be converted to extra checkpoints, seeds or tasks.

Save physical decoded trajectories, all four generated S4 states, actual input
and noise identities, gate issues/observed values, expert geometry and errors.
Independently CPU-recompute decoding, SO3, speed, gate and reported metrics.
A checksum alone is not a trajectory reconstruction. All-row and qualified-row
statistics must be clearly distinguished; do not hide rejected predictions.

Report per task/noise and macro/worst-task values: XYZ median/p95, SO3 error,
first-point jumps, speed maxima, REJECT/WARN, and change from the corresponding
step0. The original O64 criteria are reported unchanged but are not an entrance
exam or a substitute for multitask evidence.

For semantic pairs distinguish required behavior from merely different arrays.
Compare each output against both correct- and opposite-command references on
predeclared discriminative portions of the H40; report the correct-command
margin and direction/target/route indicators. Mark genuine shared prefixes as
non-discriminative. Confirm desired-command separation from EXPERT references
before model querying; never select the easiest pairs from model predictions.
Use paraphrase consistency and empty-language results as additional evidence,
not a claim of complete causal attribution to language from one ablation.

H40 route/contour intent is not full-episode task success. Do not apply a full
lap/route completion predicate to every short window. Family-cluster counts,
not overlapping windows or two noise draws, define independent sample support.
MT-VAL is development evidence; it does not become blind generalization proof.

## 9. Hard resource and exposure envelope

All counters are NEW PI0256 counters. Old unspent budgets are not transferred.

| Resource | Planned / allowed maximum |
| --- | --- |
| Preparation diagnostics |18 planned;24 maximum, including six proof-fix checks|
| New task pilots |36 maximum|
| New TRAIN + MT-VAL expert collection |300 maximum, conditional per task|
| Strict no-result engineering physical reserve |24 starts, never valid-failure retries|
| TOTAL native expert/preparation physical starts |384 hard cap|
| TOTAL native control steps, including implicit reset steps |480000 hard cap|
| Primary optimizer updates |12000 total;6000 per arm|
| Planned discarded resume-canary optimizer updates |32 total|
| Extra discarded engineering-canary/recovery update reserve |64 maximum; no primary endpoint extension|
| TOTAL completed optimizer updates |12096 hard cap;12032 planned|
| TOTAL raw training action-head calls |109120 cap:9*12096 plus256 partial/no-update calls|
| Planned offline queries |2384 for K=3;1760 for K=2|
| Strict invalid/no-result offline-query reserve |64, not retries of valid bad outputs|
| TOTAL offline queries / inference heads |2448 /9792 hard maximum|
| Learned-policy physical episodes, hardware, TEST, old DEV/F |0|

Keep native WBC actor calls distinct from GR00T heads. Each qualified training
update has1 native +4 S4 forward +4 backward-recomputation head calls. Count
partial attempts and failed pre-update calls additively. Model loads and CPU
fixture work are recorded separately, not mislabeled as scientific updates.
Submissions are not progress. Do not impose an arbitrary two-submission stop
on a demonstrably repairable engineering failure; budgets, source identity,
valid outcomes and exact-resume integrity remain binding. No unbounded retries.

Corelab is the expert/CPU/rendering endpoint; Hopper/H200 may run the now
conditionally authorized model canaries/training/evaluation after capacity
checks. Do not kill or alter another workload or submit duplicate live jobs.
Keep the existing finite runners rather than building a new orchestration or
monitoring framework. All scheduling/dispatcher work is task-bounded.

Preflight actual disk and quota separately. Unknown quota is not quota PASS;
obtain an approved capacity bound before a large write. Protect all historical
archives and active checkpoints. No broad cleanup. Newly created PI0256 scratch
copies may be retired only after fixed-revision full-content readback and a
manifest retain the current recoverable Trainer plus final endpoints. No
history deletion is authorized by that narrow task-local lifecycle rule.

## 10. Required deliverables and truthful terminal states

Create an actual `reports/PI0256/` evidence packet containing:

- `REVIEW_AND_HYPOTHESES.md`, actual runtime hash/readback and causal timelines;
- `PREPARATION_V2_CONTRACT.json`, CPU negatives and all A/B/C/proof-fix outcomes;
- `TASK_REGISTRY_V2.json`, visibility/observability evidence and all split IDs;
- complete attempt ledger, per-family/task scorecards and immutable raw indexes;
- real dataset/window manifests, paired-input proof, C_MT_V2 and codec/gate QA;
- actual Base/G cards, 6000 source schedules and full-model resume-canary evidence;
- immutable final Trainer receipts and full offline/CPU-replay result tables;
- `CLOSEOUT.md` stating exactly which dependencies, tasks and model arms completed.

Reuse permitted HF PUBLIC destinations under unique PI0256 prefixes. Publish
only assets/evidence with clear rights; private WBC sources, licensed model or
tokenizer files, credentials and private simulator assets are never copied
into Git or public data. If derived-model publication is not permitted, retain
an authorized private durable checkpoint and publish a public hash/receipt;
do not claim public weight availability. Fixed-revision full-content readback
must state authenticated versus anonymous access truthfully.

Terminal states:

- `PREPARATION_V2_BLOCKED`: causal/preparation gate not met; no task pilot flood.
- `MT_DATA_PARTIAL_K_LT_2`: preserve independent completed work; no MT1 training.
- `MT_INPUT_OR_RESUME_BLOCKED`: real-data/processor/model-integrity dependency open.
- `MT2_OFFLINE_PILOT_COMPLETE` or `MT3_OFFLINE_PILOT_COMPLETE` with
  `STOP_FOR_LEARNED_POLICY_PHYSICAL_ADMISSION` and all measured quality results.
- A missing arm or incomplete screen is explicitly `PARTIAL_MT_PILOT`, not COMPLETE.

Completion means an interpretable executed experiment, not automatic quality,
physical, manipulation, blind-generalization or deployment PASS. Do not select
an endpoint simply because it is less bad. Historical `selected_checkpoint=NONE`
and current `deployment_selected_checkpoint=NONE` remain unchanged. Physical
admission of a learned multitask model is a separate next scientific decision.

## 11. Startup instructions for the execution agent

Read this complete protocol and Issue #11's PI0256 canonical at the immutable
publication commit, then refresh real machine/job status. Use an isolated
PI0256 worktree and output root. Never mutate PI0254 or original PI0255 runs.

First produce the narrow six-family failure timeline and source-backed
Preparation-v2 contract, with complete pre-autoreset/reference/actor-input
instrumentation and CPU tests. Execute the bounded A/B/C program, not another
blind36-pilot run. If shared preparation qualifies, continue directly to the
new task registry, expert pilots and admitted fixed collection.

M3-v2 is the explicitly authorized open-contour task. Preserve M1/M2 and all
controller/gate contracts. Seal scene families before windows and observe
real language counterfactuals; no oracle phase or target information enters
model inputs. No test rendering or old DEV/F reads.

After actual data qualify, freeze every processor/input/source/noise identity.
Use deterministic full-view preprocessing and the explicit TRAIN-only C_MT_V2.
At least two qualified tasks release real resume canaries and the fixed paired
6000-update Base/G pilot under THIS GO; do not stop for redundant reapproval.
With fewer than two, publish the exact dependency instead of training MT1.

Run only the bounded step0/final offline screen and independent CPU replay.
No learned-policy simulation, extra checkpoint sweep, R restart, loss/seed
search, gate relaxation, WBC change, storage purge, mail, monitor or PR merge.
Archive both positive and negative evidence and stop at the stated physical
admission boundary. Never confuse a healthy job, metadata match or CPU fixture
with a learned multitask model result.
