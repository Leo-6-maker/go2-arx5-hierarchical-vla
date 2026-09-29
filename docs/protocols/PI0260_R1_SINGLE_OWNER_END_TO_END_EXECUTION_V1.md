# PI0260-R1 — One accountable lead, optional assistant, end-to-end Motion2 training

Status at publication: AUTHORIZED_WORKFLOW_AMENDMENT_NOT_EXECUTED.

## 0. Decision, scope and precedence

The user explicitly requests one agent to complete the whole task, or one lead with an optional assistant, without reciprocal waiting. This amendment implements that request. It changes OWNERSHIP, REVIEW EXECUTION and STAGE ROUTING; it does not authorize a new scientific recipe or a second compute allocation.

**Single execution authority:** `PI0260_R1_SINGLE_OWNER_MOTION2_END_TO_END_GO_V1`.

**Optional subordinate role:** `PI0260_R1_ASSISTANT_READONLY_SUPPORT_V1`. This is not a second experiment owner or a prerequisite for the lead to act.

Read this document together with the immutable scientific parent:

- Repository: `Leo-6-maker/go2-arx5-hierarchical-vla`.
- Parent commit: `90cbb926a842f08c3f9a78756d7a24e0d7a63f78`.
- Parent path: `docs/protocols/PI0260_FEASIBLE_OBSERVABLE_MOTION2_TO_TRAINING_V1.md`.
- Parent canonical: Issue #11 comment `5893504272`.
- Prior producer evidence: `74aa7bcb75e2ca37dc6c50bb1a0c7f15f9c7d7f0`, comment `5892641889`.
- Prior reviewer evidence/tools: `653d280dacdd3588f937281a4f59997bebbaec5a`, comment `5892796587`.

This amendment supersedes ALL parent clauses requiring a separate Agent V, its availability, its signature or a cross-agent H0/H1/H2 handoff before progression. Their substantive checks remain mandatory and become the lead's responsibility. Parent sections 3-7 scientific definitions, all data/sampling/model/evaluation limits, and section 9 resource/budget restrictions remain unchanged except for this review-ownership substitution. Parent sections 0, 5, 8 and 10 must not be used to reinstate a mandatory second-agent signoff.

H0/H1/H2 now name immutable EVIDENCE CHECKPOINTS in one workflow, not three external approval requests. A truthful same-owner replay receipt satisfies the amended review requirement. It must not be represented as independent second-agent review.

Goal: complete actual feasible/observable M1+M3 data, the two fixed Base/G6000 endpoints, the fixed offline/semantic evaluation, durable model/evidence archival and a compact reusable results display. Do not stop at a camera manifest, DATA_READY, a draft training card, a queued job or WAITING_FOR_AGENT_V when the scientific prerequisites allow the next stage.

The publishing PI reviewed the parent and latest available Issue #11 ledger, not a live SSH session. No new data/model result is asserted by publication. Preserve PI0259 `OBSERVATION_DESIGN_BLOCKED`, historical failures and `selected_checkpoint=NONE` / `deployment_selected_checkpoint=NONE`.

## 1. One lead owns the complete critical path

Use the existing science agent (previous Agent M) as the lead by default; it already owns source generation and model execution. A different lead may take over only after verifying real process ownership and adopting the existing ledger without duplicate computation.

The lead owns:

1. Runtime/resource checks, source versions, atomic output publication and cumulative accounting.
2. Camera/cue calibration, complete geometry and visibility checks, the fixed proposal selection, dataset and processor production.
3. Read-only source-specific H0/H1 verification, including completing the prior generic PARTIAL adapters.
4. Real model recovery canaries, both original initializers, one coordinated Hopper submission path and exact recovery.
5. Fixed model evaluation, fresh CPU replay of saved predictions, result interpretation and scope labels.
6. Minimal result cards/videos from already computed arrays, HF archival, Issue #11 updates and PR #77 pointers.

A single agent is sufficient. Default operating mode is SINGLE_OWNER. The optional helper may work on code review, read-only replay, test fixtures, documentation or already-recorded media in an isolated workspace. It cannot submit a model job, generate scenes, advance physics, load/query a model, change the authoritative dataset/recipe, select different examples or write to the lead's active outputs. A helper patch is a proposal; the lead reviews/applies it and owns the resulting version.

No helper needs to be launched merely to satisfy the protocol. No response timeout, repeated chat handoff or second-agent ticket is a prerequisite. If a critical check is not already complete, the lead runs or implements it directly. An available helper can reduce work, not add a second approval chain. Cosmetic review may be omitted from the critical path, but required correctness checks cannot.

## 2. Adopt existing work; do not restart the experiment

The first action is a finite takeover reconciliation:

- Read the current Issue #11 updates and the actual PI0260 workspace/attempt manifest.
- Refresh Corelab processes/GPU UUIDs and Hopper job state using the working, authorized scheduler environment. A missing qstat is UNKNOWN, not proof that no job exists.
- Record one execution owner and one existing run/attempt index. Reuse an existing lock/ledger; add at most a small owner/state JSON if necessary. Do not build a daemon, a new scheduler service or a generalized agent framework.
- Bind the original parent authority and this workflow amendment separately. Existing valid PI0260 data, source IDs, selected camera, proposal order, processor and model outputs retain their identities. An added ownership receipt is not a reason to change scientific hashes or regenerate data.
- Carry forward every started proposal, rendered frame, query, completed update and failed-recovery cost. The remaining allocation is the parent's allowance minus cumulative PI0260 use; PI0260-R1 supplies ZERO additional model or data budget.
- Reuse completed correct checks with unchanged bytes. Old PI0259 data remain calibration/reference evidence only under the parent's rules; old blocked H0 is not promoted to new PASS.

If a healthy job is already executing under the parent, adopt and observe it; do not cancel or restart it to change an authority label. Stop any newly competing dispatcher from submitting more work, but do not kill unrelated or healthy running computation. A physical or scientific failure remains a failure after an ownership transfer.

## 3. Internal verification instead of reviewer identity

The verification principle is: check saved raw inputs and outputs through a read-only replay path, not trust the producer's PASS fields. Organizational separation is optional; computational checks are not.

The lead may use the existing V verifier at its immutable source version, extend its source-specific adapters, and invoke it in a fresh CPU process on sealed arrays. It may share verified low-level math/FK/collision primitives, but must independently reconstruct counts, transformations and verdicts from those arrays and disclose shared dependencies. A second process alone is not proof that two implementations are independent.

Retain meaningful negative tests: altered cue ID, swapped instruction, missing geometry mislabelled as invisible, omitted first quartile, mixed world/native coordinates, changed paired input/noise, source/target/future-label leakage and a producer-PASS flag contradicted by arrays. Keep actual native-resolution natural RGB review for semantic discernibility; do not substitute segmentation counts or enlarged videos for what the policy sees.

Each evidence checkpoint records at minimum:

- stage, scientific authority, workflow authority and execution-owner identity;
- immutable input/source/processor hashes and verifier source/dependency hashes;
- expected/check counts, actual recomputed values, missing rows and negative-test results;
- natural-image inspection scope and actual example IDs where required;
- PASS/BLOCKED/NOT_RUN with exact reasons, and cumulative compute counters;
- `review_mode`: `SAME_OWNER_READONLY_REPLAY` or `OPTIONAL_HELPER_REPLAY`, with the actual reviewer identified;
- exact next stage authorized by the result.

Do not forge an Agent V signature or mark generic PARTIAL adapters as PASS. If a verifier is incomplete or a required private local collision binding is unavailable, the lead first performs the same authorized implementation/access recovery itself. Persistent missing evidence is a real QA blocker, not permission to omit that check. The new policy removes dependence on a person/session; it does not create evidence that does not exist.

## 4. Straight-through execution stages

### S0 — Adopt and prepare

Complete section 2. Use the actual execution code plus pinned V/PI0257-v2 tools where useful, not an assumed PR-head snapshot. Publication worktree is not automatically the runnable scientific worktree. Preserve active paths and private dependencies. CPU/serialization/import checks stay on PC/Corelab.

### S1 — New cue/camera H0

Execute the exact PI0260 cue pack and three ordered body rigs, reusing prior valid calibration joint arrays. Keep two actual96x96 streams, the original wrist, pixel thresholds, geometry/reference/IK and collision rules. Select the first qualifying common rig; no fourth rig, per-scene marker change or sample-specific camera.

The lead saves and directly reviews/replays RGB, masks/depth, full camera/FK transforms, common-start and later-pose coverage, failures and missing geometry. On H0 correctness PASS, freeze the design and immediately enter S2. A missing helper receipt cannot stop this transition. Failure of the fixed design still returns the real CAMERA_DESIGN_BLOCKED result.

### S2 — Proposal selection and dataset

Use the parent proposal grid, exclusions, namespace and fixed strata: up to20 DEV/100 TRAIN/40 KIN-VAL families; first6/40/10 jointly eligible families respectively. Both M1 commands and both M3 commands must have complete geometry, required observations and raw-gate coverage. Retain rejected proposal denominators. Run cheap geometry/raw checks before rendering and stop drawing when the fixed stratum quota is filled.

No post-hoc workspace contraction, M2 repair, test use, invalid-prefix training, stratum borrowing or post-codec refill. The lead performs the DEV checks directly and then production without sending a separate approval request. True proposal shortfall remains a scientific-data blocker.

### S3 — Actual codec, split and H1 closure

Produce the TRAIN-only common `PI0260_C_KIN_MOTION2_V1` once; preserve the affine/no-clamp convention, deterministic processing and physical-loss/decoder identity. Check actual allowed model inputs, full coordinate transforms, label round-trip, family split separation, paired commands and all motion quartiles. Each behavior still requires at least64 usable TRAIN and16 usable VAL sequences and24/6 complete pairs after codec checks.

Run the source-specific H1 replay locally under section 3, including the parent-required original collision replay on saved accepted output states. A source-specific same-owner PASS, actual parent data counts and verified resources release S4. Do not pause for an external signature, a video package or routine PI approval. A receipt cannot turn unobserved pixels or invalid IK into usable data.

### S4 — Freeze and verify the real model path

Freeze the same6000 source/visit rows and actual random inputs for both arms under the parent seed26003. Freeze evaluation IDs and two actual noise packets before any quality query. Actual input mismatches are checked before backward/update.

Perform each initializer's real continuous8 versus4+save+fresh-process-restore+4 test with the actual6000-step schedule, within the same32 planned discarded updates and remaining parent reserve. Main training starts from the original Base/G, not from canary outputs. Use the parent's predeclared runtime tolerances; do not change them after an observed mismatch.

If no previous stage has run, a simple sequential default is Base canary/step0/main6000/final followed by G canary/step0/main6000/final; this avoids a second dispatcher and staging all weights simultaneously. Preserve any already sealed compatible order, and never use the first arm's quality scores to change the second arm. One arm's quality FAIL does not veto completion of the independent matched arm. A genuine shared integrity failure does block affected work.

### S5 — Complete both model arms

Retain fixed Base and G revisions, original G full-S4 objective, BF16 frozen backbone/FP32 action-side scope, fresh AdamW LR1e-5, batch1, accumulation1, warmup10 and cosine6000. M1/M3 each receive3000 updates per arm. No new loss, seed, LR, initializer or task-specific checkpoint search.

Save full Trainer1000/3000/6000 with existing atomic-save and fixed-HF readback practices, keeping a real recoverable source. Ordinary checkpoint quality is not an engineering retry reason. Preserve all started attempts and exact-recovery costs. Do not terminate the goal at TRAINING_STARTED or first-checkpoint-ready.

### S6 — Evaluate, recompute and package

Complete the parent's1760 fixed logical query tuples across both step0/final states and two noises: per-behavior TRAIN, VAL, semantic/opposite-command, paraphrase, empty-language and O64 regression. Exact identical tuples may be reused with truthful accounting. Do not select more attractive examples or reroll bad outputs.

The lead runs saved-array CPU metric/gate/semantics replay, reports per-task/noise and macro/worst-task results, verifies paired inputs, and produces the H2 receipt. Preserve all invalid/rejected outcomes in denominators. All parent research-quality thresholds remain unchanged. Shared-prefix nondiscriminative cases must follow the pre-frozen rule, not a denominator reduction after model inspection.

Render a compact index plus a microtuning comparison from existing arrays using PI0257-v2. Identity, scale, SO3/gate reasons and synthetic/offline labels remain clear. No new inference for presentation and no cosmetic v3 campaign. Missing optional layout work cannot stall completion of the scientific package.

Publish durable model/evidence revisions and their actual readback scope; write one scientific completion summary to Issue #11 and a pointer to PR #77. No merge. The experiment is complete only when both actual6000 endpoints and the fixed evaluation exist; their scientific quality may still be FAIL.

## 5. No reciprocal waiting: operational decision rules

| Situation | Lead action |
|---|---|
| Helper not running, unreachable, slow or has not signed | Execute the required replay/review directly; continue if the real gate passes. No WAITING_FOR_AGENT_V state. |
| Helper returns a matching receipt | Reuse it after hash/scope verification; avoid duplicate replay. |
| Helper returns only presentation suggestions | Continue the scientific stages; apply optional display changes from saved arrays. |
| Helper or lead finds a reproducible correctness violation | Pause affected descendants at a safe recoverable boundary, version the correction, and replay the affected scope. Do not ignore it because helper participation is optional. |
| A potential correctness issue is not yet resolved | The lead investigates it directly; use PRECHECK/QA_IN_PROGRESS or a precise unresolved blocker, not indefinite peer waiting. |
| PBS is queued or expected GPU is busy | Record RESOURCE_WAIT, keep one legitimate active submission, and complete independent CPU/archive work. No duplicate qsub. |
| Import/path/schema/IO defect within the frozen semantics | Repair, run narrow tests and resume without another routine GO, subject to cumulative limits. |
| Fixed camera/proposal/data constraints cannot be met | Preserve actual failure and stop that dependent branch; do not claim this governance change permits scientific redesign. |
| An arm produces a valid quality FAIL | Preserve it and complete other independent required work; no score-driven rerun. |
| Remaining raw budget cannot complete a needed recovery | Preserve exact state and report budget-related PARTIAL/BLOCKED; no reset or silent extension. |

A late helper finding is not allowed to rewrite history silently. If it proves that an already used artifact was invalid, mark affected downstream results INVALID/QUARANTINED with source IDs, preserve recovery/evidence, and correct within the existing scope and remaining budget. It is not an excuse to substitute another scientific design or erase a failure. Distinguish a valid model-quality FAIL from invalid computation.

## 6. Required implementation tests for this workflow amendment

These are acceptance cases for the lead's thin routing/check wrapper, not a request for a new orchestration platform. Adapt the existing scripts and their argument checks:

1. H1 exact data/codec/replay PASS and no helper receipt -> canary/training admitted.
2. H1 data FAIL and helper says PASS -> training denied.
3. Required replay NOT_RUN or generic PARTIAL -> lead runs/completes it; never treat absent review as PASS.
4. Existing matching helper PASS -> reuse once with same manifest; no duplicate data/model query.
5. Presentation task unfinished -> scientific next stage admitted after correctness PASS.
6. Input hash or source ID changes after sealing -> descendant stage denied until affected revalidation; budgets retained.
7. Duplicate run owner/job identity -> no additional scheduler submission.
8. Existing canonical update/query results after restart -> exact resume/reuse, not a fresh scientific retry.
9. Single-arm quality FAIL with shared integrity intact -> second preapproved arm still runs unchanged.
10. All outputs present but no helper session exists -> goal may close as COMPLETE with truthful SAME_OWNER receipts.

Run these as CPU tests before the first changed stage admission. Publishing this protocol is not proof that these tests or a new executor have been implemented. Keep `run-episode` GUARD_STUB_ONLY; this amendment does not connect physical execution.

## 7. Accounting, resources and immutable boundaries

Reuse the SAME PI0260 cumulative limits, not a new grant:

| Activity | Parent cumulative limit, including prior PI0260 work |
|---|---:|
| Camera calibration pairs |512|
| Proposal families |160 total:20/100/40 strata|
| Kinematic attempts including strict technical reserve |664|
| Active H40 windows / production image pairs |60000 /120000|
| Main updates |12000 total;6000 per arm|
| Canary plus main plus completed-update recovery costs |12096 total|
| Raw training action-head calls |109120|
| Planned offline query tuples |1760|
| Offline query attempts / inference heads |1824 /7296|
| Dynamics, WBC, hardware, TEST, oldDEV/F, learned-policy physics |0|

Corelab: zero-integration prescribed-pose rendering, CPU QA and no-gradient inference. Hopper: admitted real-model canaries/training on a compatible lawful allocation; auto routing is allowed. Ordinary CPU debug is not an H200 job. PC is the single coordinating entry. No helper-owned GPU queue, broad cleanup, unrelated process interruption, automatic mail/monitor or new long-running polling service.

Keep current recovery sources and unique evidence. Reuse unchanged HF fixed-revision verification. Separate quota from free space and use the inherited next-write-overlap plus10GiB reserve. Verify secrets/licences/private URDF/meshes/WBC/tokenizer exclusions before publishing. A faster workflow does not waive these limits.

No change to data science: M1+M3 only, M2 deferred; same cue pack/three rig options/IK/reference/collision thresholds; same proposal selection and conditioning claim; same Base/G and objectives; same evaluation denominators; selected/deployment NONE. No camera or IK parameter may be changed just because only one agent is operating.

## 8. Reporting and completion

Use the existing ledger plus a small stage-state record: owner, current stage, accepted input/output hashes, actual counters, active job identity if any, last verified outcome, precise blocker and next action. Seal scientific stage outputs; mutable progress must not overwrite them. Update Issue #11 at design freeze, data admission, training entry, model endpoints, final closure or a materially new blocker. No repetitive identical WAIT updates or needless all-history rehashing.

Human-readable states: RUNNING, RESOURCE_WAIT, BLOCKED_SCIENCE, BLOCKED_INTEGRITY, PARTIAL, COMPLETE. Existing parent detailed failure codes remain attached as reasons. Another agent not replying is not a science/integrity blocker because the lead owns every essential task.

Final objective remains `PI0260_KIN_MOTION2_PILOT_COMPLETE — STOP_FOR_PHYSICAL_TRANSFER_ADMISSION`, with the R1 workflow authority recorded. Require the actual selected family dataset, real codec/QA and recovery evidence, both6000 complete Trainers, the fixed evaluation with CPU replay, durable archive and a minimal understandable results index. `COMPLETE` means the planned experiment finished; scientific PASS/FAIL is reported separately for each model. A real missing arm/data/score is PARTIAL or a named blocker.

A same-owner audit is honestly labelled SAME_OWNER_READONLY_REPLAY. Computational cross-checking is preserved, while a second person's/agent's independent review is no longer a formal entry condition. No result is called actual WBC execution or universal physical generalization.

## 9. Copyable single-lead start instruction

Execute `PI0260_R1_SINGLE_OWNER_MOTION2_END_TO_END_GO_V1`. Read the complete scientific parent90cbb926 and this R1 amendment. Own the task from actual current-state adoption through new cue/camera calibration, conditional M1/M3 data, source-specific read-only H0/H1 checks, real Base/G recovery tests, each6000 training run, fixed1760-query evaluation, CPU replay and archival/display. Do not wait for Agent V or ask the user to relay each stage. If no helper participates, perform its essential checks yourself; label them truthfully. Preserve the scientific definitions and all already-consumed PI0260 budget. No rerun of valid data, no job restart merely for ownership, no second dispatcher. Finish the whole preapproved chain whenever its actual gates pass. Stop only on a substantive unmet frozen condition, unresolved correctness, resource/budget limitation or the declared physical-transfer boundary. A dataset-only report is not completion when training is admitted.

Optional helper instruction: support this lead with bounded read-only analysis/tests/media on exact immutable inputs. Return patches/receipts and any concrete defect, not a separate GO request. No model/scene/physics/scheduler activity, no edits to the lead's active artifacts, no independent persistent goal. The lead can complete all essential work without waiting for you.
