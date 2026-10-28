# PI0267-R1 — Observable entry contract, then complete recovery learning

Authority: **PI0267_R1_ENTRY_CONTRACT_AND_RECOVERY_TRAINING_GO_V1**
Date: 2026-10-02. One main execution owner; optional helpers are not signature dependencies.
Continue on `pi0267/executor-decoupled-recovery`. No new general Issue, PR77 pointer, branch proliferation, merge, history rewrite or remote-job cancellation.

## 0. Inheritance, actual status, and publication scope

Read the full parent protocol at `f21bca4ec5b004e968f285032fc57df904deebef`, path `docs/protocols/PI0267_EXECUTOR_DECOUPLED_TASK_SPACE_RECOVERY_V1.md`. This amendment replaces ONLY the M3 entry acquisition in the task scorer, specifies the finite re-admission check, and authorizes continuing the parent's remaining S2-S6 once real admission succeeds. Unchanged parent specifications and cumulative budgets remain binding.

Sources: PI0267 canonical 5943808755; `TEACHER_LIVENESS_REPORT.md`, `TEACHER_QUALITY.csv`, `evidence/M3_ENTRY_DECOMPOSITION.json` at `99f9c1c8e37743e6d82fb80c50a253700373fa31`; final closeout `2d6f049027d6a9803dc5406c5dc7b0dfc1344d66`; actual V2 code `scripts/pi0266_metrics.py` at that evidence SHA.

Preserve the original PRIMARY_PARTIAL / PRIMARY_BLOCKED_TEACHER_LIVENESS and all original score arrays, 64 CPU outcomes and HF receipts. They remain correct outcomes under the OLD fixed contract. GR00T queries/updates, virtual image generation and WBC starts were zero at closeout. This amendment is an explicit scientific contract change after development evidence, NOT an unrecorded engineering fix, a blind validation, or permission to rewrite a historical FAIL.

The publication author's review read report/code/decomposition and ran constructed CPU geometry fixtures. The original HF trajectory tar was not obtained in this review environment; original 64-array V3 replay and production integration are NOT claimed completed. The executor must perform them below.

## 1. Finding and bounded hypothesis

All four OLD/NEW x E0/E1 groups have M1=8/8, M3=1/8. NEW reaches all M3 final positions within 15 nm, and its local prefix progress/identity tests pass. Five failed M3 conditions per NEW backend are already beyond the negative 1 mm path deadband at the instant V2 admits entry; two more acquire a backward excursion while still approaching the true entry. This is not a new VLA fit result or a WBC test.

V2 admits entry after 0.10 s with position <=15 mm, orientation <=0.10 rad, and absolute directed entry-angle distance <=5 mm. It then immediately penalizes negative progress below -1 mm and drawdown above 1 mm. Thus a physical approach admitted at -3 mm can immediately fail before arc tracing begins; an approach admitted at +3 mm can be penalized for correctly finishing its approach to zero.

Working hypothesis: broad start-neighbourhood readiness and acquisition of the actual directed-arc entry were conflated. Shared-scorer numerical replay does not independently validate this task interpretation. We will test an observable two-step entry contract WITHOUT changing the teachers, paths, speeds, EEF data, controller, reversal deadband, bins, or endpoint criteria. Neither teacher is selected for having a better score under the new contract; NEW remains the preselected teacher.

## 2. The sole new task scorer: PI0267_PATH_V3_OBSERVABLE_ENTRY

M1 is unchanged. M3 keeps the exact current target, direction/colour mapping, 270-degree path, 12 ordered bins, positional/orientation/path-distance limits, terminal hold, safety rules and sampling identity.

Let d in {+1,-1} be the original command direction, a=-d*3*pi/4 the true entry angle, and r the declared circle radius. Using ONLY achieved TCP position/orientation and the fixed task geometry, define

    delta_i = atan2(sin(d*(atan2(y_i-c_y,x_i-c_x)-a)),
                    cos(d*(atan2(y_i-c_y,x_i-c_x)-a)))
    entry_arc_i = r * delta_i.

Retain the V2 broad readiness predicate:

    near_i = ||p_i-p_entry|| <=0.015 m
             AND SO3(R_i,R_goal) <=0.10 rad
             AND |entry_arc_i| <=0.005 m.

Keep a continuous run of `near_i`. Reset its start only when `near_i` is false BEFORE acquisition. A gap in a sampled input or invalid time is an input-integrity failure, not an artificial continuous hold.

**Acquire the arc once, at the first actual sample i for which**

    the current near-run has lasted >=0.10 s
    AND |entry_arc_i| <=0.001 m (plus the existing 1e-12 m arithmetic tolerance).

This does NOT require an additional 0.10 s dwell inside the narrow band, nor does it raise reverse tolerance. It is readiness in the original neighbourhood followed by a physically observed capture inside the already-existing path deadband. If a trajectory never acquires it, M3 fails as MISSING_ENTRY. Do not interpolate an unobserved crossing to rescue a sample.

After this first acquisition, execute the original V2 continuous angle unwrapping, gap/reverse, 12-bin dwell, contour-distance and terminal checks from that sample onward. Keep the physical entry as angle zero; do NOT subtract the acquired sample's delta. No later re-arming, re-zeroing, clearing sticky failures, selecting a best start, or restarting the evaluator can erase a real reverse/shortcut. Safety and collision evaluation applies throughout the episode, including approach.

Record separate fields for earliest broad readiness, actual entry acquisition, directed distance at both, and pre-acquisition motion. Keep full raw motion; only contour-direction scoring starts at acquisition. Desired-reference/source time, teacher phase, model output, checkpoint, command outcome and target success MUST NOT decide the acquisition time. The scorer can use achieved history; no new history/phase variable is fed into the stateless policy or teacher.

Implementation: append a new versioned scorer/wrapper, do not edit `pi0266_metrics.py` or overwrite its records. Desired/installed references may enter diagnostic tracking error ONLY, not entry or task-success decisions. Return legacy V2 side scores under their pinned identity. Downstream R1 task scores explicitly name V3.

Equivalent acquisition pseudocode:

    start = None
    acquired = None
    for i in chronological_samples:
        if not near[i]:
            start = None
            continue
        if start is None: start = t[i]
        if t[i]-start >= 0.10-1e-9 and abs(entry_arc[i]) <=0.001+1e-12:
            acquired = i
            break

No scanning backwards from a successful endpoint and no source-phase >=4 s condition.

## 3. Finite validation and re-admission — no new 64-episode rollout

A. Read the supplied `scripts/pi0267_r1_entry_contract_probe.py` and its summary. It is a constructed geometry test, NOT original-array replay or an installed production scorer. In the publication review 28 fixtures passed (12 positives,16 negatives); eight straight-approach/full-arc positives fail V2 and pass the proposed V3 acquisition. This establishes a task-contract counterexample, not the actual new scores of the original 64 cases.

B. Implement the new scorer in the source-pinned runtime and independently implement the entry event check (not just call the same function twice). Tests must include both directions/radii; approach from both angular sides; correct full 270-degree motion; first-sample tiny jitter; a real 5 mm reversal AFTER acquisition; a 3 mm negative excursion followed by recovery; wrong direction; missed entry; skipped interval; endpoint shortcut; wrong goal/orientation; unsafe trajectory; and attempted reacquisition after failure. Preserve all original V2 fixtures. Add full nominal source references including their approach, not only fixtures that start exactly at the black/white marker. At most 96 deterministic constructed/nominal CPU fixture cases, no parameter/threshold search.

C. Read the already sealed 64 raw mathematical episodes. First reproduce their original V2 rows, then evaluate those SAME arrays under V3. No new teacher plans, random streams, trajectories, EVAL scenes or GPU work are needed for this step. Report all OLD/NEW x E0/E1 rows, score disagreements and exact acquisition events. Keep float differences distinct from changed task semantics. Verify whether the observed V2 reverse/gap events precede the new observable acquisition. Any real reverse after acquisition remains a failure.

D. The preselected NEW teacher still must meet the original count targets: E0=16/16; E1>=12/16 with each task>=6/8, now explicitly under V3. Original local continuity/rotation/observable-label/prefix-progress tests must remain true. A passing saved-array replay suffices to resume: DO NOT rerun 64 trajectories simply to produce a fresh PASS artifact.

If the exact amended rule or negative fixtures still fail, stop R1_CONTRACT_OR_TEACHER_UNRESOLVED and identify the actual residual defect. Do not widen deadbands, change the path/teacher/seed or try additional start rules. This is a finite correctness gate, not an invitation to indefinite rescoring until every legacy case passes.

Once B-D pass, immediately advance S2-S6 below under this GO. Do not end the whole goal at SCORER_FIXED, CPU_PASS or DATA_READY; no auxiliary signature or new routine PI approval is needed.

## 4. Resume the actual learning study — all non-scoring science retained

Keep NEW = ARCLENGTH_EXP_CORRECTION_V1 exactly as released: original path shape/targets; weighted SE(3) arclength with 0.05 rotation metric; 0.02 m_eq/s; exp(-2*tau) world-left pose residual; no hidden phase/time. Do not change old/new teachers because their binary S1 counts happen to coincide.

S2: same frozen four DEV /24 TRAIN /eight unopened EVAL numeric families, cameras, grey virtual-tool geometry, 96x96 RGB and task-space StateV1. Generate at most10,752 declared virtual observations (1,536 nominal +9,216 recovery), from16 arclength anchors and the original six paired +/-10mm/0.06rad perturbations. Use each state’s own reference and fresh matching RGB. Preserve the parent’s coverage, visibility, observable-label, raw gate, tool-collision and codec rules, including minimum16 complete TRAIN families. No extra/replacement scene, Go2 IK filter, hidden target coordinates, phase token or mixing PI0266’s deficient physical windows as successful demonstrations.

Use the parent's one-shot shared TRAIN-only statistics rule ONLY when its stated destructive-transform test requires it. No EVAL fitting, per-arm normalizer, or relaxed gate. Preserve decoder and rot6d conventions. New-contract donor evaluation and historical old-contract evaluation remain distinct.

S3/S4: fixed PI0264 L2048 donor at HF `leoyuer/pi0246-gate-pass-campaign@e4290e6ca387abd2de4337b5f12b9c72a5f85c99`, prefix `arms/PI0264/L/checkpoint-2048`. N4096 nominal only; R4096 50% exact common nominal replay +50% paired recovery states. Same tasks/commands/eight M1 training templates/current M3 text, actual stochastic packets, original full-S4 loss, action-side trainable scope, precision and masks. Fresh AdamW LR1e-5,warmup10,cosine4096,microbatch1/accumulation2. True continuous8 vs4/save/fresh-resume4 for each arm,32 discarded updates total; restart official weights from donor. Each arm2048/4096 complete checkpoints; final-only quality, no arm-dependent recipe changes.

S5: after both endpoints freeze, execute the same192 virtual closed-loop cases (donor/N/R x8 held-out families x2 tasks x2 commands x E0/E1). Every0.4s use fresh rendered achieved observation -> model H40 -> original gate -> execute prefix, up to40s/100 queries. No teacher correction, hidden future, geometry routing or best-of sampling. Use V3 primary task metrics plus V2 side scores, maintaining the parent’s13/16 per task/backend and6/8 per command criteria, all safety requirements and fixed offline/legacy panels. Images/labels/scores of EVAL remain unopened until the original endpoint-freeze condition; the R1 DEV score repair does not consume or relax that holdout.

S6: same secondary eight expert +24 model WBC diagnostics on the predeclared first two TRAIN families, same safe reset, ROW6D, native history and conservative physical checks. Primary V3 task semantics apply and V2 is side-reported. No WBC weights/gains/dynamics changes, no physical teleport, no rescue routing, no hardware. A secondary-only failure cannot block completed primary learning and must not be relabelled virtual success.

The V3 repair does not establish robot-agnostic policy transfer; E0/E1 remain two mathematical executors, not two physical embodiments. The actual scientific treatment is recovery coverage N vs R, not this revised scorer. Do not report a V2-to-V3 score increase as a learning gain.

## 5. Counters, engineering autonomy and evidence

Do NOT allocate a second PI0267 budget. Retain already completed64 CPU episodes/128000 ticks/6400 teacher plans/2560 mathematical seconds. R1 adds only bounded deterministic fixtures and scoring/reconciliation of saved arrays before the unchanged downstream work. Count scorer calls separately from new teacher plans and model queries.

Inherited remaining maxima:10,752 virtual proposals,40,000 frame pairs;8192 official updates+32 canary+256 completed-update technical recovery=8480;192 virtual model episodes,208 attempt cap;32 secondary WBC starts,36 attempt cap/1476 actual simulated seconds;new GR00T queries <=23,248 planned,24,576 attempts/98,304 inference-head cap;210,000 training raw-head cap. Current completed downstream counts are zero, but owner refreshes reality before any start. Savings are not extra scientific cases.

PC coordinates; Corelab handles CPU/scoring/rendering/virtual feedback/WBC/no-gradient inference; Hopper handles only real GR00T canaries/training on compatible GPUs. The owner may fix paths/imports/JSON/serialization/persistence/cache/layout/exact recovery to implement frozen semantics without repeated permission. No mechanical two-retry rule; valid scientific negatives are not technical retries. Never drop evidence or downgrade a real safety/integrity defect to keep running.

Storage helper remains independent, protects donor/dataset/normalizer/active endpoints/recovery points, and does not gate main work. Reuse complete historical HF receipts. Package new arrays and logs per shard/episode; do not re-read the earlier438,115 members or198-member archive merely to reconfirm unchanged hashes. Obtain required original64 arrays from the existing local verified archive, restoring only missing objects from fixed HF when needed.

Deliver a compact R1 attribution and both metric tables, negative/positive fixture receipt, exact production scorer SHA, four real full Trainers, all fixed model results with CPU replay, fixed HF receipts and clearly labelled virtual versus physical videos. These are cumulative delivery requirements, not an instruction to generate elaborate media before training.

Terminal: **PI0267_R1_ENTRY_CONTRACT_RECOVERY_STUDY_COMPLETE — STOP_FOR_WIDER_SCENE_OR_SECOND_EMBODIMENT_ADMISSION**. Separate experimental completion, quality, old-V2 historical outcomes and secondary WBC transfer. Formal selected_checkpoint=NONE/deployment_selected_checkpoint=NONE. A virtual_research_candidate can be recorded only under the unchanged count/safety criteria, explicitly with metric V3 and its limited virtual/new-local-geometry scope.

This contract is now a development-informed revision; improved rescored legacy counts are not new independent scientific evidence. Preserve the original PRIMARY_PARTIAL rather than silently converting it to PASS. The purpose is to remove a demonstrated task-scoring inconsistency and finally execute the predeclared recovery-learning comparison, not to keep revising acceptance rules until a desired model is selected.
