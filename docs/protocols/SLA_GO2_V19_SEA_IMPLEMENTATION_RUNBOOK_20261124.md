# SEA V1.9 — implementation runbook and interface contract

Proposed implementation and preflight specification, **not executable authorization**. Keep #491 worktree intact. Isolated experimental branch only; reuse original receiver and safety guards.

## 1. 首次交接（CPU，优先）

1. Read-only verify #491 parent and closed #488/#489 evidence. Record source PT identity, source real HDF5 availability, Go2 frozen MJB and body/wrist render. Inspect actual runtime/approved pending work; do not cancel or reuse allocations.
2. Read reports/SLA_OFFICIAL_GO2_TRANSFER/O0_ACTUAL.json, reports/SLA_GO2_V18/Z1_ATTRIBUTION/{ACTUAL,INTERFACE_AUDIT,NATIVE_488_ACTUAL}.md, SOURCE_FRAMES.json, EMBODIMENT_FK.json, IMAGE_GEOMETRY.json, CROSSED.json. Retrieve private source archives only with site permission, verify hashes, do not publish bytes. Fix 4 source anchors and 4 Go2 development states before new outputs. Source missing camera/tool/geometry fields explicitly unknown.
3. Build virtual source representation in isolated renderer, compare real source vs rendered source before interpreting target gap. Establish paired common tool pose, camera reprojection and history provenance; separate oracle and deployable pathways by API type and tests. No mj_step, no GPU sampler. Produce SEA0_SOURCE_CUSTODY, SEA0_PAIRED_ROSTER and image overlays/geometry audit.
4. Concurrently build task-agnostic execution wrapper from world_intent_clock.py, reach_intent_replay.py, compiler.py, reach_student.py, standing_reach.py, native receiver/receipt and original guard. Run fixed 19-window replay plus known positive/negative cases. No physical integration until new task profile registration.
5. After CPU gates, publish one bounded scientific request for SEA1-PILOT with exact rows/hashes, two seeds, reuse of exact historical results, max64 sampler calls minus cache hits, live GPU identity/lease, deadline, no training/no native. At most one fixed repair if input integrity fails. No repetitive requests if equivalent allocation exists.

## 2. Proposed code map (not implemented by this plan)

| Module | Responsibility | Reuse / tests |
|---|---|---|
| scripts/sla_go2_v19/sea_contract.py | Typed observation/intent/receipt and provenance; oracle vs deployable separation | observation_stream.py, policy_query.py; reject stale/wrong epoch, GT fields |
| scripts/sla_go2_v19/source_custody.py | Source assets, actual observations/URDF/FK/camera/codec and source skill manifests | reach_embodiment_audit.py, O0, SOURCE_FRAMES |
| scripts/sla_go2_v19/twin_render.py | Paired robot-only scene render from matched states, no integration | dense_replay/site renderer; real RGB anchors, common physical pose, no future history |
| scripts/sla_go2_v19/crosspaint.py | Robot mask, visible pixel reprojection, source appearance, uncertainty/abstention | actual q/base/cameras; no task goal or hidden object geometry |
| scripts/sla_go2_v19/sea_observation.py | Raw / source bypass / deployable / oracle typed adapters; current+Memory packaging | official RobodojoInputs, masks, token order |
| scripts/sla_go2_v19/sea_action.py | Official inverse once, common tool transform, right-arm/gripper codec, unsupported left-arm requirement | SourceCurve, world_intent_clock, target_labels |
| scripts/sla_go2_v19/sea_reference.py | Shared reference/phase manager, world reanchor, PrefixClock, event handling | compiler, native_known, original owner/guard/STOP |
| scripts/sla_go2_v19/sea_panel.py | Immutable paired roster, resource-scoped sampler owner, exact cache proof | reach_forward, feedback_gpu; no unauthorized load |
| scripts/sla_go2_v19/sea_feature.py | External low-rank residual after frozen vision embedding, correspondence and frozen source bypass | official pi0 embed_prefix; no weight mutation |
| scripts/sla_go2_v19/sea_fit.py | Optional task-free optimizer/checkpoint/cost ledger | #491 cold preflight, independent dataset/optimizer identity |
| scripts/sla_go2_v19/sea_readout.py | Raw vs installed vs achieved, matched controls, task/resource accounting | existing ACTUAL/readout, episode denominators |
| tests/test_sla_v19_sea_*.py | CPU unit tests, then authorized fixed preflight | No automatic GPU/native from tests/CI |

## 3. Typed contracts

**ObservationPacket**: epoch, query_id, tick, sim_time_s, capture_wall_time, actual TCP/base/flange/arm q, actual left/right finger slide, measured gap if calibrated, camera frames with timestamps and intrinsics/extrinsics, robot geometry version, history entries (first + 8 samples + current), profile, missing masks, valid pixels/confidence/provenance. Original language passed through; adapter cannot branch on task meaning, object identity or phase. No teacher EEF, hidden objects, reward, score, GT depth, future frames or success metadata in DEPLOYABLE. Real sensor depth optional only if accessible. ORACLE packet is separate type; native owner rejects oracle_only=true even when pose is otherwise legal.

**PhysicalIntent**: frozen model/checkpoint/config/norm hashes, adapter id/hash, raw sampler hash, decoded action hash, query frame/common tool calibration, 50 right SE(3) knots and source times i/30s, gripper source fraction/physical codec/version (UNKNOWN allowed only noncontact diagnostic), source left proposal retained for capability audit. No WBC offset. Validate physical common tool point mapping, exact inverse and source time monotonicity.

**ExecutionReceipt**: episode/epoch/revision/source_query_id, installed wire and actual solver echo, last confirmed reference boundary and measured actual pose separately, consumed intervals/ticks, source index and phase per knot, lead/brake/hold/source labels, intended/installed/achieved pose and grip, owner ACK, STOP/zero-after counters, unknown actual MRT identity explicit, observation age/expiry, failure codes. Execution constraints from fixed registered profile, not learned goal correction. HOLD/REJECT/STOP distinct; runtime enforces cap/owner/lease, no actuation by reading registry.

## 4. Preflight tests (before any scientific registration)

- Source Mode original processor, state/action dtype/shape, quantile stats, image mask/order/history, fixed noise/seed and original frozen output equality. Adapter off is an actual bypass; store real raw outputs.
- Left/right and positive/negative XYZ, noncommuting SO(3), tool lever arm, double offsets, base change, world-relative anchor, known 26–27mm bias; no task goal in action mapping.
- Camera/geometry: projection/occlusion, common visible points, unchanged background, source/target physical tool and object relative state, actual left view missing, incomplete depth, unsupported source arm, mismatched camera/scene/hash/history and future frame; reject or label UNKNOWN.
- Leak tests: poison hidden object GT, task phase, teacher goal, reward, future frames and ensure deployable input/result independent; reject oracle before native publish. Synthetic GT isolated in offline diagnostics/training renders.
- Feature: identity/zero adapter, shuffled correspondence negative, noncorresponding token masks, frozen params/no target task labels, adapter-only gradients, source bypass stable, collapse/pose perturbation, checkpoint restore. Abstract trace is not real GPU training.
- Execution: real consumed seam, continuous interpolation and event boundaries, 41 points incl terminal pad, source-phase attribution, expiry/duplicate/late return/revision/owner mismatch, frozen H40/G7 and geometry, contact profile safety, fault STOP and no command after terminal. Static/zero/incorrect motion counted as negative by applicable fixed task score; no auto correction toward desired task. Require actual source motion before second query; no lead-in credit.
- Science: lock roster before outputs, retain missing/unsupported/abstain/technical slots; exact source-real anchor and seed control; no final TEST contamination from old task or R5 selection. Distinguish observation-only replay, true sampler, rendered synthetic and true native integration.

## 5. ACTUAL report / cost fields

Every phase writes a fixed registration with identities, predecessor evidence, population, metrics, approved caps, owner/live lease and duration; ACTUAL JSON/MD reports:
- planned/registered, reused exact cells, entered/returned, eligible, screened, unsupported, refused, technical, unstarted;
- source geometry, image correspondence and valid-pixel proportion; raw right-arm motion, physical source/target agreement, hold, signed direction, pose/geometric edit and event mismatch; seeds and per-state paired metrics;
- render groups/masks, lineage and independent split, trainable parameters, frozen encoder calls vs real sampler, external Adapter updates vs SLA updates, wall/CPU/GPU hours by stage, memory/cache, source/target training demo counts;
- real starts/ticks, source phase, actual source motion and task outcomes, contact/force, STOP/afterSTOP, terminal processes/lease;
- proof scope, limitations, decision, next stage/blocker, hashes and references to closed private evidence (no secrets/private bytes).

Compare at equivalent independent task performance and include rendering/calibration/engineering costs. Model forward, synthetic feature loss, oracle result, H0 success, assisted geometric result and genuine student success are different evidence. Task success rate null if no real attempts. Publish negative results and follow pre-registered fallback rather than arbitrarily tuning interfaces.
