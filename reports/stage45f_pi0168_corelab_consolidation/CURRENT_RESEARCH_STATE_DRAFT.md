# Current research state draft — PI0168

## Control-plane state

This draft is controlled by GitHub live state, not by a local path or an earlier handoff summary.

| Item | State at PI0168 preparation |
|---|---|
| Issue #11 gate | Comment `5658063963`, `PI_GATE_STATE_0168` |
| Launch pointer | PR #69 comment `5658065067`, read before the Issue #11 gate |
| Audited Vanda snapshot | PR #71 head `940df1ab6f428fd0260fe836e36d4cd98c6161e3` at gate issuance |
| Vanda ownership | PR #71 remains the active Validation-E execution surface; Corelab has no active science execution role |
| Corelab branch | `stage45f/corelab-execution-risk-consolidation-prep`, created from the exact audited snapshot |
| Corelab work | Static execution-risk audit, topology/archive preparation, and post-Validation harness design only |
| Scientific recommendation | None. This packet does not choose the next experiment, tier, arm, or resource. |

## What is established without payload access

- The PBS path has a single direct entrypoint with exact branch/HEAD/clean-tree/fresh-root checks, an FFmpeg capsule bootstrap, and a separate import-only probe.
- The frozen evaluator calls `model_imports_present()` before artifact admission and keeps its model/runtime imports after the admission boundary in the reviewed runner.
- The reviewed recovery source has a function-local NumPy import in `raw_inventory()` before the evaluator admission call. This is the published PI0167 blocker, not a new PI0168 finding; Vanda owns the repair.
- `checkpoint_eval.py` has a top-level NumPy import and must remain post-admission. Stage26 and the Stage45C runtime adapter contain function-local model/runtime imports and are scientific runtime dependencies, not static admission helpers.
- Historical accounting through allocation 9 is preserved as engineering evidence: allocation 8 left six valid raw cells and thirty missing cells; allocation 9 made zero new model-side calls and was blocked by pre-admission import pollution.
- The repository's canonical README, accepted-baseline document, and research timeline stop at Stage44R/Stage45A while live Stage45F PRs remain open. This is navigation drift, not silently corrected in this packet.

## Ranked findings

### P0 — 0 new findings

No new P0 was identified that plausibly changes allocation 10. The published PI0167 pre-admission NumPy contamination is a known finding with an existing Vanda-owned repair path, so no additional PR #71 comment was posted.

### P1 — 2 findings

1. **Known PI0167 parent contamination.** At the audited snapshot, `raw_inventory()` imports NumPy at recovery line 220 and is called before `validate_artifacts()` at line 1028. The existing guard correctly rejects the contaminated parent, but the earlier static guard does not detect function-local ordering. Deterministic trigger: any parent call to this inventory before admission. Mechanical fix class: isolated child process. Owner: Vanda.
2. **Post-admission-only import contract is not encoded in every reusable helper.** `checkpoint_eval.py` imports NumPy at module scope; Stage26 and the runtime adapter import Torch/GR00T/NumPy/project modules inside runtime helpers. Deterministic trigger: a static/pre-admission helper imports one of these modules transitively. Mechanical fix class: pure stdlib admission layer plus explicit process boundary. Existing tests exercise behavior, but they do not prove the entire transitive parent remains clean before admission. Owner: future consolidation maintainer under PI review.

### P2 — 3 findings

1. **Train24 versus Validation-E contract duality.** The Stage45C adapter expects 24 Train24 episodes while Validation-E freezes six windows per arm. The distinction is safe only when represented by separate RunSpec contracts.
2. **Static guard coverage is local rather than transitive.** Source markers and top-level import checks do not alone prove that a helper's call path remains runtime-clean. The proposed ledger/process schemas make this boundary inspectable without changing the frozen guard.
3. **Navigation drift.** Live Stage45F work is not reflected in the canonical Stage44R/Stage45A stopping-point documents. A documentation update should follow accepted sealed evidence, not precede it.

## Topology decision draft

The detailed table in `REPOSITORY_TOPOLOGY_AND_ARCHIVE_PLAN.md` classifies PRs #46, #49, and #51–#71. The only current active execution classification is PR #71. This PI0168 packet is the new active-support surface; all archive/close/ref actions remain proposals and were not performed.

## Post-Validation consolidation sequence

1. Vanda completes its authorized Validation-E path and emits a truthful raw/derived/failure seal.
2. Freeze the accepted source, protocol, raw/evidence, counter, and output-inventory hashes as one provenance record.
3. Port only portable validation/hash/counter/process-boundary semantics into a canonical support surface; retain device/runtime and Train24/Validation-E adapters separately.
4. Update canonical README, accepted baselines, and timeline in a separate PI-reviewed documentation change.
5. After PI acceptance, handle archive candidates with immutable provenance preserved.

No step above authorizes a new model/GPU/training/simulator/WBC run, modifies PR #71, or selects the next scientific experiment.

## Final live-head classification

This exact block is completed immediately before handoff publication and is duplicated in the manifest:

```text
VANDA_HEAD_MOVED: pending_final_live_read
NEW_HEAD: pending_final_live_read
DELTA_FILES: pending_final_live_read
DOES_DELTA_INVALIDATE_AUDIT: pending_final_live_read
UNAUDITED_LIVE_DELTA: pending_final_live_read
```

## Review gate

`HOLD_GPT_PI_REVIEW`: after the Draft PR and Issue #11 summary are published, Corelab performs no proactive experiment, merge, cleanup, or branch movement.
