# PI0168 repository topology and archive plan

This is a navigation and future-handling plan only. It does not close, merge, delete, move, tag, or retarget any branch or pull request. The inventory is based on GitHub live PR metadata read during PI0168 preparation. The source snapshot for the execution-chain audit is PR #71 at `940df1ab6f428fd0260fe836e36d4cd98c6161e3`; the final live-head movement is recorded in the current-state draft and manifest.

## Classification vocabulary

- `ACTIVE_EXECUTION`: currently owns an authorized scientific execution.
- `ACTIVE_SUPPORT`: current non-scientific support or provenance packet that remains in review.
- `SEALED_HISTORY`: retained historical evidence or infrastructure whose provenance is still referenced.
- `PREP_ONLY_SUPERSEDED`: preparation path superseded by a later gate or ownership transfer.
- `ARCHIVE_CANDIDATE`: no current execution/support role; future archival action still requires PI approval.
- `SCIENCE_RESULT_NOT_YET_CONSOLIDATED`: science-bearing branch whose result is not yet represented in one accepted canonical research record.

## Live PR inventory

| PR | Title | Base -> current head | Purpose | Classification | Later evidence / future handling |
|---:|---|---|---|---|---|
| #46 | `docs(stage45c): seal PI0100 TRAIN24 evaluation completion` | `stage45c/b2-checkpoint-train24-evaluation` -> `2c1e9800d27ebc8d0c32c7c81dd92a7ff29ff222` | Stage45C TRAIN24 completion report | `SEALED_HISTORY` | Retain as historical report source; consolidate only after accepted canonical docs are updated. |
| #49 | `[Stage45C] Freeze reusable manifest-driven train/eval infrastructure` | `stage45c/b2-checkpoint-train24-evaluation` -> `b57392575e5c8765ee1695200fe4c73fb4303f24` | Reusable manifest-driven train/eval infrastructure | `SEALED_HISTORY` | Later runtime and protocol work descends from this infrastructure; retain immutable provenance. |
| #51 | `[Stage45D] Vanda H200 D1 execution and autonomous engineering recovery` | `stage45d/corelab-d1-protocol-correction` -> `00fe100722268da3134bc7bfe2b535fa86475f7d` | D1 H200 execution/recovery history | `SEALED_HISTORY` | Retain as historical execution context; no new work assigned here. |
| #52 | `[Stage45D-D2A] Matched teacher-bridge vs free-running path divergence` | `#51` -> `a843294780c1392369bfc987e468857c63fc13a5` | D2A path-divergence study | `SEALED_HISTORY` | Preserve for method lineage; no merge into the current Validation-E line. |
| #53 | `Stage45E: N2 balanced XYZ normalization` | `main` -> `a476a581f4d9a5ad104e691d4037762c6f1685ba` | N2 normalization science path | `SCIENCE_RESULT_NOT_YET_CONSOLIDATED` | Keep branch/PR as evidence until PI decides canonical result treatment. |
| #54 | `[Stage45D-D2B] Forward-Euler solver resolution study` | `#52` -> `f83d53758e0ccda951c7af4dd7a606642a782b93` | Solver-resolution study | `SEALED_HISTORY` | Retain method history; no current execution owner. |
| #55 | `Stage45E: Vanda N2 TRAIN24 evaluator admission preparation` | `#53` -> `332dd169c3ea10c9ae8c2006b1de13f8be2205aa` | N2 evaluator admission preparation | `PREP_ONLY_SUPERSEDED` | Superseded by later Stage45F governance; preserve but do not reactivate. |
| #56 | `Stage45E: Corelab N2 replication and mechanism GPU sprint` | `main` -> `d762e731d2c11ceea3c001c37ac27890fe6f4e70` | N2 mechanism/replication science | `SCIENCE_RESULT_NOT_YET_CONSOLIDATED` | Keep as science-bearing history; any consolidation requires PI interpretation. |
| #57 | `Stage45E-L1 Physical-error-aware XYZ loss intervention` | `#52` -> `c68611a9a577bd23314afaed9c6ffb7fc829e9b2` | L1 loss intervention science | `SCIENCE_RESULT_NOT_YET_CONSOLIDATED` | Preserve exact branch evidence; do not fold into Validation-E mechanics. |
| #58 | `Stage45E prep: neutral TRAIN24 candidate-selection tooling` | `#52` -> `253906ce6543c1122a8e8bc4686074b6caea89b1` | Neutral candidate-selection preparation | `PREP_ONLY_SUPERSEDED` | Superseded by later gates; future archive candidate after review. |
| #59 | `Stage45E-L1 prep: read-only 612326 recovery audit` | `stage45e/l1-physical-error-aware-loss` -> `937db09eb2aed1d6c00f3869ff8da2c734901543` | Read-only recovery audit | `PREP_ONLY_SUPERSEDED` | Retain as audit trail; no active execution role. |
| #60 | `Stage45E-N2 prep: continuation admission and repair contract` | `#56` -> `c3866a504a01db25ac2cf35d9f75ed1699f5b359` | Continuation/admission preparation | `PREP_ONLY_SUPERSEDED` | Superseded by later Vanda-controlled gates; do not reactivate. |
| #61 | `Stage45F prep: scaling, data expansion, rollout relabel and validation governance` | `#52` -> `2e9d5190c84da8215b665f840b93ded9c69edeed` | Scaling/data/rollout governance preparation | `PREP_ONLY_SUPERSEDED` | Retain for decision history; no current science execution role. |
| #62 | `Stage45E: matched legacy FULL24 seed replication` | `main` -> `9ad05f1bc7e2efc9f0d88674d569d42ef8b11e62` | Matched legacy FULL24 seed science | `SCIENCE_RESULT_NOT_YET_CONSOLIDATED` | Preserve as science evidence; canonical consolidation awaits PI review. |
| #63 | `Stage45E-L1: matched-seed replication and mechanism closure` | `#57` -> `186efc2977e2795bebf0dfe97d6f503a672716de` | L1 matched-seed science | `SCIENCE_RESULT_NOT_YET_CONSOLIDATED` | Preserve lineage and result claims separately from current Validation-E. |
| #64 | `Stage45E-N2: H200 matched-seed replication` | `#56` -> `6d8a7f132e325faea35346b36497ceb76390b2c3` | N2 H200 matched-seed science | `SCIENCE_RESULT_NOT_YET_CONSOLIDATED` | Keep immutable evidence until accepted result record exists. |
| #65 | `Stage45F: Corelab evidence convergence and decision-readiness` | `#62` -> `d080f4e0011e9c2bed4827dc1c6988c761b2ed0f` | Evidence convergence/decision preparation | `SEALED_HISTORY` | Later gate context references convergence work; retain as historical support. |
| #66 | `Stage45E: unified H200 campaign orchestration` | `#57` -> `191120699b8219cbf62be8ffe60dc19755dec218` | Unified H200 campaign orchestration | `SCIENCE_RESULT_NOT_YET_CONSOLIDATED` | Current live campaign base lineage; do not merge or retarget during PI0168. |
| #67 | `Prepare Stage45F ICL ContextV1 adaptation protocol` | `stage45f/corelab-evidence-convergence` -> `e94be15c90f43025bb526b56ee4a982fe4c607af` | ICL adaptation preparation | `PREP_ONLY_SUPERSEDED` | Explicitly outside PI0168 and current Vanda execution; retain as superseded prep. |
| #68 | `PI0146 bounded local device census and model bridge plan` | `stage45f/corelab-local-device-runtime-smoke` -> `d98798cfa79c37a01a48a28771e8673a54c5d57f` | Local device/runtime bridge support | `ARCHIVE_CANDIDATE` | No current PI0168 role; future archive only after PI review. |
| #69 | `Stage45F: bounded local initial-state pilot` | `#56` -> `217ec2525c33bb8115d86773e5e42fe01d6c4c62` | Corelab local pilot and prior handoff provenance | `ARCHIVE_CANDIDATE` | Standby/historical; superseded as formal cross-agent provenance by this PI0168 packet. Do not close now. |
| #70 | `chore/github-connection-smoke-test-20260913` | `main` -> `140074f7cda3de2115906e93a3d0fa7f893e1d42` | Closed connection smoke test | `ARCHIVE_CANDIDATE` | Already closed; retain only as audit history unless PI requests cleanup. |
| #71 | `[Stage45F] PI0154 matched Validation-E generation 1` | `stage45e/h200-unified-replication-campaign` -> `940df1ab6f428fd0260fe836e36d4cd98c6161e3` at issuance | Vanda-owned Validation-E execution/recovery | `ACTIVE_EXECUTION` | Vanda is the sole active science owner. The final live head movement is explicitly recorded below; Corelab did not edit it. |

The new PI0168 branch/Draft PR is the current `ACTIVE_SUPPORT` packet. Its PR number is filled into the manifest and GitHub handoff after the first audit commit creates a non-empty diff.

## Navigation drift

The live repository and the canonical documentation do not point to the same research stopping point:

- Live `origin/main` at audit time: `607c7b734ba88ee0215a432aa8d2f576da9e77b0`, `Remove accidental noop file`.
- `README.md:8-17` still identifies accepted main commit `bfa14bf1b74eae908c0922575a4db671c903b9b8`, Stage44R source `bd0f029...`, and the Stage44 tiny-1 diagnosis.
- `docs/ACCEPTED_BASELINES.md:1-18,42-46,61-68` stops at accepted Stage44R plus Stage45A design-only/HOLD language.
- `docs/RESEARCH_TIMELINE.md` is titled Stage19–Stage45A and has no canonical Stage45B–F operational rows.

This drift is recorded as a navigation risk, not repaired in PI0168. Any future docs update must be a separate PI-approved change after the Validation-E evidence is sealed.

## Future consolidation sequence

1. Keep PR #71 and all source/evidence branches immutable while Vanda completes its authorized work.
2. After PI accepts the Validation-E result or failure seal, bind the accepted raw/evidence hashes and execution provenance into one canonical research record.
3. Extract only portable, model-independent mechanics into the canonical support surface; retain A40/RTX/H200 and Train24/Validation-E runtime differences as separate adapters.
4. Update README/timeline/baseline navigation in a separately reviewed documentation change.
5. Only after PI approval, close or archive candidates while preserving branch/commit provenance. No such action was taken here.
