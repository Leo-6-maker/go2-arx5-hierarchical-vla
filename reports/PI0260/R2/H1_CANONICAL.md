# PI0260-R2 actual dataset and H1 observability result

Authority: `c5070dc1b6eaff9d3070c92e3a0248ff4c0a83be`; execution source `dccac3d00e1468f677cf9e67221c1c1d92195b2e`.

**Status: H1_OBSERVABILITY_BLOCKED_EXACT_INPUT_FUTURE_ALIAS.**
This is an actual generated-data result. Base/G6000 and the1760-tuple model evaluation have **not** run; the matched model experiment is incomplete. Model quality is NOT_EVALUATED.

## Actual fixed proposal outcomes

| Stratum | Fixed slots | Protected exclusion | Executed | Geometry FAIL | Visibility FAIL | Complete pre/post-codec families | Not evaluated after target |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN |132|1|131|93|8|30 /30|0|
| KIN-VAL |52|0|38|26|2|10 /10|14|

No raw-gate or codec family failure occurred in the selected population. Old DEV4/20 FAIL remains unchanged. The R2 quantity tier is **COMPACT_30_10**. No resampling, refilling, grid/camera/IK/quality-threshold change occurred.

Historical TEST alias reconciliation added six numeric protected geometries. Original slot110 matched `pi0255-family-070 / MT-TEST_COMPOSITION`, including declared full geometry/root/DOF identity, and was excluded without replacement. No old DEV protected-test overlap was found. TEST payloads were not read.

## Completed correctness work

- Same-owner independent saved-array replay:78,982 FK/full-link collision states,26,200 raw gate windows,13,243 new body/wrist pairs; PASS. Original FK/FCL/gate primitives were shared; reference algebra and raster verification were independently replayed.
- Exactly one statistics fit from13,200 raw-qualified active TRAIN windows. No VAL/DEV/TEST fitting, no post-codec refit or refill.
- Actual pinned GR00T StateActionProcessor pose/chunk conversion, with the explicitly authorized unclamped q01/q99 affine map and1e-6 range floor. Model loss/inference bindings were not yet exercised.
- Maximum actual label roundtrip position error `1.0516509392206614e-08 m`; rotation error `3.224803001023867e-08 rad`.
- All160 selected sequences retained original95% coverage, mandatory states, quartiles and gate requirements.17,377 active windows are usable after joint checks.
- Old35 IK failures and four visibility-failed families summarized from saved arrays only; no old solver/render repeat.

## Observable-input failure

Among17,377 usable active windows there are10,288 distinct allowed raw inputs and24,623 same-input window pairs.24,543 pairs have non-identical future labels beyond original label roundtrip tolerances. All40 families are affected.

For one measured pair from **each** family, the actual full processor/collator independently confirmed byte-identical pixel values, token IDs, attention mask, image grid, embodiment ID and normalized state. No model was loaded. The tokenizer files were checked against fixed revision `9ce19a195e423419c349abfc86fd07178b230561`.

Two concrete examples:

- `pi0260-family-042-M1-c0`, t0 versus t0.7s: same RGB/FK/language, future XYZ RMSE difference15.77mm and maximum pointwise difference25.91mm.
- `pi0260-family-026-M3-c0`, t0 versus t0.5s: same input, future XYZ RMSE difference15.25mm and maximum pointwise difference21.16mm.

The fixed residual-tolerant IK can retain exactly the same joint state across several timestamps while the clock-driven desired H40 reference changes. This is within the same command; the text does not distinguish the hidden time.

**The stop concerns the inherited observable-input contract, not failure of the20mm model-quality threshold.** These measurements do not prove that a model cannot meet that looser threshold, nor that the robot/controller is globally infeasible. No new scientific acceptance threshold was introduced; label roundtrip tolerances identify differing labels, not model FAIL.

PI0258's geometry/observability clause and PI0260 section4.2 item5 require resolving conflicting futures under identical permitted observations before learning. R2 preserved these input-quality requirements. Adding clock/phase/source fields, changing reference timing/IK, replacing labels or drawing replacement families would change the authorized scientific setting. None was applied.

H1 natural-image signoff, complete evaluation-readiness, cross-process/recovery admission, canaries and model stages are not claimed. Diagnostic RGB/curve cards were produced from saved arrays only.

## Cumulative usage and preservation

84 original calibration pairs;756 total sequence attempts;92,228 solver calls;25,520 active raw windows;15,377 total production pairs. Model loads/queries/updates, physics/WBC/real hardware:0. No Hopper submission. No email, monitor, cleanup or merge.

Result manifest SHA256: `0cdb4419cdff204f378f288fc60f5f8704e8a6849669b7d40de67f4748bd8567`.
Sealed source files:30,465; source bytes:1,523,681,670.

HF PUBLIC: `leoyuer/pi0246-gate-pass-campaign`, prefix `evidence/PI0260/R2-motion2-r2-20260930`, revision `3d248c1e54f8eb2caf771b067ef48cd995367a12`.
Public member tree: `793a1ff78c3417933202311187fa87daf538304a5fcbfdacd2ea6016066ff571`.
Full anonymous readback: **PASS_PUBLIC_ANONYMOUS_FULL_FILE_AND_MEMBER_READBACK**, recorded in `R2_PUBLIC_RECEIPT.json`. Both public files and all30,466 archive members were read back and checked. The receipt's legacy field `producer_H0_manifest_sha256` binds the R2 result manifest above; it does not claim another H0 admission.

`selected_checkpoint=NONE`; `deployment_selected_checkpoint=NONE`.
Next boundary: **STOP_BEFORE_MODEL_CANARY_FOR_DATA_OBSERVABILITY_REVIEW**. Existing arrays suffice for PI review; no scientific rerun is authorized by this report.

## Immutable delivery links

- [Result and receipts](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/tree/54366ee/reports/PI0260/R2)
- [HF PUBLIC fixed revision: arrays, RGB/depth/masks, codec, alias evidence and diagnostic cards](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/tree/3d248c1e54f8eb2caf771b067ef48cd995367a12/evidence/PI0260/R2-motion2-r2-20260930)

This is the execution canonical for PI review. Data quantity no longer blocks progress; the inherited observable-input contract does. Resolving hidden time / future-reference aliasing would change the scientific setting and was not attempted. No request for routine training GO is implied: the existing training authorization remains conditional on actual H1 admission.