# CR1 Saved-TRAIN Codec Review

## Decision

**RANGE_SUPPORTED**, with independent TRAIN diagnostic support. This is an offline data-representability result, not evidence of policy learning, S1 safety, task success, or whole-body transfer. `selected/deployment=NONE`.

Authority: `e7593fdde645b2f22fc4d06431f98c2da180f855`. Runtime: `8e2097ed2513aecc205d71532f49b72f5cf0b08f`; exact GitHub Actions 39412546118 / job 118593595499 succeeded. The frozen WBC, plant, gripper-PD, S1, scoring, state/gripper normalization, and PV1 bank remain unchanged.

## Reproduction and Mechanism

The unchanged official path exactly reproduced anchor 209 / action 31, reference time 12 s: SO3 error `0.00013817330386000284 rad` exceeds the original `1e-4 rad` gate. Relative-transform residual is `1.6653345369377348e-16`; channels 4/6 are inverse-clipped. All 601 original windows were checked, plus 14 predeclared diagnostic windows including an original PASS control.

At anchor 209, float64 transport with the original clamp still fails. Float32 without clamp passes this isolated window. Its trace/arccos maximum rounds to zero, while the atan2 cross-check is about `2.24e-8 rad`, not mathematical zero. There are no active isclose-near-constant EEF9 channels. No-clamp arrays are diagnostic only, never a production decoder. These controls support clipping loss for this case; they do not prove every failure has the same mechanism.

## Frozen Population and Paired Evidence

191 eligible, original 4 mm, full-safe-success TRAIN parents form 96 geometric lineages: 141 parents FIT, 50 DIAGNOSTIC, grouped before windows. Three previously opened DATA-COV1 parents are KNOWN_CHALLENGE only. All 139 old admitted parents are checked as an overlapping retention role, not an additional independent population. The 82 exclusions from 273 original attempts have sealed reasons; codec FAIL was not an exclusion criterion.

| Role | PASS to PASS | FAIL to PASS | PASS to FAIL | FAIL to FAIL |
|---|---:|---:|---:|---:|
| FIT | 125 | 16 | 0 | 0 |
| DIAGNOSTIC | 43 | 7 | 0 | 0 |
| KNOWN_CHALLENGE | 2 | 1 | 0 | 0 |

All FIT windows pass. The seven recovered diagnostic parents cover **six lineages disjoint from FIT**. All 139 retention parents remain PASS; no diagnostic or checked-retention regression occurs. Main-archive R0 passes total 168, matching the prior D0 original-reference union; 23 main-archive FAILs recover. The known challenge adds one recovery, separately reported rather than presented as blind generalization.

| Maximum / Count Across 194 Parents | Original R0 | Isolated R1 |
|---|---:|---:|
| XYZ error, m | 0.00040843984618 | 1.08455182335e-8 |
| SO3 error, rad | 0.00076094300185 | 4.71216091539e-8 |
| Gripper error | 0 | 0 |
| Relative SE3 component error | 8.88178419700e-16 | 8.88178419700e-16 |
| Inverse-clipped action components | 76 | 0 |
| State preclip components, unchanged | 1424 | 1424 |

Each variant checks **116,594 official H40 windows** on identical arrays. Each also reads six real dual-camera anchors per parent, totaling 1,164 samples per variant, passes the original masks/collator and deterministic repetition, and passes a mixed batch from two distinct parents. This is actual CPU data-chain consumption, not metadata qualification. The paired JSON includes task/direction/lineage breakdowns, first failures, error quantiles, clip amplitudes, and native event counts. Events come from saved physics, not Teacher labels or new integration; overlapping event categories are not summed or counted as independent demonstrations.

## Scope and Compatibility

Only `new_embodiment.relative_action.eef_9d.min/max` changes. The single presealed PAD5 formula uses 84,741 FIT windows, with no diagnostic/challenge refit or padding search. A new processor instance reads back exactly the intended active parameters; the original instance remains unchanged. State, gripper, relative transform, clamp, cast path, rot6d reconstruction, and all original error gates stay fixed. Inactive means/std/quantiles and other keys are inherited, not reestimated.

Range spans expand by about 1.1000 to 1.1157 times; corresponding normalized signal scales are about 0.8963 to 0.9091 of their old values. The candidate JSON reports all per-channel bounds, absolute changes, and same-z physical shifts. Changing ranges changes the physical action corresponding to a fixed normalized output. **Old CORR weights plus this decoder are not the old B0 baseline.**

The reused scanner's legacy `statistics_refit=false` means the scanner itself does not fit ranges during its call, not that R1 was never fitted. Likewise `native_decode_unchanged=true` describes the decoding algorithm, not unchanged R1 parameters. The paired JSON explicitly disambiguates these fields and preserves the native result hash.

No DEV/CONFIRM/LOCKBOX data were read. Inherited generator labels are lineage metadata, not reads of similarly named evaluation splits. Original statistics have historical TRAIN exposure; this diagnostic and retention check are not new policy generalization tests. BF16, flow loss, neural compatibility, closed-loop safety, learning, and Go2/WBC transfer remain untested. A codec PASS cannot reverse DATA-COV1's old 2/4 gate or create a BANK/READY receipt.

## Unique Scientific Next Recommendation

Propose a **separately authorized, small new-version bank/task pilot** using this fixed candidate and frozen safety rules. Only after data-chain and task qualification should a matched training comparison be proposed. Do not resume the closed DATA-COV1 remainder or change PV1.

PV1 remains independent, with B0 DEV48 still required before its original matched A/B training. During this CR1 run, child11/678230 terminated before model loading because fresh project authentication failed; B0 remains 14/48 and optimizer updates remain zero. Its original reservations and leases are pending settlement, not falsely reported as released. See the [separate engineering receipt](../../PV1/20261018T182956Z-localpc-pv1/PV1_B0_11_FRESH_AUTH_PREMODEL_TERMINAL_PENDING_SETTLEMENT.json). CR1 did not repair, retry, or alter that job.

## Audit and Closure

Seven fixed artifacts accompany this report. Input hashes, exact source/CI, effective parameters, split, rule seal, numerical summaries, and stage-array hashes are public; raw RGB and stage arrays remain in registered private storage. The final lease/resource state is in `CR1_RESOURCE_FINAL.json`.

Independent JSON-only reconciliation: `python -m scripts.pi0294.cr1.review <report-directory>`. All 194 IDs, 139 retention members, 601 windows per parent, 388 parent-reader receipts, and two cross-parent mixed receipts reconcile; FIT/DIAGNOSTIC lineage sets are disjoint. Share with the explicit offline-only and historical-exposure caveats above. No GPU utilization, new policy success rate, or task success rate is inferred. A closed CR1 scope is not automatically reopened.
