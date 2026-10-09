# Actual PI0286 correction admission

**C1 NOT_ADMITTED.** The fixed 32 slots closed once: 32 faithful restorations, 28 physical successes, and 15 branches passing the unchanged strict codec. The required count is 24. No C1 optimizer launch or training transfer is allowed; the transfer manifest contains zero files.

| Takeover | Requested | Faithful restore | Physical success | Physical failure | Strict codec pass | Codec failure | Codec not tested |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3 s | 16 | 16 | 15 | 1 | 14 | 1 | 1 |
| 12 s | 16 | 16 | 13 | 3 | 1 | 12 | 3 |
| Total | 32 | 32 | 28 | 4 | 15 | 13 | 4 |

## Codec failures

The unchanged gates are XYZ <= 1e-5 m, relative SO3 <= 1e-4 rad, gripper <= 1e-5, finite valid rotations, and zero native preclip out-of-range values. The 13 failed branches contain 15 failed H40 windows. Preclip and SO3 counts overlap; they must not be added as distinct failed windows.

- **3 s:** only 285118 fails. Its three failed windows trigger both preclip and SO3 gates. Branch maxima: XYZ 2.686345013103164e-8 m; SO3 0.010654909013566369 rad; gripper 0. The failed windows contain 98 preclip out-of-range value occurrences in total.
- **12 s:** 285108, 285109, 285111, 285112, 285115, 285116, 285117, 285119, 285120, 285121, 285122 and 285123 fail. Each has one failed window, with 80 preclip out-of-range values; only the zero-preclip gate fails. Their measured XYZ, SO3 and gripper roundtrip errors satisfy the physical error thresholds. This is still a strict-codec failure under the frozen rule.
- Every failed branch first fails at suffix anchor 0 (original tick 80 for 3 s; tick 260 for 12 s). No XYZ or gripper threshold failures occur. No reported nonfinite or degenerate-rotation failure occurs.

| Codec maximum across tested branches | 3 s | 12 s | Gate |
|---|---:|---:|---:|
| XYZ (m) | 3.1828893298779803e-8 | 2.60770320892334e-8 | 1e-5 |
| SO3 (rad) | 0.010654909013566369 | 6.956590542729907e-8 | 1e-4 |
| Gripper fraction | 0 | 0 | 1e-5 |

The stored audits record preclip counts, not the maximum excess magnitude or affected native channels. Those details are unknown in this metadata-only report. No normalization or gate was changed.

## Physical failures

| Slot | Missing sustained history | Failed terminal hold predicates |
|---|---|---|
| 285113 at 3 s | lift; horizontal transfer | zone 0/41; TCP_clear 0/41 |
| 285113 at 12 s | lift; horizontal transfer | zone 0/41; TCP_clear 0/41 |
| 285114 at 12 s | grasp; lift | zone 0/41; TCP_clear 0/41 |
| 285118 at 12 s | grasp; lift; horizontal transfer | zone 0/41; TCP_clear 0/41 |

All four ran to the original 32 s completion boundary. They pass the other stored terminal hold predicates. Their suffixes were not admitted or codec-tested for training. Full original score fields, restoration predicates and source-result hashes are retained in the JSON and CSV.

## Closed evidence and limits

Collection began 10:48:33 SGT; the collection/CPU assembly/manifest chain closed 11:04:31 SGT. Each slot used one physical pass, with zero quality retries. There are no missing or duplicate slots or duplicate branch states. The correction chain made zero new learner model queries. This report reads existing closed metadata only and performs no additional physics, model, teacher or codec execution.

Bank: `CORRECTION_NOT_ADMITTED`; plan: `CORRECTION_NOT_ADMITTED`; manifest: `CORRECTION_NOT_ADMITTED_NO_TRAINING_TRANSFER`, files 0. The sampler was not run after bank rejection; no actual sampler infeasibility is claimed. Raw evidence and failed branches remain preserved. Learning causality is UNPROVEN; the 28/32 physical rate makes no generalization or source-policy qualification claim.

Local companions: `ACTUAL_CORRECTION_ADMISSION_AND_FAILURES.json`, `ACTUAL_CORRECTION_CLOSED_RECEIPTS.json`, `ACTUAL_CORRECTION_SLOT_DECOMPOSITION.csv`, `ACTUAL_CORRECTION_FAILURES_BY_TAKEOVER.json`, and exact byte copies in `closed_receipts/`.
