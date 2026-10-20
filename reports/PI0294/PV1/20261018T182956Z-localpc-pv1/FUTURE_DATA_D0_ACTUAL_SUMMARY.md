# D0 Existing-TRAIN Qualification: Actual Closeout

Observed: 2026-10-20 13:57:58 SGT. Child `future-data-d0-01` is terminal;
its native lease is CLOSED and CPU reservation is zero.
[Full qualification list, coverage matrix and evidence hashes](FUTURE_DATA_D0_ACTUAL_AND_QUALIFICATION.json).

## Results

The original 44 not-codec-reviewed rows include one independently rejected
raw capture, `PI0294-PV1-TRAIN-F2-030-T2-C04`. That capture remains excluded.
The other 43 completed the original CPU qualification path:

| Existing saved TRAIN subset | Reviewed | Full official-chain PASS | Codec blocked |
|---|---:|---:|---:|
| T1, original 4mm | 16 | 16 | 0 |
| T2, original 4mm | 15 | 13 | 2 |
| T2, alternate 12mm | 12 | 3 | 9 |
| Total | 43 | 32 | 11 |

Every reviewed episode checked all 601 H40 windows. A PASS also includes the
original full raw/image hashes, six real image/label reader-collator anchors,
mixed batch and deterministic repeat. Original v3.0 statistics/decoder/gates
were not changed. These are Teacher-data qualification results, not new GR00T
task successes or model improvement.

The **prospective**, original-reference union would contain 168 episodes,
91 T1 groups, 77 T2 groups, 93 groups overall and 75 paired groups. Relative
to PV1's 139 episodes/77 groups/62 paired groups, D0 provides 29 qualified
original-reference episodes and 16 newly covered groups. This union is only
an eligibility calculation, not an assembled/sealed or authorized training
bank. The three new alternate 12mm PASSes remain separate.

All newly covered layouts are F2/+Y. D0 does **not** repair the direction/family
confound: F0/-Y and F2/-Y remain absent; original-reference T2/-Y remains 16.
The 21 prior qualified 12mm alternatives add zero independent groups beyond
the original 77. In total, the 273-attempt population now accounts for 192
codec/official-chain qualified episodes, 79 codec exclusions and two raw-audit
exclusions. No original FAIL was overwritten or retried.

## Prior 68 Rejections

The exact original saved receipts yield 56 XYZ-only gate violations, 11
SO3-only and one joint violation. All 68 report native inverse decoding clip
components; 52 have clipping in relative-action codec coordinate 2. This is
not a world-Z collision attribution or proof of student-policy causation.
Original receipts lack first-error anchors and physical-stage joins; those
remain explicitly unknown. The actual JSON retains all 68 row hashes and
per-episode gate/channel evidence.

## Resources and Preserved Mainline

The actual reader consumed 381.0051 CPU seconds. Another 360 CoreLab control
seconds and 600 local control seconds were conservatively charged to the
existing PV1 ledger; there is no new resource window. The independent reader
used CPU 19, while B0 had CPU 5 reserved. Original 15 numerical files and 26
assets passed unchanged checks, as did the frozen PV1 139-parent manifest.

Outputs comprise 41,174 files and 1,473,122,191 logical bytes. Existing RGB
payloads were hardlinked, not downloaded or recopied. New exclusive payload
blocks total 3,850,240 bytes; this excludes directory/capsule metadata and is
not a claimed filesystem/quota delta. There were zero new engines, task
starts, integrations, GPU/PBS submissions, neural queries or optimizer updates,
and no unknown effects. Raw data, old reports and current bank remain intact.

The sole `pv1-child09-owner` remained active. At the last native read,
`677540.hopper-m-02` was Q for unavailable H200 GPUs, 128GB, with no new
allocation effects. PV1 remains B0 12/48 (8 successes, 4 valid failures),
updates 0, settled H200 16,096 seconds, submissions 8/24,
selected/deployment NONE. The existing queued job was not duplicated,
cancelled or modified for D0.

## Next Decision

Continue PV1 B0 then matched A/B; do not wait for a new bank. The
[24-group/48-attempt draft](../../design/FUTURE_TRAIN_DATA_D1_PILOT_DRAFT_20261020.json)
and [protocol](../../../../docs/protocols/2026-10-20_FUTURE_TRAIN_DATA_D0_AND_PILOT_DRAFT.md)
cross appearance/family, direction and distance. **D1-D4 remain NOT_AUTHORIZED**;
new geometry, physics, statistics, sampling or training needs its own approval.
The next useful data expansion addresses the missing cross-strata, rather
than adding more F2/+Y variants.

Verification: four focused negative/metadata tests pass; repository tests
827 passed/14 skipped/58 subtests; PI0294 CPU suite 898 passed/818 subtests.
Native D0 used the already-CI-qualified runtime `2ac48b1a`; the new code here
only reconciles saved receipts and does not change scientific runtime.
