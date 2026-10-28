# DATA-COV1 P24: Closed At First-Four Gate

Status: `CLOSED_FIRST_FOUR_QUALIFICATION_GATE_NOT_MET`. This is an actual
negative collection qualification, not a released training bank or a policy
evaluation. The original T0, counters, failures and raw records are preserved.

## Actual Coverage

| Fixed first attempt | Task result | Final data status |
| --- | --- | --- |
| AP0 / -Y / NEAR / 000 / T1 | Unknown: original geometry conflicts with frozen scorer input | ENGINEERING_INCOMPLETE; retained in denominator |
| AP0 / -Y / NEAR / 000 / T2 | Full 32-second success; independent raw PASS | CODEC_REJECT |
| AP2 / -Y / FAR / 000 / T1 | Full 32-second success; independent raw PASS | ADMITTED per parent |
| AP2 / -Y / FAR / 000 / T2 | Full 32-second success; independent raw PASS | ADMITTED per parent |

The two admitted parents form one actual geometric task pair. Each has641
actual two-camera observations,601 checked H40 windows and the unchanged
official CPU processor/collator chain with deterministic repeat. A mixed
first/terminal-anchor batch within one parent is not a cross-parent bank
qualification. No model forward or training was performed.

Only two of the planned12 crossed-factor strata have physical attempts.
The other44 P24 conditions are explicitly NOT_RUN; P72/P144 were not started.
Do not report3/3 known task successes as general Teacher reliability,4/4 task
success, GR00T improvement or full Go2/WBC transfer. The first unknown result
is neither a valid task failure nor a successful demonstration.

## Why Expansion Stopped

The frozen first-four requirement is at least3 complete data admissions,
with both tasks represented. Actual admission is2/4. The AP0 T2 trajectory
succeeds physically, but its first failing window is anchor209/action31,
reference12s: SO3 roundtrip error0.0001381733rad exceeds0.0001rad; normalized
rotation channels4/6 are inverse-clipped. This co-occurrence is not a
counterfactual cause test. XYZ, gripper and relative-SE3 checks do not fail at
that point. The official data chain was correctly skipped after this codec
rejection. No thresholds, statistics, decoder or labels were changed.

All11 child leases and the independent DATA control lease are CLOSED,
all registered workers are terminal, reservations are zero, unknown effects
are empty. The failed gate, not resource exhaustion, stops expansion.
The recorded cost is5057.476842139 CoreLab CPU seconds including conservative
control charges,2700 conservative local CPU seconds,78 static and356 dynamic
constructors,64000 live plus64000 shadow ticks. The own-tree disk measurement
is338321408bytes; it is not incremental filesystem allocation or an HF copy.

## Availability And Boundaries

All four raw attempts, negative evidence and two qualified parent exports
remain in registered private storage. No stage bank READY, cross-parent bank
qualification, HF dataset release or downstream-consumption PASS is claimed.
The two individually qualified parents are not automatically injected into
PV1 or promoted to a substitute mini-bank. PV1's139 parents and sampler remain
unchanged. No checkpoint was deleted.

[All48 planned conditions](DATA_COV1_P24_ALL_ATTEMPTS.json),
[12-stratum coverage and actual event/codec evidence](DATA_COV1_P24_COVERAGE_AND_EVENTS.json),
[first-four gate](DATA_COV1_P24_FIRST_FOUR_GATE.json),
[P24 gate](DATA_COV1_P24_STAGE_GATE.json),
[native closure and identities](DATA_COV1_P24_TERMINAL.json),
[receipt reproducer](DATA_COV1_P24_REPRODUCER.json).

The next authorized main-line work remains PV1 B0 and matched A/B, not more
DATA-COV1 attempts. A successor data experiment must first explain the saved
codec failure and obtain an explicit protocol/GO; this receipt does not
authorize a changed codec, normalizer, Teacher, new seed or training.
WBC, plant, S1, scoring and EEF10 are frozen. selected/deployment=NONE.
