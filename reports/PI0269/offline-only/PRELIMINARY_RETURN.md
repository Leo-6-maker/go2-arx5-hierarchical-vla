# PI0269: completed offline comparison; full feedback verdict pending

This is an offline-only result from all 2904 fixed logical rows: the exact 968 donor outputs plus 1936 new final-policy outputs. Both arms completed 4096 updates before comparison. Four complete Trainer HF readbacks and the matched consumed-input audit are closed. No new model, gradient, rendering or teacher call was made by this analysis.

Each saved record was matched to its original card and the common source row; donor records additionally matched the sealed exact-reuse member hashes. All tensor hashes were checked. Independent physical metrics agreed within 1e-6. Native Euler/decoder replay and full task/retention scoring remain pending the complete final report. This is not a completed qualification or a substitute for the 192 main /96 component panel.

## Physical errors and safety outcomes

XYZ values below are millimetres; SO3 is radians. Prefix covers the actual executed indices 0..8; suffix covers 9..39. All fixed denominators and rejected rows are retained.

| Point | Panel | Rows | REJECT | Nonfinite | Median H40 XYZ | Median prefix XYZ | Median suffix XYZ | Median H40 SO3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ROT2048 | FIT |224|0|0|9.940|3.760|11.113|.036871|
| UNIFORM4096 | FIT |224|0|0|7.847|2.622|8.750|.017511|
| PREFIX4096 | FIT |224|0|0|7.468|2.322|8.353|.018446|
| ROT2048 | FACTOR |304|0|0|10.977|3.576|12.365|.037925|
| UNIFORM4096 | FACTOR |304|0|0|9.294|2.533|10.464|.022000|
| PREFIX4096 | FACTOR |304|0|0|9.106|2.441|10.266|.020922|
| ROT2048 | LEGACY |440|0|0|15.121|3.507|17.100|.046878|
| UNIFORM4096 | LEGACY |440|8|0|17.085|4.244|19.178|.070897|
| PREFIX4096 | LEGACY |440|0|0|17.587|4.581|19.770|.067752|

The preregistered temporal fit contrast passes: paired median PREFIX-minus-UNIFORM prefix XYZ is **-0.220427 mm for M1 (112 rows)** and **-0.295505 mm for M3 (112 rows)**. FACTOR/LEGACY refusals do not increase relative to UNIFORM. These paired medians are not differences of marginal medians. Prefix weighting is the combined pose/increment treatment; this does not isolate a position-only effect or establish a unique cause.

Both continuations improve the genuinely consumed FIT prefixes relative to the donor. Both regress on the original LEGACY XYZ and SO3 medians; therefore neither meets the required legacy-retention condition for a scoped reliable virtual candidate. PREFIX has fewer legacy refusals than UNIFORM, but that does not remove its donor-relative physical error regression. Keep executing every frozen main and component condition; no candidate/loss/noise change follows from this result.

The subgroup CSV retains task/command/nominal-recovery/noise and factor axes with their fixed denominators. The full row CSV retains family, source row, signed direction where defined, teacher-hold flags, residual motion and prefix/suffix errors. Zero teacher displacement is not assigned a directional success.

## Live execution snapshot

At 2026-10-03T11:57:51Z, the actual finite driver PID3611282 was live. UNIFORM main64/64 and both offline968/968 were complete; PREFIX main1/64 was complete and its next episode was active. Component diagnostics had not started. Journals showed8479 actual attempts,33916 recorded heads,66 starts and6544 observation pairs; this is a mid-flight snapshot, not final budget accounting. Driver/source/cards were preserved.

## Completed UNIFORM main64: initial achievement does not imply retention

CPU recomputation on the complete donor/UNIFORM saved arrays reproduced every original score; independent M3 entry checks also matched. No new neural query was needed. Both points are full64 safe with zero gate refusals, dangerous collisions or execution failures. Original success and reliable retention remain different endpoints:

| Point | Task/backend | Original success /16 | Original command0/1 /8 | Reliable /16 |
|---|---|---:|---|---:|
| ROT2048 | M1/E0 |11|7,4|0|
| ROT2048 | M1/E1 |11|7,4|0|
| ROT2048 | M3/E0 |0|0,0|0|
| ROT2048 | M3/E1 |0|0,0|0|
| UNIFORM4096 | M1/E0 |15|7,8|0|
| UNIFORM4096 | M1/E1 |16|8,8|0|
| UNIFORM4096 | M3/E0 |0|0,0|0|
| UNIFORM4096 | M3/E1 |0|0,0|0|

UNIFORM improves M1 original success22/32 to31/32, but reliable task success remains0/64 and M3 remains0/32. This is a valid partial quality result, not a reason to cancel PREFIX or the fixed96 diagnostics. PREFIX main results, component starts, complete native Euler/codec replay and the final combined verdict are pending. Exact per-case scores, source hashes and tick-level retention records are preserved in `DONOR_UNIFORM_MAIN_CPU_PRELIMINARY.json`.

## Sources and limits

- `OFFLINE_ONLY_PRELIMINARY.json`: all per-row numeric/hash checks, exact panels and paired contrast.
- `OFFLINE_SUBGROUPS_PRELIMINARY.csv`: all fixed subgroup results.
- Corelab `/home/liuyu-nfs/Projects/go2-arx5/PI0269/evidence/offline-only/OFFLINE2904_PRELIMINARY.csv`: full2904-row saved-array comparison; its identity is in `ARTIFACT_IDENTITIES.json`.
- `LIVE_EVALUATION_PROGRESS.json`: actual counts at the stated timestamp.
- `DONOR_UNIFORM_MAIN_CPU_PRELIMINARY.json`: all128 completed donor/UNIFORM main scores and independently checked entry/retention outcomes; PREFIX pending.
- `scripts/pi0269_execution/offline_saved_summary.py`: CPU-only analysis, reusing frozen metric/prefix helpers and forbidding neural/optimizer calls.

The donor keeps its original PI0268 card hash; its source rows are identical to the PI0269 card. An initial CPU helper incorrectly expected the newer card hash for donor records, stopped before producing a result, and was corrected to verify the original hash plus sealed reuse members. This incurred zero model queries/updates and did not change a card or stored result.

Final closed-loop, component, native CPU replay, same-tick analysis, videos and new result archive readbacks remain required. Old FAIL and WBC PARTIAL remain. Formal selected/deployment remain NONE; these development scenes and mathematical executors do not prove blind generalization, WBC, physical or real-time success.
