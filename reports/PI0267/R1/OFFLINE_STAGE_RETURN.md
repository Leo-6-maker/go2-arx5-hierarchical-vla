# PI0267 R1 — complete fixed offline results; closed-loop study remains active

All2088 logical rows are complete: donor/N4096/R4096 each696. Donor's440 old outputs were reused;1648 new queries were made. Independent numerical replay covers all2088 metrics and all1648 saved native four-step Euler/noise/decoder outputs. Every output is finite. No new model, gradient or physics query was used for replay.

## Reference-window results

Median XYZ RMSE in millimetres; REJECT is the original unmodified gate. Each new-domain column contains128 windows; each legacy column contains440 rows. New arclength-teacher and old time-paced-reference errors are separate diagnostic contracts.

| Point | New nominal mm | REJECT/128 | New recovery mm | REJECT/128 | Legacy mm | REJECT/440 |
|---|---:|---:|---:|---:|---:|---:|
| donor |12.042|0|14.820|0|7.350|0|
| N4096 |9.274|0|12.650|0|16.017|0|
| R4096 |13.941|0|519.739|86|19.987|128|

The aggregate N improvement must not conceal task differences. On nominal M1 command0/1, N median errors are6.174/4.415mm versus donor10.007/7.951mm. On nominal M3 they are21.679/19.722mm versus donor17.018/17.874mm. On recovery M3 command0, N is26.263mm versus donor15.990mm.

R's recovery degradation spans both tasks and commands:

| Recovery subgroup | Rows | Original REJECT | Median XYZ RMSE mm |
|---|---:|---:|---:|
| M1 command0 |32|22|525.159|
| M1 command1 |32|20|376.391|
| M3 command0 |32|22|714.638|
| M3 command1 |32|22|573.482|

These valid negative outputs are retained. No intermediate checkpoint, extra updates, new noise or recipe is substituted. Offline reference errors and gate decisions do not establish the original closed-loop candidate qualification. All192 frozen virtual episodes and32 secondary WBC conditions remain required.

## Replay and source boundary

- `evidence/COMPLETE_OFFLINE_CPU_REPLAY.json`:2088 numerical metric checks and1648 original-noise/four-Euler-step/decoder checks. The earlier1392-point metric receipt is reused with unchanged record/tensor hashes.
- `evidence/OFFLINE2088_NUMERICAL_ROWS.csv`: complete row indices, point/domain/window/family/task/command/arclength/variant fields and numerical failures. Full raw gate records remain in the source cells.
- `evidence/OFFLINE_TASK_COMMAND_SUMMARY.json`: all fixed new-domain task/command subgroups, including failures.
- `evidence/BATCHED_DECODER_CPU_RECONCILIATION.json`: the initially failing CPU comparison omitted the original bridge's final float32 conversion. The actual full processor and correctly typed CPU output exactly reproduce the saved output. Only the CPU auditor was corrected; original predictions, decoder, metrics, gate and comparison tolerance were unchanged.

Training integrity, four complete Trainer PUBLIC readbacks and the explicit cross-host label-byte deviation remain documented in the earlier training return. Neither that deviation nor the reporting dtype correction is relabelled as a model-quality PASS.

## Separate physical evidence

The first two fixed TRAIN expert M1 controls completed40s with recorded success. The first M1's2001 states passed independent full-tree FK and shared-scorer replay. The first expert M3 command0 completed40s with valid `MISSING_ENTRY`: installed-reference entry5.180s, achieved broad readiness5.600s, no achieved narrow entry. Same-tick tracking RMS9.766mm, p9513.009mm. This is a physical executor-transfer result; no entry tolerance or WBC setting is changed. Remaining fixed controls continue.

Primary virtual episodes are `VIRTUAL_EEF_KINEMATIC / NOT_WBC / NON_REALTIME_LOCKSTEP`. Expert controls are not learned-policy successes. This return closes only the offline lane, not the whole goal or a candidate qualification.

`selected_checkpoint=NONE`; `deployment_selected_checkpoint=NONE`.

## Expert-only secondary update — 2026-10-02 10:18 UTC

All eight fixed expert WBC conditions are now VALID and independently replayed. Three M1 conditions meet the existing scorer; three M3 conditions fail the task criterion with no acquired observable entry; family005 M1 command1 and M3 command1 stop on the conservative local H40 `IK_RESIDUAL` witness. The latter does not prove global unreachability. No tolerance, teacher or WBC setting was changed.

Full-tree FK replay across all eight actual root/DOF histories gives maximum XYZ difference 4.353e-16 m and SO3 difference 3.799e-16 rad. Shared-scorer replay and independent M3 entry reconstruction agree with the saved results. Eight actual physics starts consumed 13208 control ticks, 52832 simulate calls and 26400 actor calls; the recorded simulated duration is 264.159994 s. CPU replay added no model query or physical step.

The three recorded successes use the original criterion-hit rule. Their final XYZ errors are 30.903, 30.957 and 23.977 mm; they must not be described as accurate terminal tracking for the whole 40 seconds. Existing movement/hold partitions and final errors are retained without changing the scoring rule.

`evidence/EXPERT_ONLY_SECONDARY_CPU_REPLAY.json`, `EXPERT_ONLY_SECONDARY_VERDICT.json` and `EXPERT_ONLY_SECONDARY32.csv` preserve this intermediate snapshot: eight completed expert conditions and 24 explicitly NOT_RUN model conditions. `SECONDARY_PARTIAL` at this snapshot means the model conditions are still pending, not a completed secondary verdict. Primary virtual execution continues; the original 24 model WBC conditions remain required.
