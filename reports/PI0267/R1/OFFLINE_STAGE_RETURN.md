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

## Native lifecycle recovery — 2026-10-02 11:46 UTC

The first virtual process ended with signal11 after preserving47 valid donor results. Start48 (`pi0265-family-033-M3-c1-E1`) reached native renderer construction but produced no observation, prediction or physics step. The completed records contain4700 virtual queries; cumulative inference accounting at the failure is6348 attempts/25392 heads, including1648 offline queries. All48 start entries remain. The exact native fault is unavailable without kernel/core access; this is an engineering lifecycle failure, not a valid model-quality result.

Recovery uses a fresh native process for each of the SAME eight fixed families, in the original donor/N/R order. A narrow CLI family selector bounds repeated native environment construction to eight conditions per process. `read`, `advance`, `message`, `language` and `run_episode` have identical ASTs; the renderer, worker, scorer, gate, data and noise are unchanged. The runnable native self-check verifies exact family selection/rejection, prefix execution and the original timestamped gate with zero queries/renders/physics.

The new entry is isolated at `R1/runtime/virtual_family/pi0267_virtual_episode.py`, SHA256 `9902baf2df3fff4e5a1ff52383dd0d522282305cb68223ad6797baabdb2a63ca`. Original `code/pi0267_virtual_episode.py` stays at `36165145558c56ea025ab87af8413164e89c6c33f1f7866a51d87586325044b9`; secondary imports and their seal remain unchanged. The earlier quality seal and log are retained. Two launcher seal-path preflight errors occurred before any recovery process/query/start; the actual seal is under `R1/design`.

The restored first required query is actual attempt6349 with original key `M3_E1_000`. Its RGB/state/TCP/time arrays and processed-condition identity exactly match the corresponding saved E0 initial observation from the same family and command; its distinct E1 noise matches the frozen registry. Model binding is unchanged. Hashes of all47 completed results/trajectories remain exact. These checks add no model query or render. The interrupted start remains counted and only missing conditions continue.

Evidence: `VIRTUAL_NATIVE_EXIT_20261002.json`, `VIRTUAL_FAMILY_LIFECYCLE_CHECK.json`, `VIRTUAL_RECOVERY_INPUT_IDENTITY.json` and the amended `QUALITY_EXECUTION_FILESET.json`. This is verified execution recovery, not completion of192 virtual cases or a quality PASS.

## Completed donor virtual baseline — 2026-10-02 12:18 UTC

All64 donor virtual conditions are VALID and complete, consuming6400 actual queries. The original V3 groups are:

| Task/backend | Success/16 | Command0/8 | Command1/8 | Qualification |
|---|---:|---:|---:|---|
| M1/E0 |6|2|4|FAIL|
| M1/E1 |7|3|4|FAIL|
| M3/E0 |0|0|0|FAIL|
| M3/E1 |0|0|0|FAIL|

The baseline has13/64 recorded task successes and zero recorded dangerous collisions. All four groups fail the original13/16 and6/8-per-command conditions. V2 side scores have the same group totals. This baseline result does not cancel or reconfigure N/R; N4096 has started automatically under the same fixed card.

`evidence/DONOR64_VIRTUAL_CPU_REPLAY.json` verifies all64 trajectory/result hashes and shared-scorer replays, plus independent entry geometry for all32 M3 trajectories. This check made zero queries/renders/physics steps. Full saved prediction noise/Euler/codec replay remains part of the final192-case report; this receipt does not claim that larger scope is already complete. Virtual motion remains `NOT_WBC / NON_REALTIME_LOCKSTEP`, and no candidate is selected from this partial comparison.

### Donor raw result archive

PUBLIC fixed revision `fe7f45d3faeeec0e25a8f7df909ac0bcc966e58c`, repo `leoyuer/pi0246-gate-pass-campaign`, prefix `evidence/PI0267/R1/final-results-20261002/donor`, has passed anonymous whole-file and all33776 internal-member readback. Source bytes:1116150256; tree SHA256 `afbcb47d1999a615269128f0b6d821ba99c8b406220065c87023983013edef29`. Receipt: `evidence/RESULT_ARCHIVE_donor.json`. Local originals remain retained.

Completed point partitions may now archive while the remaining model points run. `PARTITION_ARCHIVE_ADMISSION_CHECK.json` verifies the exact64 virtual conditions and696 offline rows, rejects incomplete N and global-summary partitions, and confirms unchanged packaging/upload/full-readback code. This archival boundary change makes no quality query and does not change whole-study completion or qualification.

## Coordinator continuation at a completed-family boundary — 2026-10-02 13:14 UTC

After the prior local SSH transport reset, the remote coordinator continued several families and then disappeared after completing N family031. The coordinator, native renderer and model worker were all confirmed absent before continuation. No coordinator stderr was available for that exit, so its exact cause remains UNKNOWN. All96 completed valid results (donor64, N32) and their trajectory hashes were checked and preserved. Their actual9479 virtual queries plus1648 offline queries reconcile to11127 attempts and44508 heads;97 virtual starts include the earlier zero-query native failure. The first preflight incorrectly assumed100 queries for every completed case and stopped before launch; it was corrected to sum actual result records.

The unchanged finite family sequence now runs with a detached coordinator and a durable remote log. It resumes at start98, `N4096/pi0265-family-032-M1-c0-E0`; the first61 new saved predictions and the live coordinator were verified at13:14 UTC. No completed condition, model query or training update was repeated. Episode/model/gate sources and all scientific budgets are unchanged. `evidence/VIRTUAL_COORDINATOR_TRANSPORT_RECOVERY_20261002.json` preserves the before-state hashes, actual accounting and continuation evidence; the exact coordinator source is retained in the remote audit-code bundle. N remains partial and R remains pending; this is execution recovery, not a final quality verdict.
