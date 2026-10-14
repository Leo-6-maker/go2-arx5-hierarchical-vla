# PI0294 Actual Closeout

Outcome: PARTIAL_ACTUAL_GOAL_NOT_ACHIEVED. No working dual-task policy has
been proven. Selected/deployment remain NONE; WBC, mobile Go2 and hardware
remain NOT_RUN. The original 09:00 SGT new-work cutoff is closed; the 10:00
absolute deadline is not extended. This report is not a replacement science GO.

## Model

Formal M512 completed 512 updates and 16384 consumed samples on frozen v3.0.
T1/T2 exposure was exactly 8192 each; old T1 accounted for 891/8192 (10.88%).
Every consumed sample used sealed actual stage anchors; prefetch was excluded
and the complete consumed trace hash was verified. The backbone stayed frozen
and 537 trainable tensors changed. Exact full-state engineering resume passed
at the original tolerances within 16 engineering updates, but independent
M512 resume has not run. Normal M4096, intermediate M2048 and conditional
M6144 were not reached; the candidate-selection/extension gates were not used.

The 25,541,023,437-byte local checkpoint and full inventory are preserved.
Final readback found all 16 members present with recorded sizes; this was
metadata verification, not a repeated full-weight hash or neural load.
HF preupload failed with account private-storage HTTP 403. No M512 fixed
revision/READY exists. There are no actual model TRAIN/DEV/confirmation/
regression results or policy videos. Success and safety are unknown, not zero
success estimates or a safety pass. Loss 0.315059 is not a task result.

## Data And Pipeline

All 224 prefrozen new first attempts ended: 215 admitted, 9 rejected and
preserved. T1 TRAIN/DEV were 93/96 and 16/16; T2 were 91/96 and 15/16.
The 112 planned new groups produced 106 complete admitted task pairs and
109 groups with at least one admitted task. Six rejections were static
field-of-view failures in three TRAIN pairs; three T2 rejections exceeded the
independent reconstruction threshold despite safe physical task completion.
No failed seed was replaced and no audit threshold was relaxed.

v3.0 is delivered at private HF READY revision
`6aa94f4f40d791bedd6b4e75489e4ede6504d10d`, with 158 native episodes.
v3.1 includes v3.0 and has 295 native episodes, including 80 unchanged legacy
episodes. All 149048 TRAIN/28247 DEV windows passed complete CPU qualification,
including 54 stratified real tensor checks and native 0/2/4-loader exact
consumption/resume. Five DEV inverse clips and 175 state preclips are disclosed.
The first CPU timeout remains; frozen recovery reused exact bytes without
export or stats refit. v3.1 is not yet HF-delivered and was not used by M512.
TRAIN/DEV group intersections are empty; confirmation remains sealed.

Actual Teacher double-SSH fresh-observation/ACK routing passed for both tasks.
This is not a model route test. The end-to-end learned-policy pipeline is
therefore incomplete, even though real four-worker data production worked.

## Performance And Resources

Latest fixed-source serial/2/4-worker diagnostic throughput was
17.194/31.600/49.171 admitted episodes/hour (1.838x/2.860x speedup).
Measured formal production plus its independent auditors cost 29.164 core-hours
over all 224 attempts, or 0.135647 core-hours per admitted new episode.
Total charged CoreLab cost including overhead/conservative upper receipts was
37.842 core-hours. The original throughput receipt is preserved unchanged.

M512 synchronized optimizer throughput was 0.268837 updates/s, peak GPU
allocation 35,403,078,656 bytes, loader wait 257.308 s (11.70% of elapsed
training). Configured-horizon Trainer speed estimates were excluded. The
frozen profile was microbatch 2 x accumulation 16, workers 0; 4/8 microbatches
were not benchmarked and no global-fastest claim is made.

PBS was 4/4, allocated GPU time 3669 s, engineering updates 16/16 and main
updates 512. All four allocations are reconciled terminal; current native
own-job selection contains no PI0294 live job. Final CoreLab parent/child/own
groups are terminal and all physical/CPU reservations empty. No healthy
unrelated job was cancelled, no new deletion was performed, and no service,
tunnel, fifth submission, second owner or login-node/CoreLab neural work ran.
Registered bytes are not disk-free deltas or a continuously sampled peak.

## Remaining Requirements

Available authorized private HF capacity, explicit additional PBS authority
and valid new execution-window authority are necessary for further science.
Existing credentials must be reused; do not export tokens, pay, make assets
public or delete private HF history. Actual fixed-revision model delivery,
M512 model routing, later learning/DEV selection, core/challenge/regression
behavior, confirmation safety and policy videos remain missing.

Use RESUME.md for the preserved request/source/processor/stats/data identities
and original accounting. The actual recorded main request is bound to
`scripts.pi0294.neural` in the legal Hopper profile and parent allocation;
there is no currently authorized executable continuation command. Do not
reinterpret these partial results as MULTITASK_SIM_WORKING_V1.
