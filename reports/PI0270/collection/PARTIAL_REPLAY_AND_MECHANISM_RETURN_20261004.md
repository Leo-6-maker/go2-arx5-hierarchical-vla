# PI0270 actual partial collection replay and learner-state mechanism

Authority: release7a6eea5 / Issue #11 canonical5969189182. This uses saved completed work, without new neural, teacher, optimizer or render calls. It neither changes the recipe nor substitutes for full data admission or AGG evaluation.

## Completed CPU work

Finite CPU batch PID3924701 started at2026-10-03T18:26:12.354679Z and ended normally with its completed-batch receipt and terminal log. It processed13 completed families,104 complete original-start cases and10400 saved native-S4 outputs. Every four-step BF16 Euler transition, final raw output, original no-clamp decoder, registered FP32 noise and actual BF16 initial-noise bytes were checked. Decoder maximum absolute difference was0. Saved observation state/timestamp and result/prediction/observation identities were checked. No model forward, optimizer, teacher or Renderer call occurred.

This is arithmetic replay of the recorded neural trace; it does not recompute the learned network, independently validate every RGB embedding, or replace the later complete trajectory/V3/reliability replay. Per-case evidence remains on Corelab and is indexed by SHA256 in `COLLECTION_S4_REPLAY_FAMILIES_13.json`. Existing valid per-case checks are reused when later families close.

Receipt SHA256: `a8d2833b17fe81602a90e14597b3e41c5fbf4d8465bec08d4294f1a677f987c7`; process-start SHA256: `27a0576ae4bb60d37cf41331dfffed1c276d14e8c15aee1c298445ee76718dd8`.

## Actual closed label support, separate from pending strata

The subsequent14-family snapshot contains4224 closed fixed proposals:4223 eligible,1 excluded for VISIBILITY,2560 still pending of the fixed6784. The exclusion is `original/pi0265-family-015-M1-c1-E0/086`; it is retained in the denominator and will not be replaced. Poor prediction, reversal and large error remain eligible when the independent data criteria pass. This partial snapshot does not declare final96-cell admission.

Teacher first-pose position discrepancy has maximum5.5511e-17m; rotation-matrix element discrepancy maximum1.4655e-14. These are consistency checks against the same measured starting state, not evidence of a unique root cause or overall task correctness.

Actual affine normalized ranges are state[-5.4680,7.4545], action[-29.1258,36.4056]. These finite values were preserved under the frozen ROT map, without clipping, refitting or a new loss. Their magnitude alone does not prove an optimization failure; the real canary and training must inspect finite losses/gradients under the unchanged contract.

There are1575 projected-terminal labels and ZERO zero-prefix-displacement labels in this snapshot. Projection to the teacher endpoint does not mean a zero-action hold: the existing exponential correction from the measured state remains active.

## Reference-generation discrepancy on the exact same measured inputs

For original-start M1 proposals whose teacher projection is terminal:

| Executor | Proposals | Model displacement opposed to teacher displacement | Median prefix XYZ discrepancy | Paired median teacher reference goal-error change at0.4s | Paired median model reference goal-error change at0.4s |
|---|---:|---:|---:|---:|---:|
| E0 |723|594/723|40.439mm|−60.629mm|+2.681mm|
| E1 |699|645/699|33.774mm|−49.907mm|+2.935mm|

The error-change columns first subtract each row's measured-current goal error from that same row's teacher/model reference error at H40 index8, then take the median. They are not differences of marginal medians. These are GENERATED REFERENCE endpoints, not installed-reference tracking, achieved execution or task success. The discrepancy supports investigating recovery on visited states, while representation, numerical fitting and stochastic policy dynamics remain alternatives. No AGG treatment effect exists yet.

For original-start M3, the saved same-state teacher-displacement cosine is negative in370/896 E0 and347/896 E1 proposals. This diagnostic is not the original ordered-arc/reverse-tolerance verdict and does not relax V3. Proposal windows and executor variants are correlated observations, not independent scene replications or blind TEST.

Source diagnosis SHA256: `084a48b9631b25bef3b52ca21ff22503e4789f87b5568de4d2739824142c5ee2`; paired-reference receipt SHA256: `648f4f851144d14a29c39c7ab66718371a66975745c1a4e77e0d4e35ac19ee08`.

## Latest real state and continuation

At2026-10-03T18:38:33.657232Z (02:38 SGT), actual `ps` confirmed collector3810700 and finite label/codec owner3854541 live with unchanged commands.120/192 collection cases,4224 codec records and14 completed family receipts existed. Counters observed:12052 inference attempts,48207 heads,121 starts,12052 frame pairs and4480 teacher calls. The head count is reported literally: the snapshot can intersect an in-flight four-head query; no fourth head is fabricated. The replay above covers completed valid outputs only.

Finish the fixed192 cases and6784 proposals, inspect real96-cell/NEW_FIT admission, freeze the single4096 recovery mapping, then proceed with actual new-data archive/readback and Hopper staging/admission, real8-vs4+save+restore4 canary and AGG4096, complete2048/4096 Trainers and both HF readbacks. Complete all fixed2904/384/192/96 logical panels, new CPU replay and original qualification/legacy/reference/achieved analyses, the fixed all-eight-case failure-preserving video, all result archives/readbacks and the one terminal Issue #11 return. Explicit insufficient-data/correctness stops retain their actual evidence without extra samples.

Goal ACTIVE; selected/deployment NONE. No healthy restart, repeated valid query, new scientific branch, cleanup, monitor/mail, WBC/physics, PR77 pointer or merge.
