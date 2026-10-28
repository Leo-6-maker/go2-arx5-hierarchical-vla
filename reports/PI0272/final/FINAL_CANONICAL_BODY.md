## PI0272 R1 terminal canonical — fixed fitting COMPLETE, primary DIRECT advantage not established

**PI0272_MATCHED_DIRECT_READOUT_FITTING_COMPLETE — STOP_FOR_FULL_SUPPORT_FEEDBACK_OR_FITTING_DECISION**

Authority: original release `e62e6a469dce1f4d3df5d49b38bad4a3c627e969`, canonical5979236384; phase-specific R1 release `8a2ffbefe0c57536c0a7e340da0616c7987c74f1`, canonical5980876835. Evidence commit **`23695db372cf8ca44fd6e2ace2c6b36657500c03`**, existing `pi0267/executor-decoupled-recovery` branch.

### Actual completion and integrity

- Both **S4FIT1024 and DIRECT1024** completed from the same AGG4096 weights with fresh optimizers and the frozen296 consumed conditions. Actual all4096 main microbatches match CPU input/label identities and paired native epsilon/tau. No new data or supervision.
- Both real continuous8 versus4+full-save/fresh-resume4 canaries PASS:1030 model and1413 optimizer tensors, maxdiff0.0, exact scheduler/RNG/cursor. Actual generated gradients cover4 versus1 nodes; same-phase parity raw/decode maxdiff0 and original gate agreement.
- Explicit training query **FP32**, native epsilonBF16, original mixed parameter storage/autocast; quality model-only and query **BF16**. DIRECT zero query follows explicit phase feature dtype. FP32 physical loss/decode unchanged; no `.training` phase inference or cross-phase equality claim.
- Same471 trainable/559 frozen tensors, physical/native objective,1024 schedule, LR1e-5 and accumulation2. Native regularizer retained in both arms. DIRECT is one zero-query/bucket0 readout through the same modules with direct trajectory interpretation; no new head, Euler, target/prefix, parameter reset or precision sweep.
- Four complete512/1024 Trainers, **25 files each**, have full fixed-revision readback PASS including optimizer/scheduler/RNG, processor/statistics and effective-mode provenance. Corelab received model-only files; unique recoveries were not deleted.
- Exactly **2148 new quality calls /6228 heads**, no technical neural retries. S4FIT1168 support+192 guard, DIRECT0 296, DIRECT1024 296+192 guard+4 exact repeats. All four repeats preserve native/output bytes and generation RNG. **1360 old AGG outputs reused**, giving3504 non-repeat metric rows.
- CPU replay PASS on all2148 new outputs: S4 saved Euler/decode, DIRECT raw/decode (no fictitious Euler), unchanged original gate and independent FP64 metric arithmetic. All192 K8 banks complete; no reduced K or replacement draws.

### Scientific verdict: matched fitting helps both, no established primary DIRECT superiority

| M3 terminal policy | Outputs | Median prefix XYZ mm | Median prefix SO3 rad | Wrong direction | REJECT |
|---|---:|---:|---:|---:|---:|
| AGG4096/S4 reused |48|8.954|.006020|35/48|0|
| DIRECT0 |24|11.077|.027799|19/24|0|
| S4FIT1024 |48|2.668|.004510|0/48|0|
| DIRECT1024 |24|2.573|.004227|0/24|0|

These are24 conditions from **two TRAIN families**; /48 stochastic outputs are not48 scenes. The predeclared statistic is the median of each condition's DIRECT MSE minus mean S4 MSE, not subtraction of marginal medians:

- DIRECT minus old AGG: **-6.67133945e-5m2**, lower on23/24 conditions.
- DIRECT minus matched trained S4FIT: **+4.59272529e-7m2**, lower on10/24. Only one of four family-command groups favors DIRECT. Both trained readouts have zero wrong directions in both commands/families. Thus DIRECT's slightly lower marginal XYZ median does not establish the registered advantage.
- Both fits improve full XYZ/SO3 relative to their step0 policies. Direction is a local diagnostic under the unchanged1e-9m reporting guard; some corrective displacements are nearly zero. Continuous vectors, magnitudes and complete gate details remain in the tables, not converted to a task success claim.
- Secondary96 M1 Q2/Q3 conditions: DIRECT-minus-S4FIT median paired MSE **+1.78785107e-7m2**; command0 +9.21381681e-7, command1 -6.20793418e-8. All296 strata and family/command tradeoffs retained.
- **Early-support spillover worsens.** Original192 guard prefix XYZ medians AGG→S4FIT→DIRECT: M1 **2.701→4.264→5.026mm**, M3 **2.120→3.525→3.264mm**. DIRECT versus S4FIT has paired M1 prefix XYZ +0.4084mm and SO3 +.0003777rad; M3 tradeoffs are mixed. All guard rejects remain0. These are developed TRAIN-support guards, not blind TEST or LEGACY.
- DIRECT0 had18/296 finite gate rejects; final S4FIT1024 0/1360 and DIRECT1024 0/488 distinct outcomes. Reused AGG's1/1360 finite Q3 j6 rejection is retained. No invalid/poor output was retried.

The result supports learnable same-support corrective fitting in both formulations and counters a uniquely direct-head explanation. It does not solve the separate M1 visited-state/generalization gap or prove that data quantity alone is the cause. No new task threshold, selector, deployment or physical claim.

### Actual budgets and costs

Hopper job646325 finished exit0 (2026-10-04 22:35:05–23:31:21SGT). **2080 total completed updates** including32 planned canary updates, **24987 training heads**, zero technical recovery updates. S4FIT main18440 heads /1890.6555s loop; DIRECT6145 /937.9158s (load/archive excluded). Equal updates are not equal FLOPs. Quality2148/2560 attempts and6228/7168 heads. New teacher labels/episodes/renders/WBC/physics/hardware **all0**. No mail, monitor, broad cleanup, PR77 pointer or merge.

### Complete public preservation

HF repo `leoyuer/pi0246-gate-pass-campaign`, Trainer prefix `arms/PI0272/{arm}/checkpoint-{step}`:

| Arm | Step | Fixed revision | Complete tree SHA256 |
|---|---:|---|---|
| S4FIT | 512 | 0e12a801640fafdddbd80c8c7a49a95424b5cc8e | 8ef5a9190b4fcc4fa2570ae1d1c21d4e0669d2443f86ff82141badc2fb6a4e71 |
| S4FIT | 1024 | dadca28b18645f66d502c230d368378c6335da2c | b49b1234dd33ba98dc1ae789069256ed194bc949f6ccff65beeb5c26219880ba |
| DIRECT | 512 | b6c6ae4ebe1f4c6a9b9308f0bb20f0e6008a52c5 | d2df6662d226f1ea3c9ff298c1e80a0dd828ed25a329a2e08b4fc20f46aeb484 |
| DIRECT | 1024 | 888ec80628932e5dec294330e69cf171d13a4122 | e25154502f778e6a3c1c47e4b5b4260e9268f8a3e5308871bd9dc94f2c9ce67e |

DIRECT requires the explicit DIRECT readout contract; stock S4 loading is a different effective policy.

- Predictions: revision `e7cf79062be247d298ea50b9a96999c27e335927`, tree `cbb096a9fd550c6006273a30e557dc5d861edf649a69838f212121a8629ddd57`, **4296 members**, full anonymous file/all-member readback.
- Summary: revision `4ae9b50af558fb438b1794e58e8bab592a11cf18`, tree `28611f2d40a868085ab01a54360d54fc51e56303e43e01119d433e91129013c9`, **229 members**, full anonymous file/all-member readback. Includes all tables, effective cards, code, actual audits, budget journals, four Trainer receipts and166-file source/log export.
- [Full research return](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/blob/4ae9b50af558fb438b1794e58e8bab592a11cf18/evidence/PI0272/matched-direct-fitting-20261004/summary/FINAL_RESEARCH_RETURN.md), SHA256 `b00f052f9fe05cc822de1654fe21f4bf91b40e0a84b80764301d40c3a6c3375f`.
- [CPU replay](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/blob/4ae9b50af558fb438b1794e58e8bab592a11cf18/evidence/PI0272/matched-direct-fitting-20261004/summary/CPU_REPLAY.json).
- [Complete requirement/archive audit](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/23695db372cf8ca44fd6e2ace2c6b36657500c03/reports/PI0272/final/PREPUBLICATION_REQUIREMENTS_AUDIT.json). All229 summary members checked against actual source bytes; this single canonical and exact readback close publication.

Execution is COMPLETE; **local DIRECT advantage is NOT ESTABLISHED**. Stop for the PI's full-support/feedback-or-fitting decision; no successor run is authorized by this result. PI0270 original/reliable VALID_QUALITY_FAIL with reliable6/64 and positive LEGACY result, older FAILs and PI0267 WBC PARTIAL remain unchanged. `selected_checkpoint=NONE`; `deployment_selected_checkpoint=NONE`.
