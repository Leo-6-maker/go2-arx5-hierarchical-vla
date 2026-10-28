# PI0253 FULL: fixed 1000/2500 retention and continuation

Authority: `PI0253_LIVE_BRIDGE_TO_FULL_SUPPORT_GO`, commit `82554098fb55b39ceca4b955252f8952c91c0dc0`, Issue #11 canonical `5870278574`. This report retains the original G M64 PASS and TRAIN-A/C0 sim FAIL.

## Scientific decision

`CONTINUE_FIXED_FULL`. Independent CPU replay confirms **0/64 LINEAR_SPEED_REJECT at both 1000 and 2500**. The preregistered two-timepoint 64/64 speed barrier is not triggered. Ordinary intermediate quality failures remain failures and do not stop this family or authorize a loss/seed/step change.

The verified training prefix is **2500/10000**; FULL training is not complete. `selected_checkpoint=NONE`. No FULL CALIB, selector, winner sim, DEV/F or new hypothesis was executed.

## Actual quality table

Each row uses original S4, processor/statistics, source identities, actual B0/B1 packets, physical decoder, gate and metrics. XYZ units below are millimetres; SO3 is radians. Old16 is a subset, not additional queries.

| Snapshot | Support | Noise | Valid | REJECT | Median XYZ mm | p95 XYZ mm | Median SO3 rad | Original quality |
|---|---|---|---:|---:|---:|---:|---:|---|
| G2000 / FULL step0 | all64 | B0 | 64/64 | 0 | 14.808 | 31.454 | 0.016274 | PASS, reused |
| G2000 / FULL step0 | old16 | B0 | 16/16 | 0 | 12.546 | 25.101 | 0.014118 | PASS, reused subset |
| FULL1000 observation | all64 | B0 | 64/64 | 0 | 67.649 | 156.129 | 0.041260 | FAIL |
| FULL1000 observation | old16 | B0 | 16/16 | 0 | 56.320 | 112.664 | 0.037452 | FAIL |
| FULL2500 candidate | all64 | B0 | 64/64 | 0 | 54.544 | 194.305 | 0.050837 | FAIL |
| FULL2500 candidate | all64 | B1 | 64/64 | 0 | 57.874 | 196.700 | 0.054379 | FAIL |
| FULL2500 candidate | old16 | B0 | 16/16 | 0 | 50.803 | 95.426 | 0.059744 | FAIL |
| FULL2500 candidate | old16 | B1 | 16/16 | 0 | 51.871 | 101.532 | 0.059629 | FAIL |

Original four requirements remain zero REJECT, median XYZ <=20 mm, p95 XYZ <=35 mm and median SO3 <=0.050 rad. FULL1000 is model-only: it is neither a selection candidate nor an exact-resume source. FULL2500 remains a formal preserved candidate even though its current retention quality fails; selection awaits the complete family and original CALIB protocol.

These observations show degraded fixed-support quality relative to G. They do not establish data scale as its unique cause, and neither gate acceptance nor this intermediate continuation is task success. No model output was used to alter training or selection.

## First-valid and independent CPU evidence

- FULL1000: 64 new queries, 256 S4 head forwards, no reuse or score-based retries; screen SHA256 `8efa7594cacf0ac59d02559ef84dc758c8754cf088e1a25f0a9ac1678a1ee72b`.
- FULL2500: 128 new queries, 512 S4 head forwards, no reuse or score-based retries; screen SHA256 `831538224fa40cdd261e41ac955b7e159fc3dbca2dd7d163641dd400c8ec9328`.
- Independent CPU replay decodes actual saved tensors and recomputes metrics and gate issues. Both receipts are `PASS_INDEPENDENT_CPU_REPLAY`; CPU replay added zero model queries or optimizer updates.
- CPU receipt hashes: 1000 `90c2ce977932acf66852fd74a1ba261dd8c1e67b7a2aac2d12dd817015f99506`; 2500 `33635b0ebf53fc3b9582c0448df645674bd3926717772cc42a2a98f6a30df7b0`.
- Barrier SHA256 `5196da1bdd77d7b4593e23091f01982bab3cf2888e5146bb29642a4d675c20cd`.
- Executed screen orchestrator SHA256 `f8d45f5d8980e6a37cb90dfda56471b08291d1852d751d103acb6bb63c99e8cc`; unchanged qualified Q runner SHA256 `1f668589f3ad9d8004c35de0a2ea3550102d7074390e06e081367f5ae83de5cc`.

Total new retention queries so far: **192/576**, 768 head forwards. The 64 step0/B0 results are reused. D1 remains its original 8 logical cells, 2 reused and 6 new. Old effective cells and previous failures were not rerun.

## Durable identities and execution accounting

Public HF repository: `leoyuer/pi0246-gate-pass-campaign`.

- FULL1000 observation: revision `6cdfd62f0a8ee15a6f6fdd4ab4f2fae95416f888`, prefix `arms/PI0253/full-g-support-20260928/G/model-only-1000`; fixed-revision complete inference-file readback passed.
- FULL2500 complete Trainer: revision `289b3a8c98a8f8aa8cc62a94c26c3a827c154a4b`, prefix `arms/PI0253/full-g-support-20260928/G/checkpoint-2500`, tree `fadc5581ebfce50e79a94078d7d6434ee333a55f7e8c1f8e9350f475c5a580cb`, 9 files / 20,882,714,202 bytes; durable and full HF readback passed.
- Sealed run card SHA256 `3bece4c02d8158ec09a35fc597d6ffd34290b8d1559ea55ed877da75c6714277`, unchanged execution commit `568fe6a16dfbf9aa6700375f668715efae655d7f`.
- Initial job `636930.hopper-m-02` completed 2500 updates, then stopped intentionally at `WAIT_FIXED_2500_BARRIER`. Its actual prefix loss, consumption, keyed RNG, four-step gradient/mode and frozen-range evidence passed read-only verification; this is not a 10000-step integrity manifest.
- Resume attempt `637034.hopper-m-02` exited during restore preflight: **0 new updates**. The local retired checkpoint path contained only late configuration/processor/statistics artifacts; it did not contain weights or full Trainer state. The mature restore guard correctly rejected it rather than starting fresh.
- The metadata-only directory was preserved with its 9-file inventory under `engineering_prepare/attempt-02-resume-637034`, leaving the canonical restore path available for the existing pinned HF restore helper. No execution source, card, loss, schedule, optimizer plan or scientific identity was changed.

## Continuation status

The existing pinned HF restore helper completed on the CPU path with `PASS_PINNED_HF_EXACT_RESTORE`: all 9 files / 20,882,714,202 bytes, exactly the original FULL2500 tree. Restore receipt SHA256 `888d7558a6c59a049e30bac9386db096e8e69621634df48fe9f3fb073739f42d`. This restore added zero project model queries and zero optimizer updates.

Exact continuation job **637051.hopper-m-02** was submitted at 2026-09-29 00:09:40 SGT. At submission verification it was **Q**, with no hold, `Mail_Points=n`, legal auto->small routing; PBS reported insufficient available GPU resources. It has not yet proved an actual 2501 update or Trainer-state load. The source/card/code are unchanged; a fresh optimizer restart is not authorized for this suffix.

The actual persistent-fileset probe before submission reported 175,811,592,192 bytes free, with 31,620,064,634 bytes required for the next complete checkpoint plus 10 GiB reserve. Quota headroom remains `UNKNOWN`; this is not a quota proof. No new broad cleanup was performed in this goal.

## Complete HF evidence readback

The complete first-valid retention arrays, original source condition tensors, actual noise packets, qualified source report, CPU replays, cards and executed screen code are preserved in HF PUBLIC:

- Prefix `evidence/PI0253/full-barrier-20260929`.
- Fixed revision `5043765c8d3b35c1e7d036bf71bb96264e66dbc0`.
- **795 files / 241,443,594 bytes**, tree `d75abaf5df60b181794bc2f1c226136594c92d40c8119933be50ffd77d648323`.
- Status `PASS_FULL_FILE_READBACK`: every file was streamed from this fixed revision and its complete size/SHA256 matched the local inventory.
- Receipt SHA256 `eabb61901ee0d3c179221cfae4437afcf2f115bddaf2e93e36ab66a6456cda69`.

The earlier canonical correctly reported readback as in progress. This appended evidence closes preservation without rerunning a cell or retraining. Current continuation job 637051 was subsequently observed **R** on `hopper-17`, with no hold and email disabled. Actual loaded-state/first-suffix-update evidence is still separate from PBS running status.

## Observed exact suffix

Job `637051.hopper-m-02` actually loaded the complete optimizer/scheduler/RNG state and produced `RELEASED_AFTER_OPTIMIZER_SCHEDULER_AND_RNG_RESTORE`. The release receipt binds the same FULL2500 tree and archive/readback receipts; its SHA256 is `cbf3df964b3c81edb13a8881c48f802d014ef4d26869ff5b2ed13c43096295ad`. The restored local checkpoint was retired only after this load, while the pinned HF recovery source remains intact.

At 2026-09-29 00:21:55 SGT, actual Trainer progress and mode records had both advanced through at least **2701/10000**. The read-only observation receipt is `FULL_EXACT_RESUME_OBSERVATION_637051.json`; it is not a full-training integrity/quality PASS.

The first suffix record is exactly scheduled step2501, fixed/episode index8457, source `pi0227_persistent_train_d_j03_yp015_s04_ym050_r03_r03_227619:obs-0060`, yield ordinal2501. Its key is `PI0253|22301|stage=FULL|optimizer_step=2501`, seed8024279732684664450, and its pre-forward guard passed. This confirms the suffix began at the intended cursor rather than skipping again or restarting at step1.

The same actual row retains S4, four checkpointed Euler steps, four backward recomputations, no expert action in condition, no active-value intervention, original loss coefficients (.4/.1/.1/.1), and restored mode. Cumulative rollout/backward head counts at step2501 are both10004, including the verified 2500-step prefix. These observations added **zero** model queries or optimizer updates beyond the authorized running training itself. Formal selected remains `NONE`; FULL5000/7500/10000, complete retention/live side reports and CALIB remain pending.
