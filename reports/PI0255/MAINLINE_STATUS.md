# PI0255 — fixed pilot negative result

Authority: eac535e7c4016df39821798b111fae7ad79e8253 / Issue #11 5884624474.

Original prospective status is retained in MAINLINE_STATUS_PROSPECTIVE_FREEZE.md and Git history. Source/CPU preparation, all36 fixed expert-only pilots and independent replay are complete.

| Task | Valid fixed pilots | Success | Admission |
|---|---:|---:|---|
| M1 REACH_POSE |12/12|0|FAIL|
| M2 ROUTE_AROUND |12/12|0|FAIL|
| M3 TRACE_CONTOUR |12/12|0|FAIL|

36 first-valid failures:18 SETTLING_EXPERT_GATE_REJECT and18 SETTLING_PHYSICAL_TERMINATION. All occur before task phase. They establish that frozen registered preparation failed, not that task trajectories or GR00T generalization were measured and failed. Six reset families are shared across task/command siblings.

18 actual reset pairs match root, DOF, physics seed, Python/NumPy/Torch RNG and initial body/wrist images. Original24 V1 full handoff arrays were missing; deviation is retained and no reconstruction/full gate-codec parity is claimed. Remaining12 use prospectively sealed recording V2, with complete actual chunks and independent CPU gate decisions/issues/metrics replay. Experts/scorer/registry/WBC/gate/physics/seeds unchanged; no valid pilot repeated. Four recorded-camera videos cover both failure classes and recording versions.

Scientific dependency stop: no family reaches10/12, so300 fixed TRAIN/MT-VAL collection attempts remain unexecuted. Successful TRAIN/VAL episodes and eligible imitation windows are0. C_T new-data support, C_MT numerical statistics, every-window task-phase QA and observation sufficiency cannot be closed with settling-only evidence. Future TEST remains metadata only. Verdict: PARTIAL_MULTITASK_PREPARATION_BLOCKED, never DATA_MODEL_READY.

Independent model metadata:1030 common Base/G weight shapes agree, with one documented tied alias and537 known BF16→F32 action-head storage differences. Existing embodiment slot10 is proposed without new rows. Draft6000 cards share new budgets/loss/runtime, task-balanced hierarchical sampling and source/visit RNG CPU checks; old2000 substitution is rejected. Actual augmentation pairing, tokenizer/common processor and real full Trainer resume remain data/execution admission gaps. CPU fixtures are not training qualification.

The registry contains 96 whole scenario families: 6 PILOT, 40 TRAIN, 10 MT-VAL, 20 future composition TEST and 20 future geometry TEST. It schedules 336 expert episodes and seals 240 additional command rows as TEST manifest-only. Hard physics-start cap is 360, including at most 24 invalid engineering starts. Valid FAIL is retained once and cannot be rerun.

PI0254 is independent: U4608 training integrity/full Trainer HF readback PASS; the unchanged original384-cell final screen is launched on Corelab. R637808 stopped on source/visit processed-input hash mismatch before forward, with0 updates and both allowed submission slots consumed. Canonical5886087016 retains the partial block; no third qsub or paired causal claim. All PI0255 files remain under Projects/go2-arx5/PI0255; quota headroom UNKNOWN. No unrelated workload changed.

New PI0255 GR00T loads/queries/backward/optimizer=0, learned-policy episodes=0; expert physics starts=36, invalid engineering starts=0. selected_checkpoint=NONE; deployment_selected_checkpoint=NONE. No new Hopper job, old DEV/F, contact trial, WBC/gate change, broad cleanup, merge, mail or monitor.

PUBLIC raw archive PASS: leoyuer/pi0246-gate-pass-campaign@c6bc2a041323788a4b19b7b8ef11227f9950a0a6, evidence/PI0255/fixed-pilot-audit-v1-20260929. Anonymous full file/archive-member readback verified all789 members/21933826 logical bytes, tree8d7f94322144ba2b3620d9b45e913528c3d5a37e0b9297646093d857b9875190. Receipt: HF/pilot_HF_public_receipt.json. Original pending verdict is retained; READINESS_VERDICT_WITH_HF.json adds this durable binding without changing the scientific stop. Private WBC code/weights/assets and credentials are excluded.

Observation caveat: M2 registered paths progress monotonically in world X; M3 completion returns to the visible start marker. A single measured pose/image may not reveal completed-lap count. The actual dataset audit must report that ambiguity; no phase/source ID enters the loader and no history contract is silently introduced.
