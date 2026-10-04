# PI0270 actual fixed data, full collector replay and training continuation

Authority: release `7a6eea5ae38854bec7b80782213a92ca02fd0318`, Issue #11 canonical5969189182. This is a progress return. It is not an AGG quality verdict or the complete-study terminal. Historical PI0269 valid FAIL, WBC PARTIAL and selected/deployment NONE remain.

## Actual data and fixed archive

The original collection finished192/192 cases from all24 fixed TRAIN families. All6784 teacher/codec proposals closed:6782 eligible,2 VISIBILITY exclusions, no replacement samples. All96 family/task/command cells pass; the minimum distinct processed-state count is62 against the registered floor8.

The immutable replacement mapping contains4096 recovery microbatches, with4096 distinct source proposals, each used once. COMMON retains4096 historical microbatches. M1 and M3 each have1024 recovery records per command. M1's T0..T7 each have256 records; M3 retains its original text. This is a frozen plan, not proof of neural consumption.

| Recovery source bucket | Scheduled microbatches |
|---|---:|
| ORIGINAL_Q0 |1032|
| ORIGINAL_Q1 |1028|
| ORIGINAL_Q2 |996|
| ORIGINAL_Q3 |936|
| M1_TERMINAL |32|
| M3_ENTRY |24|
| M3_MIDARC |24|
| M3_TERMINAL |24|

The selected mapping has1630 projected-terminal records. The strict zero-prefix-displacement count remains0; projection at the endpoint does not mean zero corrective action. The full proposal support contains2633 projected-terminal labels. Normalized state spans[-5.4680085,8.0226297], action[-29.1257687,36.4056320]; no clipping or statistics refit occurred.

`RECOVERY_EXPOSURE_PRETRAIN.json` reports continuous current XYZ/SO3 errors, teacher prefix-displacement magnitude and projection fraction by task/source bucket. This measures frozen planned exposure, not actual optimization. NEW_FIT192 is exactly the first two distinct conditions per cell: one Q0 and one Q1 condition, with48 rows per task/bucket. It does not represent all late or component support, and the frozen card was not changed to make it more representative.

The new training archive passed anonymous fixed-revision full-file and all-member readback:

- Repo: `leoyuer/pi0246-gate-pass-campaign`.
- Prefix: `evidence/PI0270/learner-state-recovery-20261004/training-data`.
- Revision: `d16394d8e32d535465828ef5d572b491a373e759`.
- Tree: `fa95b40b358254e5a0196548c6c5e805edc500973670aed6389e88e56ba7fbde`.
- Members4298; source83496951bytes; tar86794240bytes; selected recovery samples4096.
- Mapping SHA256: `53fe049e406f079e079d03af71e25c0ffd339b587fab7d8e5dc80bebe35fd633`.
- NEW_FIT192 SHA256: `553d254cea5ad5c32d0d8acf615f2abfe222459ac10e9d0bb0e09b786884d226`.

The fixed archive contains the full mapping, NEW_FIT, admission source indexes and selected samples. The earlier collection-input archive is separate. Neither receipt claims complete study preservation.

## Full saved collector replay, without new queries

The finite CPU continuation reused104 prior S4 cases and136 prior trajectory checks; only remaining saved arrays were checked. It closed192 cases/19200 native-S4 outputs and192 same-tick trajectory/V3/reliability checks. New model, teacher, optimizer, render and physics queries for replay were0.

| Task / mathematical executor | Original completion | Reliable | Safe | Median reference final XYZ | Median achieved final XYZ | Median full-episode same-tick XYZ RMS |
|---|---:|---:|---:|---:|---:|---:|
| M1 / E0 |47/48|0/48|48/48|140.541mm|140.541mm|0mm|
| M1 / E1 |47/48|0/48|48/48|168.322mm|167.774mm|1.255mm|
| M3 / E0 |0/48|0/48|48/48|62.169mm|62.169mm|0mm|
| M3 / E1 |0/48|0/48|48/48|62.937mm|62.982mm|0.827mm|

All94 successful M1 episodes subsequently exited the pose region. Each group has12048/12048 late ticks outside the joint XYZ/SO3 criterion. This is not a claim that both errors individually exceed tolerance at every tick. E0 follows its installed reference exactly; E1 RMS tracking is much smaller than final goal error. The observed reference behavior can fail without WBC tracking error. These are fixed PREFIX TRAIN-collection results, not AGG failures, blind TEST, WBC or independent robot trials. V3 and executor replay share their frozen mathematical implementations; independent M3 entry/physical checks retain that disclosed limitation.

On the full original-start M1 projected-terminal support, E0 has1266 and E1 has1214 eligible labels. Median paired teacher 0.4s endpoint-error changes are-65.715mm/-52.193mm; predicted-reference changes are+2.790mm/+3.023mm. These are same-state generated-reference comparisons, not assumed teacher execution or differences of marginal medians.

## Actual Hopper admission and CPU encoder repair

Hopper completed local readback of all4298 archive members. Existing ROT2048 initializer bytes were rehashed and bound without historical model downloads. Storage admission observed403074711552available bytes, with403190054912parsed quota headroom, against115151444400required bytes including save overlap and10GiB margin. The quota command returned1 while yielding its valid table; that return code was preserved. Project credits remain UNKNOWN because `hpc project` requests `amgr login`; scheduler acceptance is still required.

The first actual CPU stream stopped before microbatch197 (unit99/micro0), after196 valid records, with `RECOVERY_PROCESSOR_INPUT_OR_LABEL_MISMATCH`. Its original partial audit is preserved. No neural/optimizer update occurred.

The exact failing packet was independently re-encoded on both hosts. RGB/language/token/state/mask identities matched. Only12 FP32 normalized rotation-label elements differed, by1–2ULP, maximum7.450580596923828e-09. NumPy1.26.4, SciPy1.15.3 and processor/pose source hashes match. Corelab's actual NumPy OpenBLAS core isPrescott, Hopper's default isSapphireRapids. Disabling NumPy AVX512 alone and selecting Haswell did not restore parity. Setting `OPENBLAS_CORETYPE=Prescott` restored the registered action bytes exactly. No raw labels, processor map, statistics, precision, tolerance, mapping or scientific loss were changed.

The full CPU check subsequently completed8192/8192 microbatches: COMMON4096 input/label identities match the historical control exactly; RECOVERY4096 match the unchanged frozen mapping exactly; all192 NEW_FIT positions are bound. `HOPPER_CPU_ADMISSION.json` reports `PASS_PI0270_ACTUAL_FROZEN_MAPPING_AND_CONTROL_INPUTS`. Actual sample-audit SHA256 is `938d1514a372b48a506ab538d7aa206689bc73c7f9258af3c66b5274af9e5cbc`. No tolerance replaced any byte-identity assertion.

Only after that CPU process exited was the isolated training entry updated to bind the same CPU core and record its actual loaded-library identity in each phase/checkpoint. The data-loader AST and optimization configuration remain unchanged. The actual frozen helper import/hook-install check passed, followed by a4480-file execution seal. The scientific recipe, data, label map and dtypes were not changed.

One `auto`/H200/12CPU/225GB/12h submission was accepted as `645483.hopper-m-02`, routed by PBS to `small`, running on `hopper-25`. Mail points are `n`. The post-qsub receipt construction initially raised a duplicate-key TypeError after the job had already been accepted; the owner recovered the receipt from the unique actual scheduler job, exact submitted path/resources and original intent. No second qsub, cancellation or training restart occurred. PBS timestamps have whole-second resolution and are preserved as emitted.

The actual phase receipts identify the loaded Prescott OpenBLAS library SHA256 `9254d0854dd7615e11de28d771ae408878ca8123a7ac204f21e4cc7a376cc2e5`. The real continuous8 versus4-save-freshresume4 comparison then passed:1030 model tensors and1413 optimizer tensors have maximum difference0, scheduler LR difference0, actual packets equal, modes restored, and all Python/NumPy/CPU/CUDA RNG states equal. Total canary updates16. Only the explicitly discarded compared-canary payloads were retired by the existing authorized helper; its exact inventory is retained.

At2026-10-04T03:45:37.693420Z, the actual job remainedR and had automatically completed74/4096 main updates,90 cumulative completed updates and1662 recorded heads, with150 main microbatches observed (an in-flight unit is not rounded into a completed update). No quality query occurred. This is a dated live observation, not main training completion. PBS node inventory independently reports `gpu_model=H200` for the assigned `hopper-25`; other jobs were not changed.

## Next authorized actions and accounting

The canary PASS automatically admitted the now-running AGG4096 from ROT2048 weights/fresh optimizer. Preserve both complete2048/4096 Trainers and fixed-HF readbacks; only final4096 quality is allowed. Do not retrain PREFIX or recollect data.

Remaining fixed quality includes968 AGG offline rows,384 paired NEW_FIT rows,64 AGG original-start episodes and32 AGG component proposals, with exact historical reuse, new CPU replay, actual all-eight-case video, complete preservation and one terminal canonical. No result-dependent recipe or card change is authorized.

Collection consumption remains19200 inference attempts/76800 heads/192 starts/19200 frame pairs/6784 teacher calls. Actual training is16 canary plus74 main at the latest cited observation, against4112planned/4368hard completed-update budget;1662 heads against80000hard. AGG quality queries remain0. Read current journals for later counts. Kernel diagnostics and saved-array arithmetic consume no scientific model/gradient budget. Formal selected/deployment remainNONE.
