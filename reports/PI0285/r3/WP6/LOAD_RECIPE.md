# WP6 prospective recovery components

State: READY for owner integration, not collected, admitted, or physically validated.

`scripts/pi0285_execution/r3_recovery.py` reuses `r2_branch.freeze()` through the owner's existing frozen branch card. It never modifies that selector or healthy runtime. Importing the module executes no teacher, renderer, physics, or model calls.

After a real completed4096 RECOVERY decision, the owner calls `authorize_collection(measured_remaining_reserve_s)` once before constructing episodes. It validates the frozen proof hashes, excludes confirmation candidates, enforces latest06:00 start and one durable45-minute clock. Repeated calls retain the original clock. Reserve includes measured training/evaluation/confirmation time and must be at least45 minutes; owner additionally checks the06:30 optional optimization cutoff and available cumulative resource counters.

For each of `slots()` (TRAIN285100..285107 x2/6/10/16s), load `prefix(original_folder, slot)`. Missing/unvisited states produce a failed slot record, never another seed. Under the sole source-consumer lock, use existing `recorder.Episode` construction and counters. Call `collect_episode(ep, bound, same_teacher_sha256, clock['deadline_epoch'])`; on errors close the recorder with ENGINEERING_INCOMPLETE, retain original raw files and do not automatically retry. The future owner remains responsible for the normal physics/renderer resource lease, process accounting and the per-attempt ledger. No collection launcher was executed or installed by this worker.

Replay starts at the exact saved initial arrays and executes the saved installed native actions, including20 neutral controls. Every100Hz sample is compared at1e-6 absolute tolerance; takeover RGB and StateV1 require byte equality. Only replay that meets those checks reaches the same unchanged `Teacher`. Its internal feedback clock starts at the takeover boundary; stored control timestamps remain global. Teacher may reapproach/regrasp using the actual learner state; no old teacher actions are substituted. The episode runs to original task32s, never success-based stopping. Prefix actions are evidence only and are excluded from the supervised arrays.

On the recorded episode, call `suffix(isolated_data_root, episode_folder, data.codec(original_data_root), cpu_inputs.physical_errors)`. This writes only fresh suffix arrays in the existing `data.Dataset` layout and checks every fullH40 window. Instruction/processor/statistics/mapping identities stay frozen. Raw arrays and all failed codec rows are retained; any unrepresentable supervision stops recovery-bank admission. No H40 spans takeover or reset; no terminal padding. RGB comes from genuine recorder observations. `bank(reports, same_teacher_sha256)` retains32 requested slots and requires24 distinct valid suffixes. Duplicate states count as failed slots; conflicting duplicate slot IDs or any codec failure reject admission.

`recovery_plan(expert_window_rows, admitted_bank)` returns16384 draws, exactly8192 expert and8192 corrected suffixes. Integral allocation keeps correction branches equally weighted (largest-remainder rounding) and global25/25/25/20/5 behavior totals exact. It reuses the event-first8 and eight-stratum semantics. If available suffix phases cannot satisfy both constraints, allocation fails explicitly; it does not synthesize buckets or silently change weights. Owner stages the admitted arrays beside original expert arrays, freezes plan/card/ADMISSION identities and invokes the inherited REPAIR_A2048 phase with carried Adam and original fixed LR schedule.

DOSE is mutually exclusive. `dose_plan(existing_PLACE4096_rows, frozen_decision)` returns exactly16384 deterministic existing-plan positions from at most32768 rows, retaining instruction/source/bucket fields and exact25/25/25/20/5 bucket counts plus the original event-priority proportions. No new row, event weighting or layout search is introduced. It is plan preparation only; the owner binds actual4096 parent and2048 schedule after the selector allows DOSE.

Run the disposable CPU check:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
py -3.12 scripts/pi0285_execution/r3_recovery_check.py --report reports/PI0285/r3/WP6/SYNTHETIC_CHECKS.json
```

All fixtures are explicitly synthetic. PASS does not establish native codec fidelity, successful learner-state corrections, deterministic real rendering/replay, actual24/32 admission, time feasibility, learned-policy quality, or source qualification. Real restore mismatches remain failed requested slots. selected/deployment=NONE.
