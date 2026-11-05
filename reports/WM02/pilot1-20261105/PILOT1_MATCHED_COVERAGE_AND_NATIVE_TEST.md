# Real TRAIN coverage improved the small Actor, without beating the strong rule

The matched experiment completed real collection, two fresh GPU fits, exact checkpoint recovery and a newly preregistered native TEST. Expanded TRAIN coverage improved the small Actor from **0/4 to 2/4** local XY goals and reduced mean actual error from **0.578mm to 0.253mm**. Frozen ARX scored 2/4 and the TRAIN-mean rule 3/4. This is directional evidence from four goals on two physical parents, not established superiority or full manipulation success.

## Native comparison on new P25/P26 TEST parents

| Frozen method | Native-qualified successes / goals | Mean actual XY error | Chosen actions |
|---|---:|---:|---|
| Geometric rule | 0/4 | 0.7853mm | −X, −X, −X, −X |
| TRAIN-mean response | 3/4 | 0.1365mm | HOLD, +Y, +X, −X |
| Frozen ARX | 2/4 | 0.1845mm | HOLD, +Y, +X, −X |
| Fresh Actor, old TRAIN coverage | 0/4 | 0.5780mm | +X, +X, HOLD, −X |
| Fresh Actor, expanded TRAIN coverage | 2/4 | 0.2528mm | +X, +Y, +X, −X |

Two single-scene qualifications preceded 20 evaluations in the qualified two-slot CPU-native configuration. **All 22 native workers passed**, consuming 90,266 physical ticks and 1,408 real task atoms. Every trial has all 64 exact atom-consumption records, solver-target confirmation and a zero-drop pause-sealed trace. All exits were 0, STOP and outer-rejection counts were zero, and all child/batch/coordinator leases closed. There were no retries for scores.

P25/P26 and the 20-row panel were frozen before the expanded fit. Original goal indices, 0.1858445664mm success threshold, action set and WBC remained unchanged. Each condition uses an independent native restart, not an exact hidden-state fork. ARX and the TRAIN-mean rule chose identical actions but obtained different outcomes across these restarts: finite native variability still matters. Four goals do not support a reliable algorithm-superiority claim. P23/P24 and P25/P26 remain consumed TEST and cannot be fitted or reused as fresh blind tests.

Frozen native C08 binary: `29b261541b9dc2b2ed5c6df065026b33a11c3019181b9a8cfa980a7878d91df2`.
Native/runtime source: `c255713e3a8aa988070d454f4e1a8fb66293249b`.
Panel-driver SHA256: `3b40ae37e9edb39a852fb0bb674087ade7d9050ea80dcf40d5b0b4f7284f1984`.
Panel SHA256: `892dbb19a7d4536354889b71dead652e81bb35beb21bdf7ac91ee475673b218b`.
Scene P25: `4298d2e204703d9dc0d1dfbd8cb9300f6f257069b1a9f84acbacadbd96cf9e4d`.
Scene P26: `8d1753ca2f9c075e9f8140c698970eeef1cf95aa32bdd923afb50a6323087df9`.
Native data identity: `3319c4bd1ad7df7bf5d9e9ed5a55342676eebc66a18457a853ced0dc1c33c142`.

[A4_NATIVE_TEST_FINAL_SUMMARY.json](A4_NATIVE_TEST_FINAL_SUMMARY.json) contains every trial, data-file identity, score, worker exit, cost and sealed-state facts.

## Data and matching: 30 completed conditions, 29 eligible for fit

The reserved P18/P19/P20 TRAIN panel completed all 30 original feedback conditions, adding V1 beside V0. There were also three native scene qualifications, one zero-tick startup failure and one fresh 50-tick clock diagnostic: **35 native attempts, 34 PASS, one preserved failure, 135,449 actual ticks**. The 30 complete feedback conditions consumed 1,920 raw task atoms.

The existing replay builder correctly rejected the completed child of the failed pair. Its peer had zero integrations; the batch remained FAILED. The first attempted replay build therefore exited 1 before producing a completed replay or fitting a model. That attempt, its partial files and cost are preserved and its original Journal intent was independently reconciled. The next build retained the existing whole-batch-PASS rule and excluded this one completed child. It did not relabel the batch or repeat the child to obtain eligible data.

Actual feedback appended: **29 windows, 696 atoms**, rather than the preregistered prospective 720. TRAIN increased from 24 to 53 windows; all eight DEV windows stayed unchanged. The original frozen task and its scoring were not recalculated. A direct audit of all 29 new native decisions found **exactly zero difference** between the live Actor input and its corresponding replay input. It did not read TEST or use future states as student features.

TRAIN data identity: `4d48a8aba66e5956da85e9e4df314021b8de370e3172310c55834bbc625dbc4b`.
Old replay: `ea11b27e568443b5f9964f36da7563040a59e4dfb012391c0708a9b747d04734`.
New feedback replay: `c1a8ff3d3838a18c83c4488da29dbc33c55ebd2bb1d6b47165adadf37cabc66a`.
Merged replay: `de7982577751b8db1e54b1039b6b4d0082c45f0c927696e9c513414ef5879647`.

The engineering failures remain separate from scientific task failures: the first private builder omitted the qualification reference and stopped before feedback admission; one native startup never consumed its first clock request; the coordinator then tried reading that child's absent score; a reconciliation driver expected a standalone worker file where the parallel runner uses a batch-child exit and CLOSED child lease. This last driver failed before any new native worker, was independently sealed and corrected. No successful qualification or completed feedback condition was repeated.

The new clock wrapper waits for exactly one matched key subscriber, records a single ENTER publication, and never repeats an unacknowledged toggle. The fresh 50-tick check, remaining three TRAIN conditions and all 22 subsequent TEST trials passed. These results establish successful new executions; they do **not** prove that the historical zero-tick cause was DDS. Delivery versus native keyboard-event consumption remains unresolved in that historical attempt. See [startup evidence](PILOT1_TRAIN_COVERAGE_STARTUP.md).

## Real matched GPU optimization and recovery

Both arms use identical trainer bytes/source `5dbe7d6c2eebc899e4e00c8d84a445e174aef5d6`, fresh seed 36001, two hidden layers of 128, AdamW lr 0.001, batch 128, weight decay 0.0001, and 2,000 optimizer updates. Each fits its normalizer only on its own TRAIN inputs. Both use pure soft ARX-teacher cross-entropy; no success-clone/failure-unlikelihood continuation is mixed into this comparison. Teacher labels are model labels, not physical counterfactual outcomes.

Frozen ARX: `c69ae25164b023cb005e17aaa0c6a45b9d75f756963a1887c782597e7acf55d1`.
Frozen task: `5e0653617bc735dd39ec75cac1598d4b76488d1c6d72ecc50e30ae6d6d6c7b24`.

| Metric | Old coverage | Expanded coverage |
|---|---:|---:|
| TRAIN / DEV windows | 24 / 8 | 53 / 8 |
| Main updates / separate restore probe | 2,000 / 1 | 2,000 / 1 |
| Fixed DEV teacher agreement | 59.9121% | 87.0117% |
| Fixed DEV mean model regret | 0.19573mm | 0.03852mm |
| Full restore / endpoint reload error | 0 / 0 | 0 / 0 |
| Peak GPU allocated / reserved | 147.34 / 341.84 MB | 226.35 / 467.66 MB |
| Worker wall / child CPU | 6.330 / 4.848s | 6.730 / 5.262s |
| Trainer CUDA event time | 2.504s | 2.677s |

Old-coverage checkpoint: `87230a2341a7c79d4686ab14fe001aec9c6de8add06dcbfe249f2bb717a94cf3`.
Old-coverage portable Actor: `e3ee91086a19922e072bedbe4f1028bab4f80df1676b44a58bc20df9731935fe`.
Expanded checkpoint: `92d8e8f59946c2fc4cd3ce36f94560496f7cb126324801d412e2ff8014e3e5b7`.
Expanded portable Actor: `8f965774a280eb7bdc42d4780f5b221ded8aee3b6ad3b627cf55ea7b9090e7ae`.

Both neural workers exited 0 and closed their fresh leases. The endpoint model, optimizer, normalizer, random states and sample cursor restored exactly. Native inference used the exported NumPy model without loading Torch into the ROS environment. [TRAIN_COVERAGE_FINAL_SUMMARY.json](TRAIN_COVERAGE_FINAL_SUMMARY.json) includes the complete data and model comparison.

## Costs and final owner state

| Sealed work | Actual CPU | Wall measure |
|---|---:|---:|
| 35 TRAIN/qualification/diagnostic native attempts | 2,368.15s | 832.36 accumulated worker s; each two-slot batch once |
| 22 new TEST native attempts | 1,551.92s | 562.79 accumulated worker s; coordinator elapsed 578.96s |
| Two GPU Actor jobs | 10.11 child CPU s | 13.06 worker s; 5.181s CUDA event time |
| Rejected first replay build | 11.00 child CPU s | 15.66s, exit 1 |
| Valid replay build | 12.35 child CPU s | 13.35s, exit 0 |
| Static new TEST scene preparation | 1.33 child CPU s | 2.73s, exit 0; zero integrations |

Coordinator and native-worker costs are distinct; TEST coordination added 30.83 CPU seconds. Read-only audits, source transfer and all engineering overhead were not fully metered. Dollar cost was not metered. Do not treat the worker totals as a complete billing statement.

At the final canonical cold boundary, the cumulative local PILOT comprises **129 native attempts, 125 PASS and four retained native failures, 542,506 ticks**. Main model updates: **WM/residual 16,000 plus eight separate restore probes; Actor/Policy 10,000 plus five probes**. Native-worker CPU totals 9,156.30s; accumulated worker wall is 3,735.12s with each concurrent batch counted once. Task-score failures are retained even where native safety passed; they are not excluded from the method denominators.

Independent registry reconciliation checked 193 Core PILOT entries, found no unexpected ACTIVE worker lease and no matching native executor alive. The parent/PILOT authorizations remain ACTIVE with the original deadline; they are not running workers, and no CLOSED lease was reopened. No queue or auto-dispatch successor was left running by this panel.

Both neural fits used GPU1. A fresh final query found GPU0 busy with other-user work and only 513 MiB free; GPU1 had 15,868 MiB free and 0% utilization. Those other-user tasks were untouched. The two-slot native path is CPU/OCS2/WBC based; extra GPU memory alone does not create additional native environments. No four-world or GPU-physics execution, distributed job, H200 allocation or replacement WBC was started. New bulk data, caches and model outputs remained on the qualified existing non-root local disk.

## What this local pilot delivered, and what remains

- **A line:** a trained/reloadable four-action Actor, actual native comparisons, real TRAIN feedback, a successor, and a matched data-coverage experiment with fresh TEST. The original V0/V1 TEST generalization gap remains in the [earlier report](PILOT1_FIXED_TEST_AND_SUPPORTED_CLOSE.md). The new result supports further coverage work, with no claim of robust advantage over the strong rule.
- **M line:** differentiable ARX equivalence and model-gradient checks completed, followed by small residual training/recovery. Its DEV error did not beat ARX; this negative result remains preserved. Numerical model gradients are not validated physical Jacobians.
- **C line:** real contact entry/exit data, matched contact-WM training, and a trained exit Policy evaluated on DEV/TEST. The Policy matches always-+Y; the action-ablated WM remains better on its reported metric. A separate scripted supported pinch/hold/release achieved 1.2s continuous hold and 1.836s terminal release. Learning a supported-grasp WM/Policy and full lift/carry/place remain future work; that scripted success is not labeled learned control.

The next scientific comparison should use additional preregistered physical parents/replicates to estimate native variance, or a separate supported-grasp learning task with independent TRAIN/DEV. The consumed TEST parents must stay out of fitting and future blind panels. No current external permission blocker was found. GPU-physics/H200/distributed expansion still requires the plan's scale review before any construction or job.

Actual executable entries used, with paths bound by their registered private specs:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$WM02_SOURCE/src:$WM02_SOURCE" \
  "$WM02_TARGET_PYTHON" -B -m scripts.wm02.pilot1_parallel_worker \
  --spec "$WM02_BATCH/EXECUTION_SPEC.json"

CUDA_VISIBLE_DEVICES='' PYTHONPATH="$WM02_SOURCE/src:$WM02_SOURCE" \
  "$WM02_TARGET_PYTHON" -B -m scripts.wm02.build_pilot1_feedback \
  --manifest "$WM02_FEEDBACK_MANIFEST" --base "$WM02_OLD_REPLAY" \
  --output "$WM02_NEW_REPLAY_DIRECTORY"

CUDA_VISIBLE_DEVICES="$WM02_GPU1_UUID" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH="$WM02_MODEL_SOURCE/src:$WM02_MODEL_SOURCE" \
  "$WM02_GPU_PYTHON" -B -m scripts.wm02.pilot1_neural_worker \
  --spec "$WM02_MODEL_RUN/EXECUTION_SPEC.json"
```

These are historical commands, not grants to replay CLOSED specs. The exact executed arguments and next safe read-only audit command are retained in the local private handover. Validation: 195 world-model tests pass; all three new clock-resume seam tests also pass under `-O`. Native results above are additional evidence, not inferred from synthetic tests. The branch remains an unmerged draft PR.
