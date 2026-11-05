# PILOT1: two native CPU slots, GPU WM coverage and trained contact-exit Policy

The two-slot native contact batch passed. A trained two-action contact-exit Policy was then evaluated in new native DEV and frozen TEST runs. It succeeded 2/2, matching the always-+Y rule 2/2. Increasing TRAIN scene coverage improved one-step TCP prediction, but did not establish action-dependent model superiority or learned control gain. Full grasp/place remains untested.

This extends PR #362 and fixed plan `fec841e74d406b3b556c84b733d2de1615581bf7`; earlier reports and failures remain unchanged. The machine-readable evidence is [PARALLEL_COVERAGE_SUMMARY.json](PARALLEL_COVERAGE_SUMMARY.json). Raw data, trained weights, vendor assets and host-bound execution specs remain private.

## Execution and scientific identities

| Item | Complete identity |
|---|---|
| First two-slot contact runtime | `cfdea0629fc4bf1f15345f3e815580f500fa4690` |
| Matched GPU WM runtime | `5850fab8ff4fb2abaca367eefa9b8491a564a586` |
| Trained contact Policy and native evaluation runtime | `31926780c729b52809f0afa7bd8f4eb98ab4b64f` |
| Free-space evaluation/counting successor | `5f329b9b3128e26f5393592f02585918e794e98d` |
| Frozen native C08 binary | `29b261541b9dc2b2ed5c6df065026b33a11c3019181b9a8cfa980a7878d91df2` |
| Robot dynamics/native safety contract | `5495b40fd499cdce97c78724cce49fa2eec9809b16e7aefc6559b06d6b5f6898` |
| C02 TRAIN bundle | `281d8421bc5532dc1f48367558d8c1832b29a11c68e2e1ae3e0d313cdd41ba3f` |
| New C04/C05 TRAIN bundle | `81397691702b1f1dd517bc674f8a082dd248a24762234929f6f8fc09b28c5c0d` |
| Independent C03 DEV bundle | `c7dc2ea71f904bf0a6a70e19aefdd0b08f886842299b46aba97d75db9bbe57cd` |
| Frozen C06 TEST scene | `1e9940790b136a5dba2e39fd6058d805352bfab5580243a2327082bf3daf96f0` |
| Contact Policy checkpoint | `4c7562944a878de8a50cfad63fe387c779ce05100414b603aeeb9e6975c40171` |
| Portable contact Policy | `0cf6f0ae69f66b32d8aca97b305f59aec842900d852fdaeae305b5e713d6ac3f` |
| Empirical TRAIN-return teacher | `4f6f6cc8c0efc900f924cc43d0002c69557b896f588194ded6469905cf011bbe` |

## Native parallelism

C05 first passed single-environment qualification: 4,103 real native ticks, 913 positive force-contact ticks, no native STOP/outer rejection, clean exit 0. The following HOLD/+Y batch reused the previously qualified topology: one canonical physical owner with two inherited locked file descriptions, disjoint CPUs, ROS domains 220/221, distinct caches/receivers/logs and original consumption-clock checks. Both children separately passed trace, solver-target and native safety checks. No WBC or physics replacement was introduced.

The two children each executed 5,703 ticks and 128 task atoms. Their wall times were 57.3/57.8 seconds; the complete batch took 58.9 seconds and 219.24 CPU seconds. The preceding C04 sequential HOLD/+Y pair took 113.9 seconds. This is approximately 1.9× observed throughput relative to that preceding pair, not a matched same-scene benchmark. The two-slot job had an 18GiB RAM limit; aggregate measured peak RAM was not retained. Both leases and the batch lease closed, all exits 0.

C05 HOLD did not exit contact. +Y moved the actual TCP 0.9657mm away and had 1.474 seconds of terminal noncontact, with table support. These are primitive controls, not trained Policy evaluations.

Native execution is CPU-based MuJoCo/ROS2/OCS2. GPU headroom alone does not make this WBC chain a GPU physics backend. Existing local single-GPU neural work and two CPU environments were used; no H200, GPU multiworld construction, distributed work or batched WBC replacement ran.

## Matched GPU WM coverage experiment

Both fits used the same source, GPU, seed 36003, 64×64 tanh architecture, AdamW 0.001, batch32, TRAIN-only normalization and 2,000 updates per arm. The two arms were ACTION_CONDITIONED and NO_NEW_REFERENCE. This ablation removes the new reference, while retaining causal action history. All original 25 native force events per atom were checked. It is a history-conditioned, partial task-space 50ms model, not a full-state or qualified free-rollout simulator.

| Fit | TRAIN transitions/parents | DEV TCP RMSE, conditioned / ablated | DEV event accuracy | Trainer wall |
|---|---:|---:|---:|---:|
| C02 only | 325 / 1 | 0.4474 / 0.3894mm | 68.1% / 69.7% | 7.42s |
| C02+C04+C05 | 801 / 3 | 0.05523 / 0.04109mm | 96.64% / 96.64% | 8.38s |

DEV remained 238 transitions from one independent C03 physical parent and two episodes. Persistence TCP RMSE was 0.29235mm and event accuracy 97.48%. The coverage intervention improved TCP forecasting substantially; the conditioned model still did not beat its action ablation or persistence for event accuracy. These correlated transitions do not provide 238 independent physical conditions. TEST was not used in fitting or recipe selection.

All four fits restored model, optimizer, normalizer, RNG and cursor exactly, including a separate final optimizer continuation probe. Main updates: 8,000; restore probes: four, reported separately. Maximum allocated/reserved GPU memory was 85,218,816 / 94,371,840 bytes. The per-process allocator limit was 4GiB; ample GPU memory does not identify the native throughput bottleneck.

| Checkpoint | SHA256 |
|---|---|
| One-parent conditioned | `140d177eb7f1ee0bd3f8b741df264e2fcb78d2723b9732c52a52376df5eca066` |
| One-parent ablated | `bc21e1752ccf84ab6f255f9d98b5ef9281daf43a7561b05b698eb806a6f18176` |
| Three-parent conditioned | `82a3e95b15ac8ee4ea84f4ccda1e589ad1f11ef993f3737f183cd61ab1024102` |
| Three-parent ablated | `e0e5b6dca60c452c723a25a96a10dcae05d4b3f668387ff5b867e37db645868c` |

## Contact-exit Policy: actual training and native evaluation

The [recipe and evaluation order](../../../experiments/wm02/pilot1_contact_exit_policy_v1.json) were frozen before training and new Policy rollouts. The teacher is an empirical two-action TRAIN return model and mean 64-atom TCP/object response curve. Six complete TRAIN episodes, two actions on each of C02/C04/C05, yielded expected exit returns [0,1]. The earlier short 96-atom episode was excluded. This global return teacher selects +Y everywhere; it is not a state-adaptive physical simulator.

A 32×32 tanh, two-logit state Policy was actually trained for 2,000 AdamW updates, batch6, seed36004. Its 54 causal inputs are nine measured relative TCP/object positions, finger positions and native force maxima. Future outcomes, teacher returns and DEV/TEST labels are absent from fitting inputs and loss. The original exit score, qualified HOLD/+Y1mm actions, native dt, H40 ramp/renewal semantics and safety remained unchanged. Only a discrete action was submitted.

Checkpoint continuation and endpoint reload errors were exactly zero; NumPy/native versus Torch logits differed by at most 4.77e-7. Trainer wall was 5.56s; worker wall 7.13s, child CPU 6.61s. Peak allocated GPU memory was 68,281,856 bytes.

| New native trial | Actual away Y | Terminal noncontact | Original exit predicate | Native exit |
|---|---:|---:|---|---:|
| C03 DEV +Y rule | 0.9868mm | 1.486s | PASS | 0 |
| C03 DEV trained Policy | 1.4584mm | 2.368s | PASS | 0 |
| C06 TEST +Y rule | 1.1131mm | 1.260s | PASS | 0 |
| C06 TEST trained Policy | 1.2698mm | 2.508s | PASS | 0 |

Both methods selected exactly the same +Y command. These are independent native restarts, not complete MPC/WBC-state forks; differences in trajectory/contact duration are not evidence of Policy superiority. Each paired batch had one physical owner and two isolated native processes. Both methods succeeded 2/2 on two parents, with only one trial per method per parent. No confidence or reliability claim follows from this small denominator.

C06 had a new static check with zero integrations and a separate single native qualification before paired TEST evaluation. Robot dynamics/safety contract matched exactly. Its qualification had 346 positive native contact ticks and ended cleanly at 4,103 ticks. No TEST fitting or outcome-based retry occurred.

## Counts, cost, failures and current state

At the sealed contact TEST boundary: **47 native attempts = 46 native PASS + one preserved recorder failure**, 211,266 real native ticks and 3,745 raw task atoms. Aggregate native CPU was 3,470.41s. Accumulated worker wall was 1,692.94s, counting each parallel batch wall once. Raw task atoms include qualification and evaluation and are not all TRAIN samples. The immutable earlier DEV/TEST batch RESULT field `learning_atoms=256` represented recorded task atoms in that runner version; actual DEV/TEST additions to training were **zero**. A successor runner now separates raw recorded atoms from eligible TRAIN atoms.

Cumulative main training: **WM 16,000 + eight separate restore probes; Actor/Policy 6,000 + three separate probes**. All sealed native and model workers exited 0, except the previously preserved recorder failure. Their leases are CLOSED; the parent/window remain ACTIVE for fresh finite successors.

The first GPU bootstrap imported an unused Torch-dependent replay module using system Python before registration. It failed with ModuleNotFoundError, with no spec, registry entry, lease, supervisor or fit directory. An independent read verified zero model/physics effects. Removing the unused bootstrap import and using a fresh identity fixed it; this is an admission failure, not WM learning failure. Earlier recorder, short-exit and original task failures remain in the prior reports. A readonly status observation also raced replacement of the live state file; it was retried as a read, never redispatched as a job.

New bulk source/data/cache/log/model output uses the independently probed existing non-root local block disk. Its FUSE block mount was checked against the live block-device identity and hashed fsync/rename/lock probe; no mount, permissions, system package, other-user process or legacy storage gate changed. Successful/shared checkpoints were preserved.

`LOCAL_PILOT_TRAINED=true`, `LOCAL_POLICY_EVALUATED=true`, `LOCAL_CONTROL_GAIN=false`, `CONTACT_DATA_AVAILABLE=true`, `CONTACT_POLICY_EVALUATED=true`, `FULL_MANIPULATION_SUCCESS=false`.

## Executed commands and continuation

These were real worker entries, executed with their freshly registered private absolute specs and original host/environment bindings; they are not free-standing grants. Do not rerun CLOSED job specs.

```bash
PYTHONPATH="$WM02_SOURCE/src:$WM02_SOURCE" "$WM02_TARGET_PYTHON" -B \
  -m scripts.wm02.pilot1_parallel_worker --spec "$WM02_NEW_BATCH/EXECUTION_SPEC.json"
PYTHONPATH="$WM02_SOURCE/src:$WM02_SOURCE" "$WM02_GPU_PYTHON" -B \
  -m scripts.wm02.train_pilot1_contact_wm --spec "$WM02_NEW_MODEL/EXECUTION_SPEC.json" \
  --output "$WM02_NEW_MODEL/fit-01"
PYTHONPATH="$WM02_SOURCE/src:$WM02_SOURCE" "$WM02_GPU_PYTHON" -B \
  -m scripts.wm02.train_pilot1_contact_policy --spec "$WM02_NEW_POLICY/EXECUTION_SPEC.json" \
  --output "$WM02_NEW_POLICY/fit-01"
```

All above native batches and actual GPU trainers exited 0. The source regression set passed 180 tests, with nine affected contact tests also passing under `-O`; the later free-evaluation/counting seam passed six additional targeted tests. Fixture tests are software checks, distinct from the real native qualifications and evaluations above.

Next independent task: evaluate the already frozen free-space V0/V1 and three baselines on the original reserved P23/P24 TEST goals, using one single-scene qualification per new parent followed by ten two-slot evaluation batches. The fixed TEST outcomes must not enter fitting. Supported closure/hold/release and complete manipulation remain separate future tasks. There is no current external permission/storage blocker; the outstanding scientific limitation is the absence of learned control advantage over the strong rule.

The free-space TEST continuation has now actually started under a fresh finite coordinator lease, using source `5f329b9b3128e26f5393592f02585918e794e98d` and frozen driver SHA256 `50352c24cd46fe3bff4e639edd22729378d6318586c4bf35d103999835ce8fde`. Its planned denominator is two native scene qualifications plus 20 method evaluations in ten two-slot batches. This active continuation is excluded from the sealed 47-attempt contact boundary above. It performs no optimizer updates and its TEST records are excluded from TRAIN eligibility.


### Free-space continuation: preserved finalization failure and fresh remaining rows

The first coordinator subsequently terminated with exit 1 after two scene qualifications and five paired batches: 12 real native attempts, 49,236 ticks. Eleven native reports passed. One P23 TRAIN-mean trial completed all 4,103 ticks and wrote its score, then failed at `seal_trace` because `pause_ack.json` was temporarily unavailable. The later independent read found the correct paused step/time, running=false, clock_paused=true, a fully sealed trace with 25,483/25,483 records, zero drops, no mixed versions, no STOP and clean native shutdown. The historical report remains FAILED; its task score is not silently promoted to a formally valid native PASS.

This stage cost 826.61 native worker CPU seconds and 288.49 accumulated worker wall seconds, plus 0.76 coordinator CPU seconds. The unit measured peak RAM was 2,792,656,896 bytes. All 12 jobs physically executed 64 task atoms each; one failed before publishing DYNAMIC_DIAGNOSTIC, leaving 704 structured task atoms versus 768 physically executed. These counts are separate from the earlier 47-attempt contact boundary. No TEST data entered TRAIN.

A thin-adapter repair, source `e13344f7eb95453ee93a8870f58f8f15cf42cfba`, retries only absent/incomplete pause-snapshot reads for at most three seconds, then requires exact native step, matching time and paused/running flags before atomically publishing the trace-seal request. It never modifies the step quota, WBC or physics. Wrong step/time/running state and persistent unreadability still fail. Regression: 183 tests pass, with all three new affected tests also passing under `-O`. This is an application snapshot repair; no frozen binary was rebuilt.

A fresh finite successor actually registered and started **only the ten untouched P24 evaluation conditions** in five pairs. Its driver SHA256 is `94404f1016e5c4e6cabaffbcd61abe885db21c713397c97f359b79c1474c57b4`. It reuses CLOSED qualification files as evidence while receiving entirely new worker leases. The ten physically completed P23 conditions, including the failed finalization, are not repeated. Final fixed TEST comparison remains pending this successor's terminal reconciliation.
