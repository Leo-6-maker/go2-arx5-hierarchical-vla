# Stage23 Final Report

Date: 2026-09-03 (Asia/Singapore)

## A. Classification

```text
PASS WITH DEVIATION
```

Primary diagnostic gate:

```text
STAGE23_ACTION_PIPELINE_AND_EVALUATOR_SANITY_PASS
```

The deviations are operational/provenance-only: official video decoding
required the existing Stage13 native FFmpeg library path, the environment did
not provide the standalone `lerobot` package, and the first tiny-fixture
diagnostic did not make its statistics provenance explicit. The P0 rerun now
does so and passes the exact Stage22 source identity and all three official
loader variants without a model forward.

## B. Action contract conclusion

```text
physical action: 10D
universal action: [40,132]
decoded physical action: [40,10] / batched [1,40,10]
active loss dimensions: 10
masked padding dimensions: 122
padding contributes to loss: false
EEF transform: T_reference^-1 @ T_absolute_target
reference: last observed state[eef_9d]
rotation: first two matrix rows, row-major rot6d
gripper: absolute separate channel
production timing: TBD_BY_ADAPTATION_DATASET
simulation timing: 0.05 s TEST_ONLY_STAGE22_SIMULATION
```

No action-pipeline, normalization, ordering, or evaluator defect was found.

## C. TINY fixtures and P0 locks

```text
TINY1_H2STATS: one ordered red_target_train_a snapshot plus exact Stage22 frozen stats
TINY1_TINYSTATS: same one snapshot plus regenerated one-fixture stats
TINY6_TINYSTATS: all six ordered red_target_train_a snapshots plus regenerated six-fixture stats
```

All variants were built from the train-only Stage22 materialization. Desired
labels were preserved; achieved EEF was not used as a training label; holdout
`e` and `f` were not read. The Stage22 canonical source digest was recomputed
with 85 files, 919661 bytes, zero symlinks, and SHA256
`2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789`.
The former `c371b071...` value is a separate Stage23 local-manifest hash.

The primary future O1 input is frozen as `TINY1_H2STATS`; the tiny-stat
variants remain retained diagnostics and are not interchangeable with O1.

## D. Label-as-prediction sanity

```text
TINY1_H2STATS: 1/1 identity pass, 0 m position error, 0 rad rotation error
TINY1_TINYSTATS: 1/1 identity pass, 0 m position error, 0 rad rotation error
TINY6_TINYSTATS: 6/6 identity pass, 0 m position error, 0 rad rotation error
all identity SafetyGate decisions: ACCEPT
all identity profile ratios: exactly 1.0 within the audited tolerance
```

This closes the evaluator-or-action-contract sanity gate. It does not imply
that the base model or a fine-tuned model has learned the task.

## E. Official GR00T path

```text
Stage10 runtime Python: 3.12.3
GR00T SHA: 51d4c89f72fda44cbf77285c6a8114b52676b8a1
Cosmos revision: 9ce19a195e423419c349abfc86fd07178b230561
processor resolution: local_files_only=true, offline=true
TINY1_H2STATS: LeRobotEpisodeLoader -> ShardedSingleStepDataset -> processor PASS
TINY1_TINYSTATS: LeRobotEpisodeLoader -> ShardedSingleStepDataset -> processor PASS
TINY6_TINYSTATS: LeRobotEpisodeLoader -> ShardedSingleStepDataset -> processor PASS
```

All paths produced finite `[1,40,132]` universal actions and finite
`[1,40,10]` decoded physical actions. Official model forward was false.

## F. Execution boundary

```text
Stage23 Phase A O1 training/fine-tuning: EXECUTED; each fresh BASE -> N run stopped at its requested maximum
Phase A BASE-only inference: EXECUTED in the sealed Phase A evidence
Phase A GPU workload: EXECUTED on one H200 allocation
Phase A PBS jobs: 597960, 597965, 597966 (completed evidence in the bundle)
Phase A checkpoint: evidence/metadata retained; full weights were not included in the Phase A relay
Phase B training/inference: NOT EXECUTED; read-only attribution audit only
WBC: NOT EXECUTED
hardware/ROS/ZMQ/P1M: NOT EXECUTED
```

The Phase A O1 runs are negative learnability evidence, not a production
checkpoint promotion. Phase B did not create or modify a checkpoint. The
executed CPU evidence includes six direct contract checks. CPU/static pytest
tests are committed for CI; pytest was not installed in the local runtime, so
this stage used source compilation and direct audit assertions instead.

## G. Scientific decision

The action pipeline is sufficiently specified for a future learnability test.
The Stage22 profile blocker cannot be attributed to the audited 10D/132D
representation, relative transform, normalization selection, timestamp order,
or padding loss mask based on this diagnostic.

Phase A already executed the deliberate-overfit candidates and closed them at
their requested maximum steps; the Phase B audit did not start a new run:

```text
O1 TINY1_H2STATS deliberate overfit: EXECUTED; 10/25/50/100/200 each reached its maximum, profile failed
secondary TINY1_TINYSTATS diagnostic: retained fixture; not an additional Phase A training run
O6 TINY6_TINYSTATS conditional diagnostic: not executed
Phase B new training/inference: NOT EXECUTED
```

The Phase A O1 result is a negative diagnostic, not a claim that the
representation is untrainable in general. The next gate is PI review of the
read-only Phase C fixed-noise teacher-forced flow-field protocol and separate
explicit authorization of one future C1 run. No C1 or C2 execution is
included in this closeout.

Detailed evidence is in:

```text
reports/STAGE23_ACTION_PIPELINE_AUDIT.md
reports/STAGE23_TINY_DATASET_DIAGNOSTIC.md
reports/STAGE23_HOPPER_OVERFIT_PROTOCOL.md
```

## H. Phase B attribution closure

The immutable Phase A evidence bundle was independently re-verified before
interpretation:

```text
archive SHA256: 720aca9a1a1c7146919620df4000a76cc228a71729b057a2a0d155787ea50dcd PASS
payload manifest SHA256: dfeb6b44410b0042e53bac261e49acf7dbaa23ee2a61839ab65d4171b6c42b6a PASS
payload files: 252/252 verified
credentials absent by sealed declarations: PASS
new training/inference in Phase B: NONE
```

Phase B reconstructed five independent fresh `BASE -> N` experiments at
`N={10,25,50,100,200}`. Each reached its exact requested global step and has
the expected 10/25/50/100/200 loss-log entries. Direct optimizer.step callback:
`NOT_AVAILABLE_IN_PHASE_A_BUNDLE`. Optimizer completion is supported by the
trainer state, logs, and sealed PASS manifests rather than by a callback trace.
The H200 runs used about 30.8 GiB peak allocated and 32.0 GiB peak
reserved out of about 143.2 GiB; cleanup left about 65 MiB. Resource/OOM is
therefore ruled out for this result.

The official Stage10 processor-only probe passed on both materialized fixtures
without model construction, forward, GPU use, or training. Under H2 relative
statistics, 243/360 q01/q99 cells are zero-width by the official `np.isclose`
rule, while the one-episode regenerated tiny statistics are degenerate at
360/360 zero-width cells and normalize all 400 active values to zero. The
second result is retained as a diagnostic warning and is not used as a causal
explanation for H2.

The sealed raw `[1,40,132]` and decoded `[1,40,10]` artifacts were independently
profiled without smoothing, clipping, retiming, or correction. Recomputed
profiles matched the sealed metrics for BASE and O1 at 10/25/50/100/200 steps.
At O1@200, the median path length is `3.1368 m`, approximately `25.57x` the
expert path; median adjacent direction cosine is `-0.5493`, back-and-forth
rate is `0.7632`, and high-frequency first-difference DFT fraction is
`0.8436`. All three O1@200 predictions were finite but received
`SafetyGate=REJECT` for the declared test-only linear-speed bound. These
measurements strongly support a free-running temporal oscillation/profile
failure, while they do not by themselves prove a single causal mechanism.

The conservative Phase B scientific classification is:

```text
OBJECTIVE_NOT_DEMONSTRABLY_FIT
FREE_RUNNING_TEMPORAL_PROFILE_FAILURE_STRONGLY_SUPPORTED
PARAMETER_DELTA_NUMERIC_AUDIT_NOT_AVAILABLE_FROM_PHASE_A_BUNDLE
```

The 132D inactive-noise burden is mechanistically plausible (analytic
inactive/active noise-only energy ratio `12.2x`) but not proven causal. The
four-step Euler sampler, optimizer/LR schedule, and data scarcity remain
unresolved hypotheses. No additional Hopper work, fixed-noise diagnostic,
sampler-step comparison, WBC execution, or closed-loop run was performed or
authorized by Phase B.

The full machine-readable audit and reproducible CPU audit companion are:

```text
reports/STAGE23_PHASE_B_ATTRIBUTION_AUDIT.json
reports/STAGE23_PHASE_B_ATTRIBUTION_AUDIT.md
scripts/stage23_phase_b_attribution_audit.py
```

## I. Phase C preparation (not executed)

The Phase C harness and protocol are prepared only:

```text
scripts/stage23_phase_c_flow_field_diagnostic.py
reports/STAGE23_PHASE_C_PROTOCOL.md
docs/adr/ADR-027-fixed-noise-teacher-forced-flow-diagnostic.md
```

The frozen future C1 design compares BASE and O1@200 on the same TINY1
training sample, with identical full `[1,40,132]` noise and identical fixed
time probes. It performs no model construction, forward pass, inference,
training, sampler sweep, WBC execution, or holdout access in this preparation
stage. C2 sampler comparison remains locked until C1 passes and the mirrored
four-step sampler first matches the official implementation.
