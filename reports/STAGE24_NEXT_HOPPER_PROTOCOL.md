# Stage24 A6/A7 — Prepared Hopper Protocol (Not Executed)

## Single candidate

```text
H2_LARGER_STOCHASTIC_FLOW_EXPOSURE_SAME_SAMPLE_H40_OVERFIT
STATUS: PREPARED_NOT_EXECUTED
```

This protocol is a preparation artifact for PI review. It does not submit
PBS, allocate H200, construct a model, run inference, or train.

## Frozen inputs

```text
dataset: TINY1_H2STATS
split: red_target_train_a only
sample: the first ordered TINY1 snapshot
stats SHA256: fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23
relative stats SHA256: 8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1
base model tree SHA256: 8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4
GR00T SHA: 51d4c89f72fda44cbf77285c6a8114b52676b8a1
action horizon: 40
test-only action period: 0.05 s
training seed: 2222
inference seeds: 2223, 2224, 2225
```

No validation `e` or locked test `f` is read by this diagnostic.

## Proposed recipe

```text
batch size: 1
gradient accumulation: 1
proposed max steps: 2000
optimizer: AdamW
learning rate: 1e-4
schedule: cosine, warmup_ratio=0.05
tune_projector: true
tune_diffusion_model: true
tune_visual: false
tune_llm: false
tune_vlln: false
num_inference_timesteps: 4
precision: bf16
```

The only intended scientific change is increased same-sample exposure. The
longer cosine schedule is a coupled recipe consequence and must be recorded,
not hidden. No smoothing, clipping, retiming, loss change, action-ABI change,
state-ABI change, visual/LLM tuning, or inactive-channel intervention is
allowed under this candidate.

## Resource and execution gate

If later authorized, use one uncontended compatible H200 allocation with
offline/local-file-only dependencies. Before the run, verify the exact base
tree, dataset tree, statistics, modality, GR00T revision, and Cosmos revision.
Record GPU UUID, runtime, peak allocated/reserved memory, step time, cleanup,
and the complete trainable-module inventory. Stop on any identity mismatch or
resource/provenance ambiguity. This document does not grant that future
authorization.

## Required evaluation

For each of seeds `2223/2224/2225`, audit the unchanged H40 output and report:

```text
finite values
SafetyGate decision
trajectory position/rotation error
endpoint error
path-length ratio
mean-speed ratio
max-speed ratio
roughness
initial discontinuity
```

The result must be compared with the sealed BASE/O1 profile evidence. A
healthy teacher-forced or decoded target would not by itself prove healthy
free-running behavior. If the same oscillation persists at the prepared
exposure, stop and classify exposure-only tuning as insufficient; do not
silently move to H1/H3/H4 in this protocol.

## Prohibited in Stage24

```text
Hopper/PBS submission: false
new model forward/inference: false
new training: false
WBC/native rollout: false
validation e / holdout f: false
hardware/ROS/ZMQ/P1M: false
```
