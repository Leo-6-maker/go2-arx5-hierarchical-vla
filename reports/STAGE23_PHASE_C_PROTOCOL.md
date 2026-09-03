# Stage23 Phase C Fixed-Noise Flow-Field Diagnostic Protocol

Date: 2026-09-03 (Asia/Singapore)

## Status and execution boundary

```text
PREPARED_NOT_EXECUTED
```

This document freezes a future diagnostic and records no new model evidence.
The preparation phase performed no model construction, model forward pass,
inference, training, optimizer step, PBS submission, WBC execution, or access
to validation target `e` or locked final target `f`. It also does not alter the
inactive 122 universal-action dimensions.

The machine-readable plan and static harness are:

```text
scripts/stage23_phase_c_flow_field_diagnostic.py
STAGE23_PHASE_C_FLOW_FIELD_DIAGNOSTIC_PROTOCOL_V1
```

## Frozen C1 inputs

The future C1 run, only after separate explicit authorization, compares:

```text
BASE checkpoint
O1@200 checkpoint
```

on exactly one fixed sample:

```text
dataset: TINY1_H2STATS
sample: first ordered snapshot in red_target_train_a
validation target e: not read
locked final target f: not read
```

For every probe, BASE and O1@200 must receive the same full noise tensor and
the same scalar time value. The model runner must report the raw predicted
velocity before any physical decode. The harness is intentionally not a
GR00T model wrapper; this keeps model construction and execution outside the
preparation path.

## Official flow equations

The protocol follows the audited GR00T N1.7 source:

```text
B ~ Beta(alpha=1.5, beta=1.0)
t = (1 - B) * noise_s, where noise_s = 0.999
x_t = (1 - t) * noise + t * expert_action
v_target = expert_action - noise
t_discretized = (t * 1000).long()
```

For the official `Beta(1.5, 1.0)` distribution, the exact quantile of the
actual transformed training-time variable `t` is:

```text
t_q = noise_s * (1 - (1-q) ** (1/alpha))
```

This is a quantile of `t` after the source transformation, not the q-quantile
of the untransformed Beta sample. The fixed values are:

| requested quantile of `t` | Beta sample inducing that quantile | `t` | `t_discretized` |
|---:|---:|---:|---:|
| 0.10 | 0.9321697517861577 | 0.06776241796562847 | 67 |
| 0.50 | 0.6299605249474366 | 0.36966943557751086 | 369 |
| 0.90 | 0.21544346900318836 | 0.7837719744658148 | 783 |

The `t_discretized` values use the same positive-value truncation semantics as
the source `.long()` operation.

## Nine fixed probes

The grid is the Cartesian product:

```text
noise seeds: 2223, 2224, 2225
t quantiles: 0.10, 0.50, 0.90
```

This produces nine probes. Noise is generated as a full
`[1,40,132]` tensor using a CPU Torch generator seeded by the recorded noise
seed, then transferred to the action device. The same generated tensor and
the same `t` must be reused for both model conditions.

The universal action is:

```text
expert_action: [1,40,132]
noise:         [1,40,132]
pred_velocity: [1,40,132]
action_hat:    noise + pred_velocity       [1,40,132]
active_hat:    action_hat[..., :10]        [1,40,10] model-space slice
physical_hat:  official unapply_action(active_hat) [1,40,10]
```

The first ten dimensions are the physical EEF-plus-gripper action. The
remaining 122 dimensions remain present in the noisy universal input, but
the audited loss mask is zero there and Phase C must not intervene in those
dimensions. The active slice is still in the official normalized model action
space; the existing processor/provider boundary must perform unnormalization,
relative-to-absolute conversion, physical SO(3) decoding, and gripper decode
before profile evaluation. No slice, decode, or trajectory may be silently
corrected.

## C1 measurements and decision rule

Each probe must record:

```text
active masked flow MSE
active and inactive prediction/target energy
finite raw velocity and implied action
decoded [1,40,10] profile
position and rotation profile metrics
path-length ratio
mean-speed ratio
max-speed ratio
roughness
initial discontinuity
TrajectorySafetyGate decision
```

The future C1 interpretation is deliberately narrow:

```text
all fixed probes satisfy the declared local-field checks
    -> LOCAL_FLOW_OBJECTIVE_FIT_SUPPORTED
otherwise
    -> LOCAL_FLOW_OBJECTIVE_FIT_NOT_SUPPORTED
```

Either result is a learnability diagnostic only. C1 does not authorize WBC or
claim task performance.

## C2 lock and sampler parity gate

C2 is locked until C1 passes. Its reserved sampler candidates are:

```text
4, 8, 16, 32 inference steps
```

Before comparing step counts, a mirrored four-step sampler must match the
official four-step implementation on the same frozen inputs. A mismatch is a
sampler-parity blocker, not evidence about the model profile. No smoothing,
clipping, retiming, resampling, or trajectory correction is allowed.

## Provenance required for any future execution

Any separately authorized C1/C2 run must seal:

```text
TINY1_H2STATS tree/hash and selected snapshot hash
BASE checkpoint tree/hash and O1@200 checkpoint tree/hash
GR00T source SHA
Cosmos revision/tree hash
official modality/processor configuration hash
noise seed and CPU-generator construction
t quantile, transformed t, and t_discretized
model runner output hash before any physical decode
decoded [1,40,10] hash and all unmodified evaluator outputs
```

No Phase C artifact may contain credentials or a copied private runtime.
