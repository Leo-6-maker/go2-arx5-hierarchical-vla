# Stage22 H2 Authoritative Audit

Date: 2026-09-03 (Asia/Singapore)

## Final audit state

```text
H2_AUDIT_PASS_EXCEPT_LIVE_MODULE_INVENTORY
```

This is a read-only audit of the immutable relay for the already completed
H2 direct-launch experiment. It does not rerun training or inference, query a
current GPU, read `holdout_f`, transfer a checkpoint, execute WBC, or control
hardware.

Machine-readable audit output (node-local, not committed) is
`/var/cache/users/liuyu-nfs/go2-arx5-vla-stage22/authoritative_h2_audit_20260903_v3/audit.json`.

The only open gate is the strict live role-level
`STAGE22_TRAINABLE_MODULE_INVENTORY_V1` artifact. All other authoritative
identity, scope, provenance, profile-semantic, completeness, and comparison
gates passed.

## Evidence intake and scope

| Item | Verified value |
|---|---|
| Relay archive | `77604a2dc2f9c436718139b950209943cf4c767fc7f0ae52e624f28a3c877661` |
| Relay payloads | 89; every `SHA256SUMS` entry matched |
| H2 training job | `597351.hopper-m-02`, exit 0 |
| Profile continuation | `597361.hopper-m-02`, profile-only, exit 0 |
| Failed module probe | `597644`; offline Cosmos cache failure before model construction |
| `holdout_f_included` | `false` |
| `holdout_f_accessed` | `false` |
| `contains_credentials` | `false` |
| checkpoint weights in relay | `false` |

The failed `597644` probe recorded no model construction, forward, backward,
optimizer, inference, or training. It is therefore infrastructure-only
evidence and is not counted as a failed H2 training run.

## Frozen scientific identity

The direct H2 manifests and immutable source artifacts matched the frozen
identity:

```text
training source commit:   deb920a070141e6a32a88cd0da6719828647e1e6
training launcher SHA256:  f42a1990847cf8f5326c66db37043ffed6e43458dd907fdbd08da5dd905b00a2
GR00T SHA:                 51d4c89f72fda44cbf77285c6a8114b52676b8a1
base tree SHA256:          8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4
train fixture tree SHA256: 2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789
split lock SHA256:         7e124f29d2e4d24c17fd1fefcf8753ececd3a8cd86e8c6a2f159e2e5d2217246
modality SHA256:           e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031
Cosmos revision:           9ce19a195e423419c349abfc86fd07178b230561
Cosmos tree SHA256:        1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9
```

R0 and R1 were direct launches of the frozen Stage20S launcher, with matched
base, train fixture, runtime, seed `2222`, 50 optimizer steps, learning rate
`1e-4`, H=40, and `0.05 s` `TEST_ONLY_SIM_VISUAL_REACH_V0` timing. R0 trained
projector/state/action modules with diffusion frozen; R1 additionally trained
the diffusion/action module. No loss-based checkpoint selection was done.

## Historical GPU provenance

```text
H2_GPU_PROVENANCE_SUPPORTED_WITH_FORMAT_DEVIATION
```

The claim is recovered exclusively from immutable H2 artifacts:

```text
GPU:       NVIDIA H200
UUID:      GPU-39c00fb9-0eea-3abd-2727-2027667f3f44
host:      hopper-12
PBS job:   597351.hopper-m-02
timestamp: 2026-09-02T20:11:29+08:00
```

The source PBS template and launcher output document the historical
five-column query including `memory.used`. The explicitly labelled relay
projection preserved only the four identity columns and dropped that extra
field. Its source-log binding, projection hash, job/host, and PBS accounting
window all passed. The audit issued no present-time `nvidia-smi` query.

The source display reports `143771 MiB`; the R0/R1 PyTorch telemetry reports
the same runtime total of `150110273536` bytes. This retained display-unit /
device-reporting difference is not used to manufacture a new GPU fact; R0/R1
internal consistency and the immutable identity binding are the gates.

## Profile semantic and snapshot audit

The evaluator continuation is classified:

```text
H2_PROFILE_EVALUATOR_SEMANTICS_FROZEN
```

The initial `597361` attempt failed before profile output because of a modality
symbol lookup. The continuation changed only the compatibility path/symbol;
the frozen model input, StateV1, labels, H=40 decode, inference seeds, timing,
target exclusion, no-correction policy, metrics, and SafetyGate remained
unchanged. The current repository evaluator differs from the immutable bundle
only by additive snapshot-lock support.

The independent audit recovered exactly nine JSON payloads embedded in the
immutable logs:

```text
BASE/R0/R1 × inference seeds 2223/2224/2225
6 exact locked red_target_holdout_e snapshots per seed
18 records and 15 overlap transitions per recipe
universal shape [1,40,132]
decoded physical shape [1,40,10]
```

The raw payloads predate the snapshot-lock hash field. The audit rebound their
exact six IDs to the tracked lock
`manifests/STAGE22_VALIDATION_SNAPSHOT_LOCK_V1.json` (file SHA256
`23c35e2204b9555c8431a3f86fd7e1767740bf7196341972e847bc43c141aec8`), rather
than relying on R0/R1 set equality alone.

## Independent profile recomputation

All values below are recomputed from the nine raw log-embedded payloads. The
table reports median / p95 / max across the 18 recipe records, except overlap
quantities, which use the 15 overlap rows. Position and endpoint values are in
metres; speed/tangent values are in metres per second; ratios are dimensionless.

| recipe | mean position error | endpoint position error | path-length ratio | mean-speed ratio | max-speed ratio | roughness ratio | overlap p95 position | overlap p95 tangent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE | 0.088798 / 0.111617 / 0.113451 | 0.175516 / 0.251656 / 0.256585 | 15.007 / 21.406 / 23.354 | 15.007 / 21.406 / 23.354 | 32.759 / 51.290 / 53.514 | 348.860 / 499.985 / 545.534 | 0.190152 / 0.234446 / 0.239665 | 6.913 / 7.072 / 7.225 |
| R0 | 0.081892 / 0.104805 / 0.107974 | 0.156097 / 0.224237 / 0.229816 | 14.486 / 20.431 / 22.299 | 14.486 / 20.431 / 22.299 | 32.320 / 49.524 / 51.261 | 330.814 / 475.620 / 515.128 | 0.199327 / 0.229489 / 0.233321 | 6.676 / 7.075 / 7.125 |
| R1 | 0.075335 / 0.098075 / 0.101194 | 0.139669 / 0.197418 / 0.202736 | 14.240 / 20.646 / 22.008 | 14.240 / 20.646 / 22.008 | 32.036 / 48.887 / 51.474 | 326.833 / 479.400 / 501.501 | 0.205834 / 0.225664 / 0.231916 | 6.673 / 7.332 / 7.397 |

SafetyGate decisions were `0 ACCEPT / 0 WARN / 18 REJECT` for each recipe.
These are offline profile-gate decisions under the frozen TEST_ONLY 2.0 m/s
criterion; they are not WBC execution failures.

Rotation error was identical across BASE/R0/R1: median
`2.5438e-08 rad`, p95/max `2.9802e-08 rad`. Initial position
discontinuity was zero for all 54 records; initial rotation discontinuity was
at the same numerical level. The large discrepancy is therefore translational
future-profile geometry, not an initial SE(3) jump.

The visual-conditioning field in each recipe remained
`INSUFFICIENT_COUNTERFACTUALS` with count 1. No visual-grounding claim is
made from this H2 audit.

### Required attribution deltas

Each row is a paired right-minus-left delta over the same snapshot and
inference seed. Pair counts are 18 records and 15 overlaps for every
comparison. Negative error deltas indicate lower error for the right-hand
recipe.

| comparison | median Δ mean position | median Δ endpoint position | median Δ path ratio | median Δ max-speed ratio | median Δ roughness | median Δ overlap p95 position | median Δ overlap tangent |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASE → R0 | -0.007463 m | -0.022855 m | +0.0446 | +0.3383 | +0.8846 | +0.001118 m | -0.0773 m/s |
| BASE → R1 | -0.012476 m | -0.047727 m | -0.2219 | +0.4318 | -5.7974 | -0.001139 m | -0.1360 m/s |
| R0 → R1 | -0.006181 m | -0.024148 m | -0.2139 | -0.0170 | -6.6820 | -0.000764 m | +0.0034 m/s |

R1 improves median positional and endpoint error relative to both BASE and R0,
but it does not close the action-profile blocker: path-length ratios remain
about 14x, max-speed ratios about 32x, overlap p95 position error is about
0.206 m, overlap tangent error is about 6.67 m/s, and all 18 R1 rows remain
SafetyGate rejects. The result is therefore:

```text
DIFFUSION_ACTION_TUNING_DID_NOT_CLOSE_PROFILE_BLOCKER
```

This conclusion is limited to the tested 50-step, held-out `e` simulation
distribution. It does not select a checkpoint, authorize R1@100/R1@200, or
claim task success.

## Module inventory gate

The relay includes a broad, internally recomputed per-parameter inventory with
1030 rows and actual `requires_grad` booleans. It supports the observed counts:

```text
R0 trainable: 327,356,544
R1 trainable: 1,419,078,784
```

However, the relay does not include the frozen seven-role
`STAGE22_TRAINABLE_MODULE_INVENTORY_V1` with role coverage/digests. The audit
does not infer that role inventory from grouped names. One bounded corrected
HF-cache live probe may be considered later only if the PI authorizes it and
only while this remains the sole open gate.

## Stop state

```text
H2_AUTHORITATIVE_AUDIT: H2_AUDIT_PASS_EXCEPT_LIVE_MODULE_INVENTORY
R1@100: NOT AUTHORIZED
R1@200: NOT AUTHORIZED
HOLDOUT_F_ACCESSED: false
CHECKPOINT_SELECTION: NOT PERFORMED
CHECKPOINT_TRANSFER: NOT PERFORMED
WBC: NOT AUTHORIZED
```
