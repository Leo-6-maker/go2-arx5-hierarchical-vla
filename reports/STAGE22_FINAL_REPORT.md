# Stage22 Final Report

Date: 2026-09-03 (Asia/Singapore)

## A. Classification

```text
PASS WITH DEVIATION
```

Stage22 completed the simulation-only expert/data path, H200 R1 one-step
qualification, direct R0/R1@50 training, and the immutable H2 authoritative
audit. The remaining deviation is the missing strict live role-level
`requires_grad` inventory. The action-profile blocker was not closed by R1@50.

## B. Completed gates

```text
same-scene expert oracle:                 6/6 PASS
native steps:                             144/144
replanning snapshots:                     36
official loader/processor:                PASS
CPU regression before H2 audit:           76 passed
H200 R1 one-step optimizer preflight:     PASS
H2 immutable relay audit:                 PASS_EXCEPT_LIVE_MODULE_INVENTORY
```

The relay contained 89 hashed payloads and no credentials, checkpoint weights,
`holdout_f`, WBC, hardware, ROS, ZMQ, or PolicyServer artifacts.

## C. Frozen H2 identity and provenance

R0 and R1 used the direct frozen Stage20S launcher at source commit
`deb920a070141e6a32a88cd0da6719828647e1e6`, GR00T
`51d4c89f72fda44cbf77285c6a8114b52676b8a1`, seed `2222`, 50 optimizer steps,
H=40, learning rate `1e-4`, and `0.05 s` TEST_ONLY timing. Base tree, train
fixture, split lock, modality, and Cosmos revision/tree matched their frozen
SHA256 identities.

Historical GPU provenance is:

```text
H2_GPU_PROVENANCE_SUPPORTED_WITH_FORMAT_DEVIATION
NVIDIA H200 / GPU-39c00fb9-0eea-3abd-2727-2027667f3f44
hopper-12 / 597351.hopper-m-02
```

The result is recovered from immutable log, PBS template, projection, and
accounting artifacts. No present-time GPU query was used. The projection
dropped only the extra `memory.used` column; the source and projection hashes
remain recorded in the authoritative audit.

## D. H2 profile result

The audit recovered BASE/R0/R1 × seeds `2223/2224/2225`, with six locked
`red_target_holdout_e` snapshots and five overlap transitions per seed. Each
recipe has 18 records and 15 overlaps, finite `[1,40,132]` universal output,
and `[1,40,10]` decoded output.

| recipe | median mean position error | median endpoint position error | median path ratio | median max-speed ratio | median overlap p95 position | SafetyGate |
|---|---:|---:|---:|---:|---:|---|
| BASE | 0.088798 m | 0.175516 m | 15.007 | 32.759 | 0.190152 m | 0/0/18 REJECT |
| R0 | 0.081892 m | 0.156097 m | 14.486 | 32.320 | 0.199327 m | 0/0/18 REJECT |
| R1 | 0.075335 m | 0.139669 m | 14.240 | 32.036 | 0.205834 m | 0/0/18 REJECT |

R1 improves positional and endpoint medians versus R0 and BASE, but retains
the excessive future-motion profile and overlap inconsistency. The scientific
result is:

```text
DIFFUSION_ACTION_TUNING_DID_NOT_CLOSE_PROFILE_BLOCKER
```

The complete median/p95/max summaries and paired `BASE → R0`, `BASE → R1`,
and `R0 → R1` deltas are in
`reports/STAGE22_H2_AUTHORITATIVE_AUDIT.md`.

## E. Audit exception

The relay contains a broad per-parameter `requires_grad` inventory with 1030
rows and internally consistent counts:

```text
R0: 327,356,544 trainable parameters
R1: 1,419,078,784 trainable parameters
```

It does not contain the separately frozen seven-role
`STAGE22_TRAINABLE_MODULE_INVENTORY_V1`. The audit does not infer that
role-level gate from grouped names. The failed `597644` probe was an offline
Cosmos-cache failure before model construction and did no compute.

## F. Stop and next gate

```text
H2_AUDIT_PASS_EXCEPT_LIVE_MODULE_INVENTORY
R1@100: NOT AUTHORIZED
R1@200: NOT AUTHORIZED
CHECKPOINT_SELECTION: NOT PERFORMED
CHECKPOINT_TRANSFER: NOT PERFORMED
HOLDOUT_F_ACCESSED: false
WBC: NOT AUTHORIZED
ASYNC_RUNTIME: NOT QUALIFIED
```

No actual-model rerun or WBC execution follows this audit. A single bounded
corrected-HF-cache live module-inventory probe can be considered only after
explicit PI authorization and only because it is the sole remaining audit
gate.
