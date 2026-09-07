# Stage40 RTX 4080 SUPER Engineering Mirror

Classification: `NON_SCIENTIFIC_DEVELOPMENT_EVIDENCE`

This report records the bounded RTX engineering bring-up under
`PI_GATE_STATE_0054` (Issue #11 comment `5566238390`). The branch is an
independent mirror of accepted D1 implementation `83779cbf...`; it does not
modify the H200 science lineage.

## Current disposition

`BLOCKED` at the frozen-input provenance boundary.

The RTX host has a verified native WBC runtime and R2 passed, but the exact
`STRUCTURED@1000` checkpoint and several exact processor inputs are not
present on the RTX host. BASE is recorded as reference-only and was not used
as a substitute. R1, R3, R4, and R5 therefore remain locked.

## Stage status

| Stage | Status | Evidence boundary |
| --- | --- | --- |
| R0 inventory | `PASS_WITH_PROVENANCE_BLOCKER` | Two RTX cards visible; existing processes preserved; exact WBC archive/tree verified; exact GR00T source worktree verified |
| R1 GR00T Q0 | `BLOCKED_PROVENANCE` | No model construction or forward; exact checkpoint/stats/TINY1/Cosmos inputs absent |
| R2 native-only | `PASS_ENGINEERING` | Native constructor and camera/state smoke passed on target UUID; no WBC forward |
| R3 dual runtime | `LOCKED` | Requires exact R1 model/input path |
| R4 C0 semantic dev | `LOCKED` | Requires exact R1 Q0 |
| R5 one handoff | `LOCKED` | Requires R4 and is non-scientific only |

R2 result: body and wrist RGB were both `[96,96,3]`, StateV1 was finite
`[10]`, both observations came from one simulator instance with unchanged
robot state, and `wbc_forward_count=0`. The result is a native runtime
engineering smoke, not a Stage40 qualification.

## Resource and runtime facts

- Host: `corelab-4080s`, Ubuntu 24.04.4, kernel `7.0.0-28-generic`.
- GPU 0 UUID `GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad`, 32760 MiB reported.
- GPU 1 UUID `GPU-845348f1-1511-dc19-0877-cc97658e3edd`, 16376 MiB reported;
  this was the R2 target.
- Both cards were treated as available for engineering, and pre-existing
  processes were allowed to coexist. No other-user process was killed,
  signaled, reprioritized, or modified.
- Driver `580.173.02`, host CUDA `13.0`, compute capability `8.9`.
- Native runtime: Python `3.8.20`, Torch `2.3.1`, CUDA `12.1`,
  torchvision `0.18.1`, torchaudio `2.3.1`; IsaacGym import passed.
- R2 target-card memory was 6253 MiB at baseline and 9009 MiB at native-only
  peak; the native process peak sampled by `nvidia-smi` was 2746 MiB. These
  numbers include the pre-existing target-card process and are not a
  dual-runtime capacity claim.
- All GPU-architecture-sensitive caches used by this work were placed under
  the RTX-specific runtime root. No Hopper/H200 cache was copied or reused.

## Provenance state

Accepted immutable identities:

```text
GR00T source:             51d4c89f72fda44cbf77285c6a8114b52676b8a1
STRUCTURED@1000 tree:     30b14d36e47e08dfbb9d3b353d199971863d6e7f6ceedea2d478ef367d0a1137
Structured stats:         0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206
WBC archive:              b52a53bef87340e6244ead352fa434390faee65246cd32ab8e6648336a018167
WBC extracted tree:       63311df752c9aaa5289287849635309e558f79284c91b80ddf07ec5638992567
representation:           StructuredActionV1 K16/T16 H40 D132
T_chunk:                  1.9500000000000002 s
frame/TCP:                NATIVE_WBC_RELATIVE_REFERENCE / x5_link6_gripper_center
handoff target:           0.9750000000000001 s
```

The following exact inputs were checked at the RTX host and are absent:

```text
/scratch/e1192548/leon-storage/runs/stage38_structured_groot_offline_20260906/checkpoint-1000
/scratch/e1192548/leon-storage/runs/stage38_structured_groot_offline_20260906/stage38d_train_a7f6880/experiment_cfg/structured_action_statistics.json
/scratch/e1192548/stage23_o1_20260903/input_exact/extracted/tiny1_h2stats_lerobot_v2
/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/modality/go2_arx5_modality.py
/scratch/e1192548/stage22_h200_cosmos_v2_20260902/hf_home/hub
```

The available Stage10 BASE checkpoint and its provisional modality file
remain reference/provisional artifacts only. No BASE inference, checkpoint
conversion, precision change, quantization, or processor substitution was
performed.

## Engineering fixes identified

1. The native launcher must prepend the native Python environment's `bin/`
   directory so Torch can find `ninja`.
2. The native runtime's `native_py38_4080` site-packages contains an old
   `typing.py`; a bootstrap must import the Python standard-library `typing`
   before adding that site-packages path so `yapf` can be loaded without
   changing delivered WBC source.
3. The Stage13 clean-room `pytorch3d` compatibility shim is required by the
   delivered WBC imports and is kept outside the H200 lineage.
4. The RTX launcher must bind a GPU UUID and use an RTX-only cache root.

The model-free artifact preflight implementing the provenance gate is
`scripts/stage40_rtx4080s_preflight.py`, with deterministic standard-library
tests in `tests/test_stage40_rtx4080s_preflight.py`.

## Claim boundary and next gate

```yaml
FORMAL_H200_SCIENCE_MODIFIED: false
SCIENTIFIC_CLAIM_PROMOTED: false
TRAINING: false
VALIDATION_E_READ: false
HOLDOUT_F_READ: false
HARDWARE_ROBOT: false
GR00T_FORWARD_COUNT: 0
WBC_FORWARD_COUNT: 0
SCIENTIFIC_ROWS: 0
```

The current terminal result is `BLOCKED` rather than
`ENGINEERING_READY_ON_RTX`: exact artifact identity is a hard stop. The next
action is to obtain a verified locator/mount for all frozen R1 inputs, then
rerun the model-free preflight before constructing GR00T. The requested gate
is `GPT_PI_REVIEW_AFTER_STAGE40_RTX4080S_ENGINEERING_BRINGUP`.

## PI_GATE_STATE_0056 addendum — Corelab prefix recovery and current provenance

This addendum preserves the historical R0 record above and binds the current
Corelab work to Issue #11 comment `5569559549`.

Current branch and source:

```text
branch: stage40-dev/rtx4080s-bringup
HEAD:   1ff9f8791164b97cc134550e71a142ec20ae4b9f
prefix patch: 44cec282720eeb674e3399bdf4242abd60de27df
preflight patch: 1ff9f8791164b97cc134550e71a142ec20ae4b9f
```

The Stage40-local lifecycle correction leaves
`RecedingHorizonExecutorV1` unchanged. It computes
`round(0.9750000000000001 / native_dt)` and passes that actual native control
step count as the only Stage40-allowed prefix. A dynamic CPU regression
instantiates the actual executor, submits the initial chunk and exactly one
update, checks the prefix count, handoff target, ordering, and all reset flags.

Validation on `corelab-4080s`:

```text
14 standard-library/unit tests: PASS
dynamic Stage40 prefix/lifecycle regression: PASS
py_compile: PASS
compileall: PASS
PBS shell syntax: PASS
git diff --check: PASS
```

The model-free preflight result is:

```text
path: /var/cache/users/liuyu-nfs/go2-arx5-vla-stage40-rtx4080s/R0_preflight_0056_artifact_closure.json
sha256: b6d965705979133533b6d5e4322497c01622ca3105353fad2c04b553c6fd7b0b
status: BLOCKED_PROVENANCE
```

Exact inputs currently bound on Corelab:

```text
GR00T source: PASS, 51d4c89f72fda44cbf77285c6a8114b52676b8a1
TINY1 tree: PASS, 541bf4a1b9ea459637525ef11fceea5c7f192c0902f6013f0331a2a668038157
TINY1 stats: PASS, fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23
TINY1 relative stats: PASS, 8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1
Stage22 modality bytes: PASS by expected SHA e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031
WBC archive: PASS, b52a53bef87340e6244ead352fa434390faee65246cd32ab8e6648336a018167
```

Still missing or not accepted:

```text
STRUCTURED@1000 checkpoint tree: expected 30b14d36e47e08dfbb9d3b353d199971863d6e7f6ceedea2d478ef367d0a1137
structured_action_statistics.json: expected 0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206
materialized Cosmos cache: expected revision 9ce19a195e423419c349abfc86fd07178b230561 and tree 1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9
```

An exact Cosmos archive is present, but archive-only is deliberately not
accepted as a runnable cache; no extraction or model read was performed.
BASE remains reference-only and was not substituted.

```yaml
MODEL_CONSTRUCTED: false
FORWARD: false
GPU_WORKLOAD: false
PBS_SUBMITTED: false
TRAINING: false
VALIDATION_E_READ: false
HOLDOUT_F_READ: false
HARDWARE_ROBOT: false
SCIENTIFIC_CONTRACT_CHANGED: false
SCIENTIFIC_CLAIM_PROMOTED: false
```

Current result remains `BLOCKED_PROVENANCE`; R1/R3/R4/R5 are locked until the
three missing exact inputs are available and pass the same preflight.
